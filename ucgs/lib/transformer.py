import torch
import torch.nn as nn
from torch.nn.functional import softmax
from lib.building_block import linear


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()

        self.is_first = is_first

        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))

    def forward(self, inputs):
        """
        inputs: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        if self.is_first:
            inputs = self.attn_layer_norm(inputs)
            x = self.attn(inputs, inputs, inputs)
            inputs = inputs + x
        else:
            x = self.attn_layer_norm(inputs)
            x = self.attn(x, x, x)
            inputs = inputs + x

        x = self.ffn_layer_norm(inputs)
        x = self.ffn(x)
        return inputs + x


class TransformerEncoder(nn.Module):

    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()

        if num_blocks > 0:
            gain = (2 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        for block in self.blocks:
            inputs = block(inputs)

        return self.layer_norm(inputs)


class TransformerDecoderBlock(nn.Module):

    def __init__(self, max_len, d_model, num_heads, dropout=0.,
                 gain=1., is_first=False, triu=True, block=1):
        super().__init__()

        self.is_first = is_first
        self.triu = triu
        self.block = block

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        mask = mask.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)

        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))

    def forward(self, inputs, encoder_output):
        """
        inputs: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = inputs.shape[1]

        if self.is_first:
            inputs = self.self_attn_layer_norm(inputs)
            x = self.self_attn(inputs, inputs, inputs, self.self_attn_mask[:T, :T]) \
                if self.triu else self.self_attn(inputs, inputs, inputs)
            inputs = inputs + x
        else:
            x = self.self_attn_layer_norm(inputs)
            x = self.self_attn(x, x, x, self.self_attn_mask[:T, :T]) \
                if self.triu else self.self_attn(x, x, x)
            inputs = inputs + x

        x = self.encoder_decoder_attn_layer_norm(inputs)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        inputs = inputs + x

        x = self.ffn_layer_norm(inputs)
        x = self.ffn(x)
        return inputs + x


class TransformerDecoder(nn.Module):

    def __init__(self, num_blocks, max_len, d_model, num_heads,
                 dropout=0., triu=True, block=1):
        super().__init__()

        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain,
                                         is_first=True, triu=triu, block=block)] +
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain,
                                         is_first=False, triu=triu, block=block)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, encoder_output):
        """
        inputs: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            inputs = block(inputs, encoder_output)

        return self.layer_norm(inputs)


class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()
        self.is_first = is_first
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))

    def forward(self, q, k, v):
        if self.is_first:
            q = self.self_attn_layer_norm(q)
            x = self.self_attn(q, q, q)
            q = q + x
        else:
            x = self.self_attn_layer_norm(q)
            x = self.self_attn(x, x, x)
            q = q + x
        x = self.encoder_decoder_attn_layer_norm(q)
        x = self.encoder_decoder_attn(x, k, v)
        q = q + x
        x = self.ffn_layer_norm(q)
        x = self.ffn(x)
        return q + x


class CrossAttention(nn.Module):

    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()
        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [CrossAttentionBlock(d_model, num_heads, dropout, gain, is_first=True)] +
                [CrossAttentionBlock(d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()

    def forward(self, q, k, v):
        for block in self.blocks:
            q = block(q, k, v)
        return q
