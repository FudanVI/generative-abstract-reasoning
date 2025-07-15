import torch
import torch.nn as nn
from lib.pe import CartesianPositionalEmbedding, LearnedPositionalEmbedding1D
from lib.transformer import TransformerDecoder, linear


class ImageEncoder(nn.Module):

    def __init__(self, num_hidden, y_dim, image_size,
                 num_slot=8, num_layer=4, num_head=8, dropout=0.):
        super(ImageEncoder, self).__init__()
        self.pos = CartesianPositionalEmbedding(y_dim, image_size)
        self.layer_norm = nn.LayerNorm(y_dim)
        self.mlp = nn.Sequential(
            linear(y_dim, num_hidden, weight_init='kaiming'),
            nn.ReLU(), linear(num_hidden, num_hidden))
        self.tf = TransformerDecoder(
            num_layer, 1000, num_hidden, num_head, dropout=dropout, triu=False)
        self.cls_token = nn.Parameter(torch.Tensor(1, num_slot, num_hidden), requires_grad=True)
        nn.init.xavier_uniform_(self.cls_token)

    def forward(self, emb):
        b, n, h, w, d = emb.size()
        emb = emb.permute(0, 1, 4, 2, 3).flatten(0, 1)
        emb = self.pos(emb)  # b * n, d, h, w
        emb = emb.permute(0, 2, 3, 1).flatten(1, 2)  # b * n, h * w, d
        emb = self.mlp(self.layer_norm(emb))  # b * n, h * w, dh
        slots = self.cls_token.expand(emb.size(0), -1, -1)
        slots = self.tf(slots, emb).reshape(b, n, *slots.size()[1:])  # b, n, ns, ds
        return slots


class PanelEncoder(nn.Module):

    def __init__(self, num_hidden, x_dim, y_dim,
                 num_layer=4, num_head=8, dropout=0.):
        super(PanelEncoder, self).__init__()
        self.input_map = linear(num_hidden + x_dim, num_hidden)
        self.query_map = linear(x_dim, num_hidden)
        self.tf = TransformerDecoder(
            num_layer, 10, num_hidden, num_head, dropout=dropout, triu=False)

    def forward(self, context_x, context_y, target_x):
        b, n, ns, d = context_y.size()
        context_y = context_y.permute(0, 2, 1, 3).flatten(0, 1)
        context_x = context_x.unsqueeze(1).expand(-1, ns, -1, -1).flatten(0, 1)
        target_x = target_x.unsqueeze(1).expand(-1, ns, -1, -1).flatten(0, 1)
        hidden = torch.cat([context_x, context_y], dim=-1)
        hidden = self.input_map(hidden)
        target_x = self.query_map(target_x)
        output = self.tf(target_x, hidden)  # b * ns, nt, d
        output = output.reshape(b, ns, -1, d)
        output = output.permute(0, 2, 1, 3).contiguous()
        return output


class Decoder(nn.Module):

    def __init__(self, num_hidden, y_dim, slot_dim, backbone, n_embed, image_size,
                 num_layer=4, num_head=8, dropout=0.):
        super(Decoder, self).__init__()
        self.rule_map = linear(slot_dim, num_hidden)
        self.embed_map = linear(y_dim, num_hidden)
        self.backbone = backbone
        self.n_embed = n_embed
        self.bos = nn.Parameter(torch.Tensor(1, 1, num_hidden))
        nn.init.xavier_uniform_(self.bos)
        self.image_size = image_size
        self.pos = LearnedPositionalEmbedding1D(1 + image_size ** 2, num_hidden)
        self.tf = TransformerDecoder(
            num_layer, 4096, num_hidden, num_head, dropout=dropout, triu=True)
        self.head = linear(num_hidden, n_embed, bias=False)

    def forward(self, slots, target_y):
        b, n, h, w, d = target_y.size()
        slots = self.rule_map(slots).expand(-1, n, -1, -1).flatten(0, 1)
        target_y = self.embed_map(target_y).flatten(0, 1)
        target_y = target_y.reshape(b * n, h * w, -1)
        bos = self.bos.expand(target_y.size(0), -1, -1).to(slots.device)
        inputs = torch.cat([bos, target_y], dim=1)[:, :-1]
        inputs = self.pos(inputs)
        pred = self.tf(inputs, slots)
        pred = self.head(pred)
        pred = pred.reshape(b, n, h, w, -1)
        return pred

    def predict(self, slots):
        slots = self.rule_map(slots).squeeze(1)
        gen_len = self.image_size ** 2
        z_gen = slots.new_zeros(0).to(slots.device)
        dec_input = self.bos.expand(slots.size(0), -1, -1).to(slots.device)
        for t in range(gen_len):
            dec_input_pos = self.pos(dec_input)
            decoder_output = self.tf(dec_input_pos, slots)
            decoder_output = self.head(decoder_output)[:, -1]
            z_next = self.backbone.embed_code(decoder_output.argmax(dim=-1))
            z_gen = torch.cat((z_gen, decoder_output.unsqueeze(1)), dim=1)
            z_next = self.embed_map(z_next)
            dec_input = torch.cat((dec_input, z_next.unsqueeze(1)), dim=1)
        z_gen = z_gen.reshape(slots.size(0), 1, self.image_size, self.image_size, -1)
        return z_gen


class DeterHANP(nn.Module):

    def __init__(self, rule_size, x_dim, y_dim, backbone, n_embed, image_size,
                 image_layer=2, image_head=4, panel_layer=2, panel_head=4,
                 dec_layer=2, dec_head=4, dec_hidden=256, dropout=0., num_slot=32):
        nn.Module.__init__(self)
        self.image_encoder = ImageEncoder(
            rule_size, y_dim, image_size, num_layer=image_layer,
            num_slot=num_slot, num_head=image_head, dropout=dropout)
        self.panel_encoder = PanelEncoder(
            rule_size, x_dim, y_dim, num_layer=panel_layer,
            num_head=panel_head, dropout=dropout)
        self.decoder = Decoder(
            dec_hidden, y_dim, rule_size, backbone, n_embed, image_size,
            num_layer=dec_layer, num_head=dec_head, dropout=dropout)

    def forward(self, context_x, context_y, target_x, target_y):
        context_slots = self.image_encoder(context_y)  # b, nc, ns, ds
        target_slots = self.panel_encoder(context_x, context_slots, target_x)  # b, nt, ns, ds
        logits = self.decoder(target_slots, target_y)
        prior_mu = torch.zeros_like(logits)
        prior_std = torch.ones_like(logits)
        posterior_mu = torch.zeros_like(logits)
        posterior_std = torch.ones_like(logits)
        return prior_mu, prior_std, posterior_mu, posterior_std, logits

    def predict(self, context_x, context_y, target_x):
        context_slots = self.image_encoder(context_y)  # b, nc, ns, ds
        target_slots = self.panel_encoder(context_x, context_slots, target_x)  # b, nt, ns, ds
        logits = self.decoder.predict(target_slots)
        return logits

    def metric(self, context_x, context_y, target_x, target_dis):
        context_slots = self.image_encoder(context_y)  # b, nc, ns, ds
        target_slots = self.panel_encoder(context_x, context_slots, target_x)  # b, nt, ns, ds
        logits = self.decoder(target_slots, target_dis)
        return logits

    def predict_via_post(self, context_x, context_y, target_x, target_y):
        return self.predict(context_x, context_y, target_x)

    def kl_div(self, prior_mu, prior_std, posterior_mu, posterior_std):
        prior_var = prior_std.pow(2)
        posterior_var = posterior_std.pow(2)
        kl = (torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / torch.exp(prior_var) \
            - 1. + (prior_var - posterior_var)
        kl = 0.5 * kl
        return kl
