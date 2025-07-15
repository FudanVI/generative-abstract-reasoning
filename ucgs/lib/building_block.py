from abc import ABC
import math
import pickle
import torch.nn as nn
import torch
import lib.utils as utils
from torch.nn.functional import interpolate, softmax, relu
from sklearn.metrics import adjusted_rand_score


class BaseNetwork(ABC):
    def __init__(self, args, global_step=0):
        super(BaseNetwork, self).__init__()
        self.global_step = global_step
        self.anneal_settings = args.anneal if hasattr(args, 'anneal') else {}
        self.anneal()

    def anneal(self):
        for k, v in self.anneal_settings.items():
            setattr(self, k, utils.anneal_by_milestones(
                self.global_step, v['milestones'], v['value'], mode=v['mode']
            ))

    def update_hook(self):
        pass

    def update_step(self):
        self.global_step += 1
        if self.update_hook is not None:
            self.update_hook()
        self.anneal()
        for k in dir(self):
            v = getattr(self, k)
            if isinstance(v, BaseNetwork):
                v.update_step()


class ReshapeBlock(nn.Module):
    def __init__(self, size):
        super(ReshapeBlock, self).__init__()
        self.size = size

    def forward(self, x):
        shape = x.size()[:1] + tuple(self.size)
        return x.reshape(shape)


class FCResBlock(nn.Module):
    def __init__(self, hidden_size, norm='bn'):
        super(FCResBlock, self).__init__()
        self.hidden_size = hidden_size
        if norm is None:
            self.net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif norm is 'ln':
            self.net = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        else:
            self.net = nn.Sequential(
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

    def forward(self, x):
        dim = x.size(-1)
        assert dim == self.hidden_size
        h = x.reshape(-1, dim)
        h = h + self.net(h)
        return h.reshape(x.size())


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None,
                 activate=nn.ReLU(), last_activate=None, norm=None):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        if hidden_dim is None:
            hidden_dim = []
        hidden_dim = [input_dim] + hidden_dim + [output_dim]
        for index in range(len(hidden_dim) - 1):
            if index > 0:
                if norm is not None:
                    if norm.lower() == 'batch':
                        self.net.append(nn.BatchNorm1d(hidden_dim[index]))
                    elif norm.lower() == 'layer':
                        self.net.append(nn.LayerNorm(hidden_dim[index]))
                if activate is not None:
                    self.net.append(activate)
            self.net.append(nn.Linear(hidden_dim[index], hidden_dim[index + 1]))
        if last_activate is not None:
            self.net.append(last_activate)
        self.net.apply(utils.weights_init)

    def forward(self, x):
        batch_shape, d = x.size()[:-1], x.size(-1)
        h = x.reshape(-1, d)
        for layer in self.net:
            h = layer(h)
        return h.reshape(*batch_shape, -1)


def compute_ari(gt, pred):
    return torch.tensor([adjusted_rand_score(
        gt.reshape(-1).cpu().detach().numpy(),
        pred.reshape(-1).cpu().detach().numpy()
    )]).to(gt.device)


def update_dict(d, sub_d):
    for key, val in sub_d.items():
        if key not in d:
            d[key] = []
        d[key].append(val.item())
    return d


def copy_model(model):
    return pickle.loads(pickle.dumps(model))


def batch_interpolate(img, size, mode='nearest'):
    batch_shape, image_shape = img.size()[:-3], img.size()[-3:]
    img = img.reshape(-1, *image_shape)
    img = interpolate(img, size, mode=mode)
    img = img.reshape(*batch_shape, image_shape[0], size, size)
    return img


def linear_warmup(step, start_value, final_value, start_step, final_step):
    assert start_value <= final_value
    assert start_step <= final_step
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    return value


def cosine_anneal(step, start_value, final_value, start_step, final_step):
    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = softmax(gumbels, dim)
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m


class NetWithReshape(nn.Module):

    def __init__(self, net, dim=-3):
        super().__init__()
        self.net = net
        self.dim = dim

    def forward(self, x):
        bs, ims = x.size()[:self.dim], x.size()[self.dim:]
        x = x.reshape(-1, *ims)
        x = self.net(x)
        x = x.reshape(*bs, *x.size()[1:])
        return x


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=True, weight_init='kaiming')

    def forward(self, x):
        x = self.m(x)
        return relu(x)


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    if bias:
        nn.init.zeros_(m.bias)
    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    return m


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


def build_mlp(input_dim, output_dim, features=(1024, 1024, 1024),
              final_activation_fn=None, initial_layer_norm=False, residual=False):
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))
    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU())
        current_dim = n_features
    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(nn.ReLU())
    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([
            ResidualLayer(in_dim, h_dim, res_h_dim)
            for _ in range(n_res_layers)
        ])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = relu(x)
        return x


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(), nn.Conv2d(in_dim, res_h_dim, (3, 3), (1, 1), 1, bias=False),
            nn.ReLU(), nn.Conv2d(res_h_dim, h_dim, (1, 1), (1, 1), bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x
