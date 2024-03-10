import copy
import math
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from scipy.optimize import linear_sum_assignment
from lib.building_block import BaseNetwork, ReshapeBlock
import lib.dist as dist
from lib.utils import logsumexp, cat_sigma


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None, inner_dim=512):
        super(CNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [input_dim, 32, 64, 128, 256] if hidden is None else [input_dim] + hidden
        self.net = nn.ModuleList()
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim), nn.ReLU()
            ))
        self.net.append(nn.Sequential(
            nn.Conv2d(hidden[-1], inner_dim, 4, 1, 0),
            nn.BatchNorm2d(inner_dim), nn.ReLU()
        ))
        self.net.append(nn.Sequential(
            ReshapeBlock([inner_dim]), nn.Linear(inner_dim, output_dim)
        ))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class FCEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None):
        super(FCEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [512, 512, 512] if hidden is None else hidden
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            ReshapeBlock([input_dim * 64 * 64]),
            Linear(input_dim * 64 * 64, hidden[0]), nn.ReLU()
        ))
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(Linear(in_dim, out_dim), nn.ReLU()))
        self.net.append(Linear(hidden[-1], output_dim))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None, inner_dim=64):
        super(CNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [64, 64, 32, 32] if hidden is None else hidden
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            ReshapeBlock([input_dim, 1, 1]),
            nn.ConvTranspose2d(input_dim, inner_dim, 1, 1, 0),
            nn.BatchNorm2d(inner_dim), nn.LeakyReLU()
        ))
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(inner_dim, hidden[0], 4, 1, 0),
            nn.BatchNorm2d(hidden[0]), nn.LeakyReLU()
        ))
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim), nn.LeakyReLU()
            ))
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(hidden[-1], output_dim, 4, 2, 1), nn.Sigmoid()
        ))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class FCDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None):
        super(FCDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [512, 512, 512] if hidden is None else hidden
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            Linear(input_dim, hidden[0]), nn.ReLU()
        ))
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(Linear(in_dim, out_dim), nn.ReLU()))
        self.net.append(nn.Sequential(
            Linear(hidden[-1], output_dim * 64 * 64), nn.Sigmoid(),
            ReshapeBlock([output_dim, 64, 64])
        ))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class AutoEncoder(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, logger=None):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.total_size = self.concept_size * self.num_concept
        self.rule_size = args.rule_size
        self.num_rule = args.num_rule
        self.image_size = args.image_size
        self.image_channel = args.image_channel
        self.image_shape = (self.image_channel, self.image_size, self.image_size)
        self.device = device
        self.logger = logger
        self.x_dist = dist.Normal(std_fix=self.x_sigma)
        self.f_dist = dist.Categorical()
        self.z_dist = dist.Normal(std_fix=self.z_sigma)
        self.num_node = args.num_node
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.encoder = CNNEncoder(
            self.image_channel, self.total_size,
            hidden=args.enc, inner_dim=args.enc_inner_dim
        )
        self.decoder = CNNDecoder(
            self.total_size, self.image_channel,
            hidden=args.dec, inner_dim=args.dec_inner_dim
        )

    def encode(self, x, clone=False):
        batch_shape = x.size()[:-3]
        x = x.reshape(-1, *self.image_shape)
        encoder = self.encoder if not clone else copy.deepcopy(self.encoder)
        z_post = encoder(x).reshape(*batch_shape, self.num_concept, self.concept_size, 1)
        zs = self.z_dist.sample(z_post)
        results = {'z': zs, 'z_post': z_post}
        return results

    def decode(self, z):
        batch_shape, n, d = z.size()[:-2], z.size(-2), z.size(-1)
        z = z.reshape(-1, n * d)
        x_mean = self.decoder(z).reshape(*batch_shape, *self.image_shape)
        x_param = cat_sigma(x_mean, self.x_sigma)
        xs = self.x_dist.sample(x_param).detach()
        results = {'xs': xs, 'x_mean': x_mean, 'x_param': x_param}
        return results

    def loss(self, loss_dicts):
        bs, n = loss_dicts['z_post'].size(0), loss_dicts['z_post'].size(1)
        log_px = self.x_dist.log_density(loss_dicts['gt'], loss_dicts['x_param']).reshape(bs, -1).sum(1).mean()
        kld_z = self.z_dist.kld(loss_dicts['zt_post'], loss_dicts['zt_prior']).reshape(bs, -1).sum(-1).mean()
        kld_f = self.f_dist.kld(loss_dicts['f_post'], loss_dicts['f_prior']).reshape(bs, -1).sum(-1).mean()
        log_pr = torch.zeros(1).float().to(self.device)
        if loss_dicts['rule_label'] is not None:
            label_oh = one_hot(loss_dicts['rule_label'].clamp_min(0), num_classes=self.num_rule)
            label_mask = 1. - torch.eq(loss_dicts['rule_label'], -1).float()
            log_pr_matrix_prior = (self.f_dist.log_density(
                loss_dicts['f_prior'].unsqueeze(1), label_oh.unsqueeze(2)
            ) * label_mask.unsqueeze(2)).sum(0)
            log_pr_matrix_post = (self.f_dist.log_density(
                loss_dicts['f_post'].unsqueeze(1), label_oh.unsqueeze(2)
            ) * label_mask.unsqueeze(2)).sum(0)
            log_pr_matrix = (log_pr_matrix_prior + log_pr_matrix_post) / 2.
            row_ind, col_ind = linear_sum_assignment(-log_pr_matrix.detach().cpu().numpy())
            log_pr = log_pr_matrix[row_ind, col_ind].sum()
        modified_elbo = log_px + self.beta_label * log_pr \
            - self.beta_z * kld_z - self.beta_func * kld_f

        vae_logger = {
            'recon': log_px.detach().clone(),
            'kld_z': kld_z.detach().clone(),
            'kld_f': kld_f.detach().clone(),
            'log_pr': log_pr.detach().clone()
        }
        return modified_elbo, vae_logger

    def forward(self, x):
        pass

    def check_disentangle(self, z, dim, params):
        bs = z.size(0)
        z_interpolate = self.z_dist.interpolate(bs, params)  # n, num, nc, cs
        num = z_interpolate.size(1)
        zs = torch.cat([z.clone().detach().unsqueeze(1) for _ in range(num)], 1)  # n, num, nc, cs
        zs[:, :, dim] = z_interpolate[:, :, dim]
        results = self.decode(zs.reshape(-1, self.num_concept, self.concept_size))
        img_s, img_c = self.image_size, self.image_channel
        return results['x_mean'].reshape(bs, num, img_c, img_s, img_s), num
