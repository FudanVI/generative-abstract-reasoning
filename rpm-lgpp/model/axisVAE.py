import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import lib.dist as dist
from model.building_block import Encoder, AxisEncoder, Decoder


class AxisVAE(nn.Module):
    def __init__(self, args, device):
        super(AxisVAE, self).__init__()
        self.z_dim = args.latent_dim
        self.img_size = args.image_size
        self.device = device
        self.x_dist = dist.Normal()
        self.prior_dist = dist.Normal()
        self.q_dist = dist.Normal()
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))
        self.axis_x_dist = dist.Normal()
        self.axis_y_dist = dist.Normal()
        self.encoder = Encoder(self.z_dim * self.q_dist.nparams)
        self.encoder_axis = AxisEncoder(self.z_dim, args)
        self.decoder = Decoder(self.z_dim)
        self.beta = args.beta

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = self.prior_params.expand(expanded_size)
        return prior_params

    def model_sample(self, batch_size=1):
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        x_params = self.decoder.forward(zs)
        return x_params

    def encode(self, x, s):
        x = x.view(x.size(0), 1, self.img_size, self.img_size)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        zs = self.q_dist.sample(params=z_params)
        axis_x_params, axis_y_params = self.encoder_axis(s)
        axis_x = self.axis_x_dist.sample(params=axis_x_params)
        axis_y = self.axis_y_dist.sample(params=axis_y_params)
        results = {
            'zs': zs, 'z_params': z_params, 'axis_x': axis_x, 'axis_x_params': axis_x_params,
            'axis_y': axis_y, 'axis_y_params': axis_y_params
        }
        return results

    def decode(self, z):
        x_params_mu = self.decoder.forward(z).view(z.size(0), self.img_size, self.img_size, 1)
        x_params = torch.cat((x_params_mu, torch.full(x_params_mu.size(), -2).to(self.device)), 3)
        xs = self.x_dist.sample(params=x_params).detach()
        bs = xs.size(0)
        xs = xs.view(bs, 1, self.img_size, self.img_size)
        return xs, x_params

    def loss(self, loss_dicts, val=False):
        bs = loss_dicts['x_params'].size(0)
        logpx = self.x_dist.log_density(loss_dicts['gt'], params=loss_dicts['x_params']).view(bs, -1).sum(1).mean()
        kld = self.q_dist.kld_prior(loss_dicts['z_params'].view(bs // 9, 9, self.z_dim, 2)[:, :9, :, :],
                                    0, loss_dicts['diag'].log()[:, :9, :]).view(bs, -1).sum(1).mean()
        bs_axis = loss_dicts['axis_x_params'].size(0)
        kld_axis = self.axis_x_dist.kld(loss_dicts['axis_x_params']).view(bs_axis, -1).sum(1).mean() \
                   + self.axis_y_dist.kld(loss_dicts['axis_y_params']).view(bs_axis, -1).sum(1).mean()
        kl_coef = torch.FloatTensor([self.beta]).to(self.device)
        modified_elbo = logpx - kl_coef * kld - kl_coef * kld_axis
        vae_logger = {
            'recon': logpx.detach(),
            'kl': kld.detach(),
            'kld_axis': kld_axis.detach(),
            'kl_coef': kl_coef.detach()
        }
        return modified_elbo, vae_logger

    def forward(self, x):
        pass

    def check_disentangle(self, x, dim, upper, lower, num=32):
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        zs = self.q_dist.sample(params=z_params)
        z_dim = zs.size(1)
        zs = torch.cat([zs.view(z_dim, 1) for i in range(num)], 1)
        diff = (upper - lower) / (num - 1)
        z_interpolate = torch.FloatTensor([lower + diff * i for i in range(num)]).to(self.device)
        zs[dim] = z_interpolate
        x_recon, _ = self.decode(torch.transpose(zs, 0, 1))
        return _[:, :, :, 0].unsqueeze(1)
