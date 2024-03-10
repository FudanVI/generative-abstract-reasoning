import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax, gumbel_softmax


class Normal:
    def __init__(self, std_min=0.0, std_max=0.0, std_fix=0.0):
        self.std_min = std_min
        self.std_max = std_max
        assert std_min >= 0.0
        assert std_max > std_min or std_max == 0.0
        self.std_fix = std_fix
        self.act = nn.Sigmoid() if std_max > 0.0 else nn.Softplus()
        self.num_param = 2 if std_fix == 0.0 else 1

    def act_func(self, log_sigma):
        if self.std_max > 0.0:
            return self.std_min + log_sigma.sigmoid() * (self.std_max - self.std_min)
        else:
            return self.std_min + torch.nn.functional.softplus(log_sigma)

    def unpack_params(self, params):
        if self.std_fix > 0.0:
            mu = params[..., 0].contiguous()
            sigma = mu.new_ones(mu.size()) * self.std_fix
        else:
            mu = params[..., 0].contiguous()
            log_sigma = params[..., 1].contiguous()
            sigma = self.act_func(log_sigma)
        return mu, sigma

    def sample(self, params):
        mean, std = self.unpack_params(params)
        noise = torch.randn(mean.size()).float().to(mean.device)
        sample = noise * std + mean
        return sample

    def kld_standard(self, params):
        mean, std = self.unpack_params(params)
        kld = std.log().mul(2).add(1) - mean.pow(2) - std.pow(2)
        kld.mul_(-0.5)
        return kld

    def kld(self, params, params_prior):
        mean, std = self.unpack_params(params)
        mean_prior, std_prior = self.unpack_params(params_prior)
        kld = (std.log() - std_prior.log()).mul(2).add(1) - ((mean - mean_prior).pow(2) + std.pow(2)) / std_prior ** 2
        kld.mul_(-0.5)
        return kld

    def log_density(self, sample, params, const=True):
        mean, std = self.unpack_params(params)
        inv_std = 1.0 / std
        tmp = (sample - mean) * inv_std
        if const:
            return - 0.5 * (tmp * tmp + 2 * std.log() + math.log(2 * math.pi, math.e))
        else:
            return - 0.5 * tmp.pow(2)

    def entropy(self, params):
        mean, std = self.unpack_params(params)
        return 0.5 * math.log(2 * math.pi, math.e) + std.log() + 0.5

    @staticmethod
    def interpolate(batch_size, params):
        start, end, num_split = params['start'], params['end'], params['num_split']
        diff = (end - start) / (num_split - 1)
        z_interpolate = torch.stack([start + diff * i for i in range(num_split)], dim=0)  # num_split, nc, cs
        z_interpolate = z_interpolate[None].expand(batch_size, -1, -1, -1)  # bs, num_split, nc, cs
        return z_interpolate


class Categorical:
    def __init__(self, eps=1e-10):
        self.eps = eps
        self.num_param = 1

    @staticmethod
    def log_density(params, one_hot):
        p = softmax(params, dim=-1)
        kld = (p * one_hot).sum(-1)
        return kld

    def sample(self, params, t=1.0, hard=False, dim=-1):
        return gumbel_softmax(params, tau=t, hard=hard, eps=self.eps, dim=dim)

    @staticmethod
    def kld_standard(params):
        cat_prob = softmax(params, dim=-1)
        kld = cat_prob * (cat_prob * cat_prob.size(-1)).log()
        kld = kld.sum(-1)
        return kld

    def kld(self, params, params_prior):
        cat_prob = softmax(params, dim=-1)
        cat_prob_p = softmax(params_prior, dim=-1)
        kld = (cat_prob + self.eps) * (cat_prob / (cat_prob_p + self.eps) + self.eps).log()
        kld = kld.sum(-1)
        return kld
