import math
import torch


class Normal:
    def __init__(self, std_min=0.01, std_max=3.0):
        self.std_min = std_min
        self.std_max = std_max

    def unpack_params(self, params):
        mu = params[..., 0].contiguous()
        log_sigma = params[..., 1].contiguous()
        log_sigma = torch.clamp(log_sigma, math.log(self.std_min), math.log(self.std_max))
        return mu, log_sigma

    def sample(self, params):
        mean, log_std = self.unpack_params(params)
        std_z = torch.randn(mean.size()).float().to(mean.device)
        sample = std_z * torch.exp(log_std) + mean
        return sample

    def kld_standard(self, params):
        mean, log_std = self.unpack_params(params)
        kld = log_std.mul(2).add(1) - mean.pow(2) - log_std.exp().pow(2)
        kld.mul_(-0.5)
        return kld

    def kld(self, params, params_prior):
        mean, log_std = self.unpack_params(params)
        mean_prior, log_std_prior = self.unpack_params(params_prior)
        std_prior = log_std_prior.exp()
        std = log_std.exp()
        kld = (log_std - log_std_prior).mul(2).add(1) - ((mean - mean_prior).pow(2) + std.pow(2)) / std_prior ** 2
        kld.mul_(-0.5)
        return kld

    def log_density(self, sample, params):
        mean, log_std = self.unpack_params(params)
        inv_std = torch.exp(-log_std)
        tmp = (sample - mean) * inv_std
        return - 0.5 * (tmp * tmp + 2 * log_std + math.log(2 * math.pi, math.e))

    def entropy(self, params):
        mean, log_std = self.unpack_params(params)
        return 0.5 * math.log(2 * math.pi, math.e) + log_std + 0.5
