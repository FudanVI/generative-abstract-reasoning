from numbers import Number
import math
import torch
import os
import shutil


def save_checkpoint(state, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth')
    torch.save(state, filename)


class RunningAverageMeter(object):

    def __init__(self):
        self.count = 0
        self.value_dict = None

    def reset(self):
        self.count = 0
        self.value_dict = None

    def update(self, logger):
        self.count += 1
        if self.value_dict is None:
            self.value_dict = logger
        else:
            for key, val in logger.items():
                self.value_dict[key] += val.item()

    def get_avg(self):
        avg_dict = {}
        for key, val in self.value_dict.items():
            avg_dict[key] = val / self.count
        return avg_dict

    def log(self, epoch, total_epoch, best_epoch):
        avg_dict = self.get_avg()
        log_info = '[epoch %03d/%03d][best %03d] ELBO: %.2f' % (epoch, total_epoch, best_epoch, avg_dict['elbo'])
        for key, val in avg_dict.items():
            if key != 'elbo':
                log_info += ' \t %s: %.2f' % (key, val)
        print(log_info)


def isnan(tensor):
    return tensor != tensor


def kld(mu, sigma, mean=0, std=1):
    return math.log(std) - torch.log(sigma) + 0.5 * std * (sigma ** 2 + (mu - mean) ** 2) - 0.5


def kld_non_norm(mu, sigma, mean_logstd):
    mean = mean_logstd.select(-1, 0)
    logstd = mean_logstd.select(-1, 1)
    return logstd - torch.log(sigma) + 0.5 * torch.exp(logstd) * (sigma ** 2 + (mu - mean) ** 2) - 0.5


def log_gaussian_prob(x, mu, sigma):
    return - torch.log(sigma) - 0.5 * (x - mu) ** 2 / sigma ** 2 - 0.5 * math.log(2 * math.pi)


def clear_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def gamma(x):
    return torch.FloatTensor([x]).lgamma().exp().item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
