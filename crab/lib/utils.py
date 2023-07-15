from numbers import Number
import random
import math
import time
import torch
import torch.nn as nn
import numpy as np
import os
import subprocess as sp
import shutil
from torchvision.utils import make_grid

""" 1. Utils to monitor model states """


def gpu_memory_usage(gpu_id):
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
    output_cmd = sp.check_output(command.split())
    memory_used = output_cmd.decode("ascii").split("\n")[1]
    # Get only the memory part as the result comes as '10 MiB'
    memory_used = int(memory_used.split()[0])
    return memory_used


def generate_show(name):
    def show(grad):
        print('Gradient %s : Mean %.2f \t Max: %.2f \t Min: %.2f' % (name, grad.mean(), grad.max(), grad.min()))
        return grad
    return show


def show_forward(name, var):
    print('Forward %s : Mean %.2f \t Max: %.2f \t Min: %.2f' % (name, var.mean(), var.max(), var.min()))


def time_cost(ref):
    class Timer:
        def __init__(self, r):
            self.start_time = 0
            self.r = r

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            print('{} time cost {}'.format(self.r, time.time() - self.start_time))
    return Timer(ref)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


""" 2. Model helper utils """


def save_checkpoint(state, save, name='checkpoint.pth'):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, name)
    torch.save(state, filename)


def save_checkpoint_epoch(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint_%d.pth' % epoch)
    torch.save(state, filename)


class RunningAverageMeter(object):

    def __init__(self):
        self.count = 0
        self.loss_dict = dict()
        self.metric_dict = dict()

    def reset(self):
        self.count = 0
        self.loss_dict = dict()
        self.metric_dict = dict()

    def update(self, loss_logger, metric_logger):
        for key, val in loss_logger.items():
            if key in self.loss_dict:
                self.loss_dict[key] += val.item()
            else:
                self.loss_dict[key] = val.item()
        for key, val in metric_logger.items():
            if key in self.metric_dict:
                self.metric_dict[key] += val.item()
            else:
                self.metric_dict[key] = val.item()

    def get_avg(self):
        avg_loss_dict, avg_metric_dict = {}, {}
        for key, val in self.loss_dict.items():
            avg_loss_dict[key] = val / self.count
        for key, val in self.metric_dict.items():
            avg_metric_dict[key] = val / self.count
        return avg_loss_dict, avg_metric_dict

    def next(self):
        self.count += 1

    def log(self, epoch, total_epoch, best_epoch, runtime, end='\n'):
        avg_loss_dict, avg_metric_dict = self.get_avg()
        log_info = '[epoch %03d/%03d][best %03d][time %d] ELBO: %.2f' % \
            (epoch, total_epoch, best_epoch, runtime, avg_loss_dict['elbo'])
        for key, val in avg_metric_dict.items():
            log_info += ' \t %s: %.2f' % (key, val)
        for key, val in avg_loss_dict.items():
            if key != 'elbo':
                log_info += ' \t %s: %.2f' % (key, val)
        print(log_info, end=end)


class VariableMonitor(object):

    def __init__(self, root):
        self.variable_epoch = dict()
        self.variable_time = dict()
        self.root = root
        self.round = 0

    def next(self):
        self.round += 1

    def reset_epoch(self):
        self.variable_epoch = dict()

    def update_epoch(self, key, value):
        if key in self.variable_epoch:
            self.variable_epoch[key] = torch.cat((self.variable_epoch[key], value), 0)
        else:
            self.variable_epoch[key] = value

    def update_time(self):
        for key in self.variable_epoch.keys():
            if key in self.variable_time:
                self.variable_time[key] = torch.cat((self.variable_time[key], torch.mean(self.variable_epoch[key], 0, keepdim=True)), 0)
            else:
                self.variable_time[key] = torch.mean(self.variable_epoch[key], 0, keepdim=True)

    def save_variable(self):
        torch.save(self.variable_epoch, os.path.join(self.root, "variable_epoch.pth"))
        torch.save(self.variable_time, os.path.join(self.root, "variable_time.pth"))


def gpu_id_to_device(gpu_id):
    return torch.device('cuda:%d' % gpu_id) if gpu_id >= 0 else torch.device('cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weight_init_zero(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


def weight_init_kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def weight_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # nn.init.normal_(m.weight, 0, 0.01)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('Conv2d') != -1:
        # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.normal_(m.weight, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def anneal_by_milestones(epoch, milestones, value, mode='continuous'):
    is_not_less_than = [int(epoch >= m) for m in milestones]
    index = sum(is_not_less_than) - 1
    if mode == 'continuous':
        if index == len(milestones) - 1:
            return value[index]
        else:
            return value[index] + (value[index + 1] - value[index]) \
                   * ((epoch - milestones[index]) / (milestones[index + 1] - milestones[index]))
    elif mode == 'discrete':
        return value[index]


def concatenate_panel(x, t):
    img_size = x.size(-1)
    bs = t.size(0)
    x_panel = x.reshape(bs, -1, 1, img_size, img_size)
    t_answer = t.reshape(bs, 1, 1, img_size, img_size)
    panels = torch.cat((x_panel, t_answer), 1).reshape(-1, 1, img_size, img_size)
    return panels


def make_grid_list(panels, pad=2, pad_v=0, pad_h=10):
    # panels (bs, 9, 1, img_size, img_size)
    panels = panels.view(-1, 9, 1, panels.size(-1), panels.size(-1))
    bs = panels.size(0)
    concatenated_list = []
    width = 1
    for i in range(bs):
        if i > 0:
            concatenated_list.append(panels.new_ones(3, pad_h, width))
        grid = make_grid(panels[i], nrow=3, padding=pad, normalize=True, pad_value=pad_v)
        width = grid.size(-1)
        concatenated_list.append(grid)
    return torch.cat(concatenated_list, 1)


def fill_polygon(image):
    image = torch.where(image < 1, image.new_zeros(image.size()), image.new_ones(image.size()))
    left_prod = image.cumprod(dim=-1)
    right_prod = image.flip([-1]).cumprod(dim=-1).flip([-1])
    prod = left_prod + right_prod
    prod = torch.where(prod > 0, prod.new_ones(prod.size()), prod.new_zeros(prod.size()))
    return prod


""" 3. Function utils """


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


def variance_loss(z):
    variance = torch.var(z, dim=1)
    index = torch.argsort(variance.detach(), dim=-1, descending=True)
    sorted_v_1 = variance.gather(-1, index.select(-1, 0).view(-1, 1))
    sorted_v_2 = variance.gather(-1, index.select(-1, 1).view(-1, 1))
    return torch.abs(sorted_v_1 - sorted_v_2)


def clear_dir(d, create=True):
    if os.path.exists(d):
        shutil.rmtree(d)
    if create:
        os.makedirs(d)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
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


def random_from_tensor(t, n):
    assert t.size(0) >= n
    indices = list(range(t.size(0)))
    random.shuffle(indices)
    indices = torch.tensor(indices[:n]).long().to(t.device)
    return t.index_select(0, indices).contiguous()


def check_non_singular(x):
    try:
        torch.inverse(x)
    except:
        return False
    return True


def check_non_cholesky(x):
    try:
        _ = torch.cholesky(x)
    except:
        return False
    return True


def check_det_positive(x):
    return not isnan(torch.log(torch.det(x))).any()


def create_same_size(template, value):
    return template.new_ones(template.size()) * value


def cat_sigma(mean, sigma):
    return torch.cat(
        [mean[..., None], create_same_size(mean[..., None], math.log(sigma))], dim=-1
    ).contiguous()


def correlation(x, y, eps=1e-10):
    # size num_slot * bs
    mean_x, mean_y = x.mean(-1, keepdim=True), y.mean(-1, keepdim=True)
    xm, ym = x - mean_x, y - mean_y
    r_num = (xm.unsqueeze(1) * ym.unsqueeze(0)).sum(-1)
    r_den = torch.norm(xm, dim=-1).unsqueeze(1) * torch.norm(ym, dim=-1).unsqueeze(0)
    r_val = r_num / (r_den + eps)
    return r_val


def cosine(x, y, eps=1e-10):
    # size num_slot * bs
    r_num = (x.unsqueeze(1) * y.unsqueeze(0)).sum(-1)
    r_den = torch.norm(x, dim=-1).unsqueeze(1) * torch.norm(y, dim=-1).unsqueeze(0)
    r_val = r_num / (r_den + eps)
    return r_val


def split_by_mask(a, m, dim):
    assert m.size(0) == a.size(dim)
    # 1 for target, 0 for context
    target_ind = (m == 1).nonzero(as_tuple=False).view(-1).long()
    context_ind = (m == 0).nonzero(as_tuple=False).view(-1).long()
    a_target = a.index_select(dim, target_ind).contiguous()
    a_context = a.index_select(dim, context_ind).contiguous()
    return a_target, a_context


def combine_by_mask(t, c, m, dim):
    # 1 for target, 0 for context
    target_ind = (m == 1).nonzero(as_tuple=False).view(-1).long()
    context_ind = (m == 0).nonzero(as_tuple=False).view(-1).long()
    disorder_a = torch.cat((t, c), dim).contiguous()
    combined_size = disorder_a.size()
    reshape_size = [1] * len(combined_size)
    reshape_size[dim] = combined_size[dim]
    reshape_size = tuple(reshape_size)
    trans_index = torch.cat((target_ind, context_ind), 0).contiguous()
    trans_index = trans_index.reshape(reshape_size).expand(combined_size)
    return t.new_zeros(combined_size).scatter_(dim, trans_index, disorder_a)


def masked_softmax(params, mask, dim):
    """
    :param:
        params: FloatTensor (*)
        mask: BoolTensor (*), positions filled with true will be masked
        dim: int, the dimension to make softmax
    :return:
        masked matrix: FloatTensor (*)
    """
    params_with_inf = torch.where(
        mask, params.new_ones(params.size()) * - float('inf'), params.new_zeros(params.size())
    )
    params = params + params_with_inf.detach()
    params_max = params.max(dim=dim, keepdim=True)[0]
    params_max = torch.where(params_max == - float('inf'), params_max.new_zeros(params_max.size()), params_max)
    params = params - params_max.detach()
    params = params.exp() * (1. - mask.float())
    params_sum = params.sum(dim=dim, keepdim=True)
    params_sum = torch.where(params_sum == 0, params_sum.new_ones(params_sum.size()), params_sum)
    params = params / params_sum
    return params


def batch_apply(x, dim, nets, cat_dim=None, dim_size=None):
    if dim_size is None:
        dim_size = [1] * x.size(dim)
    assert x.size(dim) == sum(dim_size)
    assert len(dim_size) == len(nets)
    results = []
    for d in range(len(dim_size)):
        s, e = start_end_by_dim(dim_size, d)
        x_dim = x.index_select(dim, torch.arange(s, e, device=x.device))
        x_dim = nets[d](x_dim)
        if cat_dim is not None:
            x_dim = x_dim.unsqueeze(cat_dim)
        results.append(x_dim)
    if cat_dim is not None:
        results = torch.cat(results, cat_dim)
    else:
        results = torch.cat(results, dim)
    return results


def start_end_by_dim(x, dim):
    return sum(x[:dim]), sum(x[:dim+1])


def invert_image(x):
    return x.new_ones(x.size()) - x


def shuffle(x, dim=0):
    bs = x.size(dim)
    random_index = list(range(bs))
    random.shuffle(random_index)
    mask = torch.tensor(random_index, device=x.device).long()
    x = x.index_select(dim, mask)
    return x


def hungarian(x):
    n = x.size(0)
    indicator = x.new_zeros(n, n)
    mask = x.new_ones(n, n)
    dist = x.detach().clone()
    for i in range(n):
        pos = (dist + invert_image(mask) * 1e10).argmin()
        p, q = pos // n, pos % n
        indicator[p, q] = 1.0
        mask[p, :].fill_(0.0)
        mask[:, q].fill_(0.0)
    return indicator


def estimate_total_correlation(qz_samples, qz_params, z_dist, max_samples=10000, z_dim=None):
    """
    :param:
        qz_samples: FloatTensor (bs, dim)
        qz_params: FloatTensor (bs, dim, n_param)
        z_dist: Object to calculate log density of z
        max_samples: int, the most number of samples to calculate
        z_dim: list, dimensions of each concept
    :return:
        total correlation: FloatTensor (1)
    """
    # Only take a sample subset of the samples
    num_sample = min(max_samples, qz_samples.size(0))
    select_indices = torch.randperm(num_sample).to(qz_samples.device)
    qz_samples = qz_samples.index_select(0, select_indices)
    bs, dim, n_param = qz_params.size()
    log_qz_i = z_dist.log_density(
        qz_samples.reshape(1, bs, dim).expand(bs, -1, -1),
        qz_params.reshape(bs, 1, dim, 2).expand(-1, bs, -1, -1)
    )  # bs, bs, dim
    # computes - log q(z_i) summed over mini-batch
    log_qz_dim = torch.cat([e.sum(-1, keepdim=True) for e in log_qz_i.split(z_dim, -1)], -1)
    marginal_entropy = logsumexp(log_qz_dim, dim=0).mean(0).sum() - 2 * log_qz_dim.size(-1) * math.log(bs)
    # computes - log q(z) summed over mini-batch
    joint_entropy = logsumexp(log_qz_i.sum(-1), dim=0).mean() - 2 * math.log(bs)
    dependence = joint_entropy - marginal_entropy
    return dependence


def extract_diag(m):
    """
    :param:
        m: FloatTensor (*, n, n)
    :return:
        diag: FloatTensor (*, n)
    """
    d = m.size(-1)
    batch_shape = m.size()[:-2]
    diag = m.reshape(-1, d, d) * torch.eye(d).float().to(m.device)[None]
    diag = diag.sum(-1).reshape(*batch_shape, d)
    return diag


def extract_from_2d_mat(m, way='diag', mask=None):
    """
    :param:
        m: FloatTensor (*, n, n, d)
        way: float, should be 'diag', 'upper_tri', 'lower_tri'
        mask: FloatTensor/LongTensor (n, n) where the positions to extract/desert are set 1/0
    :return:
        diag: FloatTensor (*, s, n, d)
    """
    batch_shape, n, d = m.size()[:-3], m.size(-2), m.size(-1)
    if mask is None:
        if way == 'diag':
            mask = torch.eye(n).to(m.device)
        elif way == 'upper_tri':
            mask = torch.triu(torch.ones(n, n), diagonal=1).to(m.device)
        elif way == 'lower_tri':
            mask = torch.tril(torch.ones(n, n), diagonal=-1).to(m.device)
        else:
            raise NotImplementedError
    mask = mask.long()
    indices = torch.nonzero(mask.reshape(-1)).reshape(-1).to(m.device)
    m_selected = torch.index_select(m.reshape(*batch_shape, -1, d), -2, indices)
    return m_selected
