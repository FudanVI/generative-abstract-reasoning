from numbers import Number
import random
import math
import torch
import torch.nn as nn
import os
import shutil
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        self.value_dict = dict()

    def reset(self):
        self.count = 0
        self.value_dict = dict()

    def update(self, logger):
        for key, val in logger.items():
            if key in self.value_dict:
                self.value_dict[key] += val.item()
            else:
                self.value_dict[key] = val.item()

    def get_avg(self):
        avg_dict = {}
        for key, val in self.value_dict.items():
            avg_dict[key] = val / self.count
        return avg_dict

    def next(self):
        self.count += 1

    def log(self, epoch, total_epoch, best_epoch):
        avg_dict = self.get_avg()
        log_info = '[epoch %03d/%03d][best %03d] ELBO: %.2f' % (epoch, total_epoch, best_epoch, avg_dict['elbo'])
        for key, val in avg_dict.items():
            if key != 'elbo':
                log_info += ' \t %s: %.2f' % (key, val)
        print(log_info)


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


def plot_panel_with_mask(panel, mask, pad=3, c_color=None, t_color=None):
    # panel: (9 * 1 * s * s)
    if c_color is None:
        c_color = [0, 0, 0]
    if t_color is None:
        t_color = [0, 0, 0]
    img_size = panel.shape[2]
    canvas = np.zeros((3 * img_size + 4 * pad, 3 * img_size + 4 * pad, 3))
    c_color = np.array(c_color) / 255
    t_color = np.array(t_color) / 255
    # plot border
    for i in range(9):
        row, col = i // 3, i % 3
        for j in range(3):
            t = pad + img_size
            canvas[row * t: (row + 1) * t + pad, col * t: (col + 1) * t + pad, j] = c_color[j]
    for i in range(9):
        row, col = i // 3, i % 3
        if mask[row][col] == 1:
            for j in range(3):
                t = pad + img_size
                canvas[row * t: (row + 1) * t + pad, col * t: (col + 1) * t + pad, j] = t_color[j]
        for j in range(3):
            t = pad + img_size
            canvas[row * t + pad: (row + 1) * t, col * t + pad: (col + 1) * t, j] = panel[i, 0, :, :]
    return canvas


def plot_prediction(panel, gt, mask, row, col, pad_outside=10, pad_inside=3, c_color=None, t_color=None):
    mask = mask.view(row, col)
    panel_canvas = plot_panel_with_mask(panel, mask, pad=pad_inside, c_color=c_color, t_color=t_color)
    gt_canvas = plot_panel_with_mask(gt, mask, pad=pad_inside, c_color=c_color, t_color=c_color)
    padding = np.ones((pad_outside, panel_canvas.shape[1], 3))
    canvas = np.concatenate((panel_canvas, padding, gt_canvas), axis=0)
    canvas = np.transpose(canvas, (2, 0, 1))
    return canvas


def plot_correlation(correlation):
    fig, ax = plt.subplots()
    ax.imshow(correlation, interpolation='nearest', cmap='Blues')
    ax.set_xticks(np.arange(correlation.shape[1]))
    ax.set_yticks(np.arange(correlation.shape[0]))
    ax.set_xticklabels(np.arange(correlation.shape[1]))
    ax.set_yticklabels(np.arange(correlation.shape[0]))
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            ax.text(j, i, np.around(correlation, decimals=1)[i, j], ha='center', va='center', color='orange')
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = np.transpose(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), (2, 0, 1))
    plt.close('all')
    return data


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


def flat_to_diag(m):
    """
    :param:
        m: FloatTensor (*, n)
    :return:
        diag: FloatTensor (*, n, n)
    """
    batch_shape, n = m.size()[:-1], m.size(-1)
    m = m.reshape(-1, n)
    identity = torch.eye(n).float().to(m.device)[None, ...]
    diag_m = identity * m[..., None]
    return diag_m.reshape(*batch_shape, n, n)
