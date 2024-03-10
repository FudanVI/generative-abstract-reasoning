from numbers import Number
import random
import math
import torch
import numpy as np
import os
import shutil
from torchvision.utils import make_grid


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


def gpu_id_to_device(gpu_id):
    return torch.device('cuda:%d' % gpu_id) if gpu_id >= 0 else torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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


def isnan(tensor):
    return tensor != tensor


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


def create_same_size(template, value):
    return template.new_ones(template.size()) * value


def cat_sigma(mean, sigma):
    return torch.cat(
        [mean[..., None], create_same_size(mean[..., None], math.log(sigma))], dim=-1
    ).contiguous()


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


def start_end_by_dim(x, dim):
    return sum(x[:dim]), sum(x[:dim+1])


def shuffle(x, dim=0):
    bs = x.size(dim)
    random_index = list(range(bs))
    random.shuffle(random_index)
    mask = torch.tensor(random_index, device=x.device).long()
    x = x.index_select(dim, mask)
    return x


def numel(m):
    s = 1
    for e in range(m):
        s *= e
    return s


def masked_mean(t, m, dim=-1, keepdim=False):
    """
    :param:
        t: FloatTensor (*)
        m: FloatTensor or LongTensor (*)
        dim: int
        keepdim: boolean
    :return:
        r: FloatTensor (*)
    """
    n = m.sum(dim, keepdim=True)
    n = torch.where(n == 0, n.new_ones(n.size()), n)
    s = t.sum(dim, keepdim=True)
    mean = s / n
    if not keepdim:
        mean = mean.squeeze(dim)
    return mean
