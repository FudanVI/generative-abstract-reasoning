import math
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from lib.building_block import BaseNetwork
import lib.dist as dist
from lib.utils import split_by_mask, combine_by_mask


class RuleParser(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, hidden_dim=64):
        super(RuleParser, self).__init__()
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.num_node = args.num_node
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.concept_size = args.concept_size
        self.rule_size = args.rule_size
        self.num_rule = args.num_rule
        self.device = device

        self.rule_enc_row = nn.Sequential(
            nn.Linear(self.concept_size * self.num_col, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.rule_enc_col = nn.Sequential(
            nn.Linear(self.concept_size * self.num_row, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.rule_comb = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, self.num_rule)
        )
        self.init_val = nn.Parameter(
            torch.zeros(self.num_row, self.num_col, self.concept_size).float().to(device),
            requires_grad=True
        )

    def init_matrix(self, x, mask):
        init_mat = self.init_val[None].expand(x.size(0), -1, -1, -1)
        mask = mask.reshape(1, self.num_row, self.num_col, 1)
        init_mat = x * (1 - mask) + init_mat * mask
        return init_mat

    def parse_row_repr(self, x, mask=None):
        b, n_row, n_col = x.size(0), self.num_row, self.num_col
        row_repr = self.rule_enc_row(x.reshape(b, n_row, -1)).mean(1)
        return row_repr

    def parse_col_repr(self, x, mask=None):
        b, n_row, n_col = x.size(0), self.num_row, self.num_col
        x = x.transpose(1, 2).contiguous()
        col_repr = self.rule_enc_col(x.reshape(b, n_col, -1)).mean(1)
        return col_repr

    def forward(self, x, mask=None):
        b, n, d = x.size()
        assert n == self.num_node
        n_row, n_col = self.num_row, self.num_col
        x = x.reshape(b, n_row, n_col, d)
        if mask is not None:
            mask = mask.reshape(n_row, n_col)
            x = self.init_matrix(x, mask)
        row_repr = self.parse_row_repr(x, mask=mask)
        col_repr = self.parse_col_repr(x, mask=mask)
        func_params = self.rule_comb(torch.cat([row_repr, col_repr], -1))
        return func_params


class ConvTargetPredictor(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0):
        super(ConvTargetPredictor, self).__init__()
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.num_node = args.num_node
        self.concept_size = args.concept_size
        self.rule_size = args.rule_size
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.device = device
        hidden_list = [self.concept_size] + args.pred_inner_dim + [self.concept_size]
        hidden_layers = []
        for layer_idx in range(len(hidden_list) - 1):
            if layer_idx > 0:
                hidden_layers.append(nn.ReLU())
            hidden_layers.append(
                nn.Conv2d(hidden_list[layer_idx], hidden_list[layer_idx + 1], (3, 3), (1, 1), 1)
            )
        self.net = nn.Sequential(*hidden_layers)

    def forward(self, z_tilde):
        b, n, d = z_tilde.size()
        n_row, n_col = self.num_row, self.num_col
        z_tilde = z_tilde.reshape(b, n_row, n_col, -1).permute(0, 3, 1, 2).contiguous()
        z_tilde = self.net(z_tilde)
        z_tilde = z_tilde.permute(0, 2, 3, 1).reshape(b, n, -1)
        return z_tilde


class NPPredictor(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, hidden_dim=64):
        super(NPPredictor, self).__init__()
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.device = device
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.total_size = self.num_concept * self.concept_size
        self.num_rule = args.num_rule
        self.f_dist = dist.Categorical()
        self.z_dist = dist.Normal(std_fix=self.z_sigma)
        self.num_node = args.num_node
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.hidden_dim = hidden_dim
        self.target_predictor = nn.ModuleList([
            ConvTargetPredictor(args, device, global_step=global_step)
            for _ in range(self.num_rule)
        ])
        self.rule_parser = RuleParser(args, device, global_step=global_step)

    def sample_slots(self, bs):
        return torch.zeros(bs, self.num_node, self.concept_size).float().to(self.device)

    def shuffle_sample(self, x):
        b, n, d = x.size()
        x = x.reshape(b, self.num_row, self.num_col, d)
        x = torch.index_select(x, 1, torch.randperm(self.num_row).to(x.device))
        return x.reshape(b, n, d)

    def forward(self, x, mask):
        b, n, d = x.size(0), self.num_node, self.concept_size
        results = {
            'f_post': [], 'f_prior': [], 'f': [], 'f_ll': [], 'zt_prior': [],
            'f_full': [], 'f_full_shuffle': []
        }
        for i in range(self.num_concept):
            xd = x[:, :, i].contiguous()
            xdt_init = split_by_mask(self.sample_slots(b), mask, 1)[0]

            # split by mask
            xdt, xdc = split_by_mask(xd, mask, 1)
            nt, nc = xdt.size(1), xdc.size(1)

            # get functions prior
            xd_with_init = combine_by_mask(xdt_init, xdc, mask, 1)
            func_prior = self.rule_parser(xd_with_init, mask=mask)

            # multiple predictor
            zt_prior = []
            xd_init = combine_by_mask(xdt_init, xdc, mask, 1)
            for j in range(self.num_rule):
                mat_after = self.target_predictor[j](xd_init)
                zt_prior_rule, _ = split_by_mask(mat_after, mask, 1)
                zt_prior_rule = zt_prior_rule.reshape(b, nt, d, -1)
                zt_prior.append(zt_prior_rule.unsqueeze(1))
            zt_prior = torch.cat(zt_prior, 1)

            # get functions posterior
            func_ll = self.z_dist.log_density(xdt.unsqueeze(1), zt_prior, const=False).sum([2, 3])
            func_post = func_ll + log_softmax(func_prior, dim=-1)

            # sample func
            func = self.f_dist.sample(func_post, t=self.tau)
            zt_prior = torch.sum(zt_prior * func[..., None, None, None], 1)

            results['f_post'].append(func_post.unsqueeze(1))
            results['f_prior'].append(func_prior.unsqueeze(1))
            results['f'].append(func.unsqueeze(1))
            results['zt_prior'].append(zt_prior.unsqueeze(2))

        results['f_post'] = torch.cat(results['f_post'], 1).contiguous()
        results['f_prior'] = torch.cat(results['f_prior'], 1).contiguous()
        results['f'] = torch.cat(results['f'], 1).contiguous()
        results['zt_prior'] = torch.cat(results['zt_prior'], 2).contiguous()
        return results

    def predict(self, x, mask, num_iter=None):
        b, n, d = x.size(0), x.size(1), self.concept_size
        results = {'zt': [], 'func': [], 'zt_prior': []}
        for i in range(self.num_concept):
            xd = x[:, :, i]
            xdt_init = split_by_mask(self.sample_slots(b), mask, 1)[0]

            # split by mask
            xdt, xdc = split_by_mask(xd, mask, 1)
            nt, nc = xdt.size(1), xdc.size(1)

            xd_with_init = combine_by_mask(xdt_init, xdc, mask, 1)
            func_prior = self.rule_parser(xd_with_init, mask=mask)

            func = self.f_dist.sample(func_prior, t=self.tau)
            xd_init = combine_by_mask(xdt_init, xdc, mask, 1)

            zt_prior = []
            for j in range(self.num_rule):
                mat_after = self.target_predictor[j](xd_init)
                zt_prior_rule, _ = split_by_mask(mat_after, mask, 1)
                zt_prior_rule = zt_prior_rule.reshape(b, nt, d, -1)
                zt_prior.append(zt_prior_rule.unsqueeze(1))
            zt_prior = torch.cat(zt_prior, 1)
            zt_prior = torch.sum(zt_prior * func[..., None, None, None], 1)

            zt = self.z_dist.sample(zt_prior)
            results['zt'].append(zt.unsqueeze(2))
            results['func'].append(func.unsqueeze(1))
            results['zt_prior'].append(zt_prior.unsqueeze(2))

        results['zt'] = torch.cat(results['zt'], 2).contiguous()
        results['func'] = torch.cat(results['func'], 1).contiguous()
        results['zt_prior'] = torch.cat(results['zt_prior'], 2).contiguous()
        return results

    def generate(self, b, x_axis, y_axis):
        pass

    def interpolate(self, x_axis, y_axis):
        pass
