import copy
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from lib.building_block import BaseNetwork
import lib.dist as dist
from lib.utils import split_by_mask, combine_by_mask, cat_sigma


class RuleParser(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, hidden_dim=64):
        super(RuleParser, self).__init__()
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.num_node = args.num_node
        self.concept_size = args.concept_size
        self.rule_size = args.rule_size
        self.device = device
        self.point_enc = nn.Sequential(
            nn.Linear(self.concept_size * 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.rule_enc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_node ** 2, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, self.rule_size * 2)
        )

    def forward(self, x, mask=None):
        b, n, d = x.size()
        assert n == self.num_node
        re = self.point_enc(torch.cat([
            x.unsqueeze(2).expand(-1, -1, n, -1), x.unsqueeze(1).expand(-1, n, -1, -1)
        ], -1))
        if mask is not None:
            mask_matrix = ((1 - mask[:, None]) * (1 - mask[None])).float()
            re = (re * mask_matrix.reshape(1, n, n, 1))
        re = re / (self.num_node * 2)
        func_params = self.rule_enc(re.reshape(b, -1)).reshape(b, -1, 2)
        if self.f_sigma > 0:
            func_params = cat_sigma(func_params[..., 0], self.f_sigma)
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
        self.net = nn.Sequential(
            nn.Conv2d(self.concept_size + self.rule_size, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, self.concept_size, 3, 1, 1)
        )

    def forward(self, z_tilde, mask=None):
        b, n, d = z_tilde.size()
        z_tilde = z_tilde.reshape(b, self.num_row, self.num_col, -1)
        z_tilde = self.net(z_tilde.permute(0, 3, 1, 2).contiguous())
        z_tilde = z_tilde.permute(0, 2, 3, 1).reshape(b, n, -1)
        return z_tilde


class Predictor(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, hidden_dim=64):
        super(Predictor, self).__init__()
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.device = device
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.total_size = self.num_concept * self.concept_size
        self.rule_size = args.rule_size
        self.num_rule = args.num_rule
        self.f_dist = dist.Normal(std_max=0.05)
        self.z_dist = dist.Normal(std_fix=self.z_sigma)
        self.num_node = args.num_node
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.hidden_dim = hidden_dim

        self.target_predictors = nn.ModuleList()
        self.rule_parsers = nn.ModuleList()
        for i in range(self.num_concept):
            self.target_predictors.append(ConvTargetPredictor(args, device, global_step=global_step))
            self.rule_parsers.append(RuleParser(args, device, global_step=global_step))

    def state_dict_base(self):
        return self.state_dict()

    def load_state_dict_base(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

    def sample_slots(self, bs):
        return torch.zeros(bs, self.num_node, self.concept_size).float().to(self.device)

    def prior_mean(self, x):
        results = {'prior': []}
        for i in range(self.num_concept):
            xd = x[i, :, :, i].contiguous()
            func_post = copy.deepcopy(self.rule_parsers[i])(xd)
            results['prior'].append(func_post[..., 0].unsqueeze(1))
        results['prior'] = torch.cat(results['prior'], 1).contiguous()
        return results

    def cluster(self, x):
        x = x.detach().cpu().numpy()
        labels = GaussianMixture(n_components=self.num_rule).fit_predict(x)
        labels = torch.from_numpy(labels).to(self.device)
        center_index = []
        for i in range(self.num_rule):
            indices = (labels == i).nonzero()
            center_index.append(indices[0] if len(indices) > 0 else 0)
        return torch.tensor(center_index).long().to(self.device)

    def forward(self, x, mask):
        b, n, d = x.size(0), self.num_node, self.concept_size
        results = {'f_post': [], 'f_prior': [], 'f': [], 'zt_prior': []}
        for i in range(self.num_concept):
            xd = x[:, :, i].contiguous()
            xdt_init = split_by_mask(self.sample_slots(b), mask, 1)[0]

            # split by mask
            xdt, xdc = split_by_mask(xd, mask, 1)
            nt, nc = xdt.size(1), xdc.size(1)

            # get functions posterior
            func_post = self.rule_parsers[i](xd)

            # get functions prior
            xd_with_init = combine_by_mask(xdt_init, xdc, mask, 1)
            func_prior = self.rule_parsers[i](xd_with_init, mask=mask)

            # sample func
            func = self.f_dist.sample(func_post)

            xd_init = combine_by_mask(xdt_init, xdc, mask, 1)
            mat_before = torch.cat([xd_init, func.unsqueeze(1).expand(-1, n, -1)], dim=-1)
            mat_after = self.target_predictors[i](mat_before)
            zt_prior, _ = split_by_mask(mat_after, mask, 1)
            zt_prior = zt_prior.reshape(b, nt, d, -1)

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
            func_prior = self.rule_parsers[i](xd_with_init, mask=mask)

            func = self.f_dist.sample(func_prior)
            xd_init = combine_by_mask(xdt_init, xdc, mask, 1)
            mat_before = torch.cat([xd_init, func.unsqueeze(1).expand(-1, n, -1)], dim=-1)
            mat_after = self.target_predictors[i](mat_before)
            zt_prior, _ = split_by_mask(mat_after, mask, 1)
            zt_prior = zt_prior.reshape(b, nt, d, -1)
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
