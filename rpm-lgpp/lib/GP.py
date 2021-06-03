import torch
import torch.nn as nn
from torch.autograd import Function
from torch.distributions.multivariate_normal import MultivariateNormal
import random
import math
import lib.dist as dist


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


def add_noise(x):
    device = x.device
    n = torch.FloatTensor(x.size()).uniform_(-1, 1).to(device)
    scale = x.detach() / 20
    x = x + scale * n
    return x


def chain_batch_matmul(mats):
    base = None
    for i, m in enumerate(mats):
        if i == 0:
            base = m
        else:
            base = torch.matmul(base, m)
    return base


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 32))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(32, 64))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(64, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(64, 64))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('linear5', torch.nn.Linear(64, output_dim))
        self.add_module('final_act', torch.nn.Tanh())


class AxisNet(nn.Module):
    def __init__(self, args):
        super(AxisNet, self).__init__()
        self.z_dim = args.latent_dim
        self.nets = nn.ModuleList()
        for i in range(self.z_dim):
            self.nets.append(nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 6)
            ))

    def forward(self, x):
        axis_list = [self.nets[i](x[:, i, :]).unsqueeze(1) for i in range(self.z_dim)]
        axis = torch.cat(axis_list, 1)
        return axis


class DGP(nn.Module):
    def __init__(self, args, device, eps_range=(1e-4, 5e-4)):
        super(DGP, self).__init__()
        self.device = device
        self.z_dim = args.latent_dim
        self.axis_dim = args.axis_dim
        self.rbf_feature_size = 8
        self.eps_range = eps_range
        self.x_axis_net = AxisNet(args)
        self.y_axis_net = AxisNet(args)
        self.axis_dist = dist.Normal()
        self.extractors = nn.ModuleList()
        for i in range(self.z_dim):
            tmp_net = LargeFeatureExtractor(2 * self.axis_dim, self.rbf_feature_size)
            self.extractors.append(tmp_net)
        self.rbf_coef_sigma = 1
        self.rbf_coef_l = 1

    def rbf(self, x):
        bs, z_dim, num_x, d = x.size(0), x.size(1), x.size(2), x.size(3)
        diff = x.view(bs, z_dim, num_x, 1, d) - x.view(bs, z_dim, 1, num_x, d)
        product = (diff * diff).sum(-1)
        return (self.rbf_coef_sigma ** 2) * torch.exp(- 0.5 * product / (self.rbf_coef_l ** 2))

    def draw_eps(self, bs, z_dim, k_size):
        eps = random.random() * (self.eps_range[1] - self.eps_range[0]) + self.eps_range[0]
        identify = torch.eye(k_size) * eps
        identify = identify.to(self.device)
        identify = identify.reshape((1, k_size, k_size))
        return identify.repeat(bs * z_dim, 1, 1).view(bs, z_dim, k_size, k_size)

    def make_non_singluar(self, x):
        bs = x.size(0)
        z_dim = x.size(1)
        k_size = x.size(2)
        if check_non_singular(x) and check_det_positive(x):
            return x, 0
        eps_matrix = self.draw_eps(bs, z_dim, k_size)
        count = 1
        while check_non_singular(x + eps_matrix) and check_det_positive(x + eps_matrix) is False:
            eps_matrix = self.draw_eps(bs, z_dim, k_size)
            count += 1
        return x + eps_matrix, count

    def make_non_cholesky(self, x):
        bs = x.size(0)
        z_dim = x.size(1)
        k_size = x.size(2)
        if check_non_cholesky(x.view(-1, k_size, k_size)):
            return x, 0
        eps_matrix = self.draw_eps(bs, z_dim, k_size)
        count = 1
        while not check_non_cholesky(x.view(-1, k_size, k_size)):
            eps_matrix = self.draw_eps(bs, z_dim, k_size)
            count += 1
        return x + eps_matrix, count

    def reparameterize(self, mu, sigma):
        bs = mu.size(0)
        z_dim = mu.size(1)
        noise = torch.randn(bs, z_dim).to(self.device)
        return mu + noise * sigma

    def axis_kl(self, axis, params):
        bs = axis.size(0)
        prior_params = torch.zeros(params.size()).float().to(self.device)
        logq = self.axis_dist.log_density(axis, params=params).view(bs, -1).sum(-1)
        logp = self.axis_dist.log_density(axis, params=prior_params).view(bs, -1).sum(-1)
        return logq - logp

    def build_points(self, x, x_axis, y_axis):
        x = torch.transpose(x, 1, 2)
        bs, z_dim = x.size(0), x.size(1)

        axis_size = (bs, z_dim, 3, 3, self.axis_dim)
        coordinate = torch.cat([x_axis.view(bs, z_dim, 3, 1, self.axis_dim).expand(axis_size),
                                y_axis.view(bs, z_dim, 1, 3, self.axis_dim).expand(axis_size)], -1)
        points = coordinate.contiguous().view(bs, z_dim, -1, 2 * self.axis_dim)
        point_x = points[:, :, :-1, :].contiguous()
        point_y = x[:, :, :-1][..., None].contiguous()
        target_x = points[:, :, -1:, :].contiguous()
        target_y = x[:, :, -1].contiguous()
        return point_x, point_y, target_x, target_y

    def forward(self, x, x_axis, y_axis, mode='train', use_diag=False):
        bs, z_dim = x.size(0), x.size(2)
        px, py, tx, ty = self.build_points(x, x_axis, y_axis)
        features = []
        for i in range(z_dim):
            features.append(
                self.extractors[i](torch.cat((px, tx), 2)[:, i, :, :].contiguous().view(-1, 2 * self.axis_dim))
                    .view(bs, 1, -1, self.rbf_feature_size))
        feature = torch.cat(tuple(features), 1)
        feature = add_noise(feature)
        kernels = self.rbf(feature)
        ks = kernels.size(2)
        cn = kernels[:, :, :ks - 1, :ks - 1]
        k = kernels[:, :, :ks - 1, ks - 1:ks]
        kt = kernels[:, :, ks - 1:ks, :ks - 1]
        c = kernels[:, :, ks - 1:ks, ks - 1:ks]

        mu = torch.matmul(kt, torch.solve(py, cn)[0]).view(bs, z_dim)
        sigma = c.view(bs, z_dim) - torch.matmul(kt, torch.solve(k, cn)[0]).view(bs, z_dim)
        sigma = torch.where(sigma < 1e-5, torch.ones(sigma.size()).to(self.device) * 1e-5, sigma)
        z_cons = self.reparameterize(mu, sigma)

        if mode == 'train' or mode == 'eval':
            if use_diag:
                kernels_diag = (torch.eye(ks).to(self.device)[None, None, ...] * kernels).sum(-1)
                results = {
                    'z_cons': z_cons, 'mu': mu, 'sigma': sigma, 'diag': kernels_diag.transpose(-2, -1)
                }
                return results
            full_py = torch.cat((py, z_cons[..., None, None]), 2)
            kernels_det = torch.det(kernels)
            kernels_det = torch.where(kernels_det < 1e-5, torch.ones(kernels_det.size()).to(self.device) * 1e-5, kernels_det)
            gp_loss = - torch.tensor([2 * math.pi]).float().to(self.device).log() * ks / 2 - kernels_det.log() / 2 \
                      - torch.matmul(full_py.transpose(-2, -1), torch.solve(full_py, kernels)[0]).view(bs, z_dim) / 2
            results = {'z_cons': z_cons, 'mu': mu, 'sigma': sigma, 'gp_loss': gp_loss}
            return results
        else:
            results = {'z_cons': z_cons, 'mu': mu, 'sigma': sigma}
            return results

    def predict(self, x, mask, x_axis, y_axis):
        bs, z_dim = x.size(0), x.size(2)

        def _build_point_by_mask(_x, _xx, _yx, _ad, _m):
            _x = torch.transpose(_x, 1, 2)
            _bs, _zd = _x.size(0), _x.size(1)
            _as = (_bs, _zd, 3, 3, _ad)
            _c = torch.cat([_xx.view(_bs, _zd, 3, 1, _ad).expand(_as), _yx.view(_bs, _zd, 1, 3, _ad).expand(_as)], -1)
            _ps = _c.contiguous().view(_bs, _zd, -1, 2 * _ad)
            _mf = _m.view(-1)
            _oi = _mf.nonzero().view(-1)
            _zi = (_mf - 1.0).nonzero().view(-1)

            _px = torch.index_select(_ps, 2, _oi).contiguous()
            _py = torch.index_select(_x, 2, _oi)[..., None].contiguous()
            _tx = torch.index_select(_ps, 2, _zi).contiguous()
            _ty = torch.index_select(_x, 2, _zi)[..., None].contiguous()
            return _px, _py, _tx, _ty, torch.cat((_oi, _zi), 0)

        px, py, tx, ty, order = _build_point_by_mask(x, x_axis, y_axis, self.axis_dim, mask)

        features = []
        for i in range(z_dim):
            features.append(
                self.extractors[i](torch.cat((px, tx), 2)[:, i, :, :].contiguous().view(-1, 2 * self.axis_dim))
                    .view(bs, 1, -1, self.rbf_feature_size))
        feature = torch.cat(tuple(features), 1)
        feature = add_noise(feature)
        kernels = self.rbf(feature)
        ks = kernels.size(2)
        cn = kernels[:, :, :ks - tx.size(2), :ks - tx.size(2)]
        k = kernels[:, :, :ks - tx.size(2), ks - tx.size(2):ks]
        kt = kernels[:, :, ks - tx.size(2):ks, :ks - tx.size(2)]
        c = kernels[:, :, ks - tx.size(2):ks, ks - tx.size(2):ks]

        mu = torch.matmul(kt, torch.solve(py, cn)[0]).view(bs, z_dim, -1)
        sigma = c - torch.matmul(kt, torch.solve(k, cn)[0])
        sigma = torch.where(sigma < 1e-5, torch.ones(sigma.size()).to(self.device) * 1e-5, sigma)

        z_cons = []
        for i in range(self.z_dim):
            z_cons.append(MultivariateNormal(mu[:, i, :], sigma[:, i, :, :]).rsample().unsqueeze(1))
        z_cons = torch.cat(z_cons, 1)
        z_full = torch.cat((py.squeeze(-1), z_cons), -1)

        z_full_ordered = torch.zeros(z_full.size()).float().to(self.device)
        z_full_ordered.scatter_(2, order.view(1, 1, -1).expand(z_full.size()), z_full)

        results = {'z_cons': z_cons, 'z_full': z_full_ordered}
        return results

    def generate(self, x_axis, y_axis):
        bs, z_dim = x_axis.size(0), x_axis.size(1)
        axis_size = (bs, self.z_dim, 3, 3, self.axis_dim)
        coordinate = torch.cat([x_axis.view(bs, z_dim, 3, 1, self.axis_dim).expand(axis_size),
                                y_axis.view(bs, z_dim, 1, 3, self.axis_dim).expand(axis_size)], -1)
        points = coordinate.contiguous().view(bs, self.z_dim, -1, 2 * self.axis_dim)
        features = []
        for i in range(self.z_dim):
            features.append(
                self.extractors[i](points[:, i, :, :].contiguous().view(-1, 2 * self.axis_dim))
                    .view(bs, 1, -1, self.rbf_feature_size))
        feature = torch.cat(tuple(features), 1)
        feature = add_noise(feature)
        kernels = self.rbf(feature)
        kernels, count = self.make_non_singluar(kernels)
        if count > 0:
            print("Redraw count: %d" % count)
        kernels, count = self.make_non_cholesky(kernels)
        if count > 0:
            print("Redraw count: %d" % count)
        z_gen = []
        ks = kernels.size(2)
        zero_mean = torch.zeros(ks).to(self.device)
        for i in range(self.z_dim):
            z_gen.append(MultivariateNormal(zero_mean, kernels[:, i, :, :].view(-1, ks, ks)).sample().unsqueeze(1))
        z_gen = torch.cat(z_gen, 1)
        return z_gen.transpose(-2, -1), kernels

    def interpolate(self, x_axis, y_axis):
        nx = x_axis.size(2)
        ny = y_axis.size(2)
        bs, z_dim = x_axis.size(0), x_axis.size(1)
        axis_size = (1, self.z_dim, nx, ny, self.axis_dim)
        coordinate = torch.cat([x_axis.view(bs, z_dim, nx, 1, self.axis_dim).expand(axis_size),
                                y_axis.view(bs, z_dim, 1, ny, self.axis_dim).expand(axis_size)], -1)
        points = coordinate.contiguous().view(1, self.z_dim, -1, 2 * self.axis_dim)
        features = []
        for i in range(self.z_dim):
            features.append(
                self.extractors[i](points[:, i, :, :].contiguous().view(-1, 2 * self.axis_dim))
                    .view(1, 1, -1, self.rbf_feature_size))
        feature = torch.cat(tuple(features), 1)
        feature = add_noise(feature)
        kernels = self.rbf(feature)
        kernels, count = self.make_non_singluar(kernels)
        if count > 0:
            print("Redraw count: %d" % count)
        kernels, count = self.make_non_cholesky(kernels)
        if count > 0:
            print("Redraw count: %d" % count)
        z_gen = []
        ks = kernels.size(2)
        zero_mean = torch.zeros(ks).to(self.device)
        for i in range(self.z_dim):
            z_gen.append(MultivariateNormal(zero_mean, kernels[0, i, :, :].view(ks, ks)).sample()[None, None, ...])
        print(z_gen[0].size())
        z_gen = torch.cat(z_gen, 1)
        return z_gen.transpose(-2, -1)
