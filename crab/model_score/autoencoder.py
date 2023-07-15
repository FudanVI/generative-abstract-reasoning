import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from lib.building_block import BaseNetwork, ReshapeBlock
import lib.dist as dist
from lib.utils import logsumexp, cat_sigma


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None, inner_dim=512):
        super(CNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [input_dim, 32, 64, 128, 256] if hidden is None else [input_dim] + hidden
        self.net = nn.ModuleList()
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim), nn.ReLU()
            ))
        self.net.append(nn.Sequential(
            nn.Conv2d(hidden[-1], inner_dim, 4, 1, 0),
            nn.BatchNorm2d(inner_dim), nn.ReLU()
        ))
        self.net.append(nn.Sequential(
            ReshapeBlock([inner_dim]), nn.Linear(inner_dim, output_dim)
        ))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class FCEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None):
        super(FCEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [512, 512, 512] if hidden is None else hidden
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            ReshapeBlock([input_dim * 64 * 64]),
            Linear(input_dim * 64 * 64, hidden[0]), nn.ReLU()
        ))
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(Linear(in_dim, out_dim), nn.ReLU()))
        self.net.append(Linear(hidden[-1], output_dim))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None, inner_dim=64):
        super(CNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [64, 64, 32, 32] if hidden is None else hidden
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            ReshapeBlock([input_dim, 1, 1]),
            nn.ConvTranspose2d(input_dim, inner_dim, 1, 1, 0),
            nn.BatchNorm2d(inner_dim), nn.LeakyReLU()
        ))
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(inner_dim, hidden[0], 4, 1, 0),
            nn.BatchNorm2d(hidden[0]), nn.LeakyReLU()
        ))
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim), nn.LeakyReLU()
            ))
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(hidden[-1], output_dim, 4, 2, 1), nn.Sigmoid()
        ))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class AutoEncoder(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, logger=None):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.total_size = self.concept_size * self.num_concept
        self.rule_size = args.rule_size
        self.num_rule = args.num_rule
        self.image_size = args.image_size
        self.image_channel = args.image_channel
        self.image_shape = (self.image_channel, self.image_size, self.image_size)
        self.device = device
        self.logger = logger
        self.x_dist = dist.Normal(std_fix=self.x_sigma)
        self.f_dist = dist.Normal(std_max=0.05)
        self.z_dist = dist.Normal(std_fix=self.z_sigma)
        self.num_node = args.num_node
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.encoder = CNNEncoder(
            self.image_channel, self.total_size,
            hidden=args.enc, inner_dim=args.enc_inner_dim
        )
        self.decoder = CNNDecoder(
            self.total_size, self.image_channel,
            hidden=args.dec, inner_dim=args.dec_inner_dim
        )
        self.prior_means = nn.Parameter(
            torch.zeros(self.num_rule, self.num_concept, self.rule_size).to(device).float(),
            requires_grad=False
        )
        self.prior_precisions = nn.Parameter(
            torch.zeros(self.num_rule, self.num_concept, self.rule_size, self.rule_size).to(device).float(),
            requires_grad=False
        )
        self.prior_weights = nn.Parameter(
            torch.ones(self.num_rule, self.num_concept).to(device).float() / self.num_concept,
            requires_grad=False
        )
        self.batch_count = 0

    def encode(self, x, clone=False):
        batch_shape = x.size()[:-3]
        x = x.reshape(-1, *self.image_shape)
        encoder = self.encoder if not clone else copy.deepcopy(self.encoder)
        z_post = encoder(x).reshape(*batch_shape, self.num_concept, self.concept_size, 1)
        zs = self.z_dist.sample(z_post)
        results = {'z': zs, 'z_post': z_post}
        return results

    def decode(self, z):
        batch_shape, n, d = z.size()[:-2], z.size(-2), z.size(-1)
        z = z.reshape(-1, n * d)
        x_mean = self.decoder(z).reshape(*batch_shape, *self.image_shape)
        x_param = cat_sigma(x_mean, self.x_sigma)
        xs = self.x_dist.sample(x_param).detach()
        results = {'xs': xs, 'x_mean': x_mean, 'x_param': x_param}
        return results

    def init_prior_params(self, f):
        for i in range(self.num_concept):
            f_c = f[:, i].detach().cpu().numpy()
            gmm = BayesianGaussianMixture(
                n_components=self.num_rule, covariance_type='full',
                init_params='kmeans', tol=1e-1
            )
            gmm.fit(f_c)
            order = np.argsort(gmm.weights_)[::-1][:self.num_rule]
            self.prior_means.data[:, i] = torch.from_numpy(gmm.means_[order]).float().to(self.device)
            self.prior_precisions.data[:, i] = torch.from_numpy(gmm.precisions_[order]).float().to(self.device)
            self.prior_weights.data[:, i] = torch.from_numpy(gmm.weights_[order]).float().to(self.device)

    def get_prior_params(self):
        return self.prior_means, self.prior_precisions, self.prior_weights

    def estimate_total_correlation(self, qz_samples, qz_params):
        """
        :params:
            qz_samples (bs, ld, dim)
            qz_params  (bs, ld, dim, 2)
        :return:
            log_prob (bs)
        """
        # Only take a sample subset of the samples
        bs, ld, dim, _ = qz_params.size()
        log_qz_i = self.z_dist.log_density(
            qz_samples.reshape(1, bs, ld, dim).expand(bs, -1, -1, -1),
            qz_params.reshape(bs, 1, ld, dim, 2).expand(-1, bs, -1, -1, -1)
        )   # bs, num_sample, ld, dim
        log_qz_dim = log_qz_i.sum(-1)
        marginal_entropy = logsumexp(log_qz_dim, dim=0).mean(0)     # ld
        log_density = self.log_mixture(qz_samples)
        return torch.sum(marginal_entropy - log_density)

    def _compute_log_prob(self, f):
        f = f.unsqueeze(1).expand(-1, self.num_rule, -1, -1)
        mean, precision, _ = self.get_prior_params()
        mean = mean.unsqueeze(0).expand(f.size(0), -1, -1, -1)
        precision = precision.unsqueeze(0).expand(f.size(0), -1, -1, -1, -1)
        log_prob = dist.MultivariateNormal().log_density(f, mean, precision)
        return log_prob

    def assign_indicator(self, f):
        kld_mat = self._compute_log_prob(f).exp() * self.prior_weights.unsqueeze(0)    # bs, nr, nc
        indicator = kld_mat.argmax(1).detach().clone()
        return indicator

    def log_mixture(self, f):
        kld_mat = self._compute_log_prob(f)  # bs, nr, nc
        post_weights = kld_mat.exp() * self.prior_weights.unsqueeze(0)
        indicator = post_weights.argmax(1, keepdim=True).detach().clone()
        kld = kld_mat.gather(1, indicator).squeeze(1).sum(-1).mean()
        return kld

    def loss(self, loss_dicts):
        bs, n = loss_dicts['z_post'].size(0), loss_dicts['z_post'].size(1)
        log_px = self.x_dist.log_density(loss_dicts['gt'], loss_dicts['x_param']).reshape(bs, -1).sum(1).mean()
        kld_z = self.z_dist.kld(loss_dicts['zt_post'], loss_dicts['zt_prior']).reshape(bs, -1).sum(-1).mean()
        kld_f = self.f_dist.kld(loss_dicts['f_post'], loss_dicts['f_prior']).reshape(bs, -1).sum(-1).mean()
        kld_f_gmm_prior = torch.tensor([0.]).to(self.device)
        if self.beta_prior > 0 or self.batch_count == 0:
            self.init_prior_params(loss_dicts['f'])
            kld_f_gmm_prior = self.estimate_total_correlation(loss_dicts['f'], loss_dicts['f_post'])
        self.batch_count += 1
        f_mean = loss_dicts['f_post'][..., 0].abs().mean()
        f_std = loss_dicts['f_post'][..., 1].sigmoid().abs().mean()
        z_mean = loss_dicts['z_post'][..., 0].abs().mean()
        modified_elbo = log_px - self.beta_z * kld_z - self.beta_func * kld_f - self.beta_prior * kld_f_gmm_prior
        vae_logger = {
            'recon': log_px.detach().clone(),
            'kld_z': kld_z.detach().clone(),
            'kld_f': kld_f.detach().clone(),
            'kld_prior': kld_f_gmm_prior.detach().clone(),
            'f_mean': f_mean.detach().clone(),
            'f_std': f_std.detach().clone(),
            'z_mean': z_mean.detach().clone(),
        }
        return modified_elbo, vae_logger

    def forward(self, x):
        pass

    def check_disentangle(self, z, dim, params):
        bs = z.size(0)
        z_interpolate = self.z_dist.interpolate(bs, params)  # n, num, nc, cs
        num = z_interpolate.size(1)
        zs = torch.cat([z.clone().detach().unsqueeze(1) for _ in range(num)], 1)  # n, num, nc, cs
        zs[:, :, dim] = z_interpolate[:, :, dim]
        results = self.decode(zs.reshape(-1, self.num_concept, self.concept_size))
        img_s, img_c = self.image_size, self.image_channel
        return results['x_mean'].reshape(bs, num, img_c, img_s, img_s), num
