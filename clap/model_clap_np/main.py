import random
import math
import torch
import torch.nn as nn
import lib.dist as dist
from lib.building_block import BaseNetwork
from lib.utils import split_by_mask, combine_by_mask, cat_sigma, logsumexp
from model_clap_np.encoder import ConvEncoder
from model_clap_np.decoder import ConvDecoder, DeepConvDecoder


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden=None):
        super(MLP, self).__init__()
        self.hidden = hidden if hidden is not None else []
        self.dims_in = [dim_in] + self.hidden
        self.dims_out = self.hidden + [dim_out]
        self.net = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dims_in, self.dims_out)):
            if i > 0:
                self.net.append(nn.ReLU())
            self.net.append(nn.Linear(d_in, d_out))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class EncoderDecoder(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.z_size = self.concept_size * self.num_concept
        self.img_size = args.image_size
        self.image_channel = args.image_channel
        self.size_dataset = args.num_sample
        self.device = device
        self.x_dist = dist.Normal()
        self.z_dist = dist.Normal()
        self.f_dist = dist.Normal()
        self.encoder = ConvEncoder(self.image_channel, self.z_size)
        if args.decoder == 'conv':
            self.decoder = ConvDecoder(self.z_size, self.image_channel)
        elif args.decoder == 'deep_conv':
            self.decoder = DeepConvDecoder(self.z_size, self.image_channel)
        else:
            raise NotImplementedError

    def encode(self, x):
        b = x.size(0)
        x = x.reshape(b, self.image_channel, self.img_size, self.img_size)
        z_post = self.encoder(x).reshape(b, self.num_concept, self.concept_size)
        z_post = cat_sigma(z_post, self.z_sigma)
        zs = self.z_dist.sample(z_post)
        results = {'z': zs, 'z_post': z_post}
        return results

    def decode(self, z):
        b = z.size(0)
        z = z.reshape(b, -1)
        x_mean = self.decoder(z).reshape(b, self.image_channel, self.img_size, self.img_size)
        x_param = cat_sigma(x_mean, self.x_sigma)
        xs = self.x_dist.sample(x_param).detach().clamp(0., 1.)
        results = {'xs': xs, 'x_mean': x_mean, 'x_param': x_param}
        return results

    @staticmethod
    def normal_regular(params, mu=0., log_std=0.):
        single_params = torch.tensor([mu, log_std], dtype=params.dtype, device=params.device)
        expand_dim = [1] * (len(params.size()) - 1)
        regular_params = single_params.expand(*expand_dim, 2).expand(params.size())
        regular_term = dist.Normal().kld(params, regular_params)
        return regular_term

    def estimate_total_correlation(self, samples, params, max_samples=2048):
        """
        :param:
            samples: FloatTensor (bs, n, nc, cs)
            params: FloatTensor (bs, n, nc, cs, 2)
            max_samples: int, the most number of samples to calculate
        :return:
            total correlation: FloatTensor (1)
        """
        # Only take a sample subset of the samples
        nc, cs = self.num_concept, self.concept_size
        num_sample = min(max_samples, samples.size(0))
        samples = samples.reshape(-1, nc * cs)
        params = params.reshape(-1, nc * cs, 2)
        b = samples.size(0)
        select_indices = torch.randperm(b).to(samples.device)[:num_sample]
        samples = samples.index_select(0, select_indices)
        log_qz_i = self.z_dist.log_density(
            samples.reshape(1, num_sample, -1).expand(b, -1, -1),
            params.reshape(b, 1, -1, 2).expand(-1, num_sample, -1, -1)
        )
        # computes - log q(z_i) summed over mini-batch
        log_qz_dim = torch.cat([e.sum(-1, keepdim=True) for e in log_qz_i.split(cs, -1)], -1)

        marginal_entropy = logsumexp(log_qz_dim, dim=0).mean(0).sum() - \
            2 * log_qz_dim.size(-1) * math.log(num_sample)
        # computes - log q(z) summed over mini-batch
        joint_entropy = logsumexp(log_qz_i.sum(-1), dim=0).mean() - 2 * math.log(num_sample)
        dependence = joint_entropy - marginal_entropy
        return dependence

    def loss(self, loss_dicts):
        bs = loss_dicts['f_post'].size(0)
        log_px = self.x_dist.log_density(loss_dicts['gt'], loss_dicts['x_param']).reshape(bs, -1).sum(1).mean()
        kld_f = self.f_dist.kld(loss_dicts['f_post'], loss_dicts['f_prior']).reshape(bs, -1).sum(1).mean()
        kld_z = self.z_dist.kld(loss_dicts['zt_post'], loss_dicts['zt_prior']).reshape(bs, -1).sum(1).mean()
        regular_z = self.normal_regular(loss_dicts['zt_post']).reshape(bs, -1).sum(1).mean()
        dependence = torch.zeros(1).to(self.device)
        if self.beta_mi > 0:
            dependence = self.estimate_total_correlation(loss_dicts['z'], loss_dicts['z_post'])
        modified_elbo = log_px - self.beta_z * kld_z - self.beta_f * kld_f - self.beta_mi * dependence
        vae_logger = {
            'recon': log_px.detach(),
            'kld_z': kld_z.detach(),
            'kld_f': kld_f.detach(),
            'dependence': dependence.detach(),
            'regular_z': regular_z.detach()
        }
        return modified_elbo, vae_logger

    def forward(self, x):
        pass

    def check_disentangle_contiguous(self, z, dim, params=None):
        bs = z.size(0)
        lower, upper, num = params['lower'][dim], params['upper'][dim], params['num']   # cs
        diff = (upper - lower) / (num - 1)
        z_interpolate = torch.stack([lower + diff * i for i in range(num)], dim=0)  # num, cs
        z_interpolate = z_interpolate[None].expand(bs, -1, -1)  # n, num, cs
        zs = torch.cat([z.clone().detach().unsqueeze(1) for _ in range(num)], 1)  # n, num, nc, cs
        zs[:, :, dim] = z_interpolate
        results = self.decode(zs.reshape(-1, self.num_concept, self.concept_size))
        img_s, img_c = self.img_size, self.image_channel
        return results['x_mean'].reshape(bs, num, img_c, img_s, img_s), num


class CompositionalFunction(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.concept_size = args.concept_size
        self.label_size = args.label_size
        self.num_node = args.num_node
        self.f_size = args.f_size
        self.embed_size = args.embed_size
        self.to_concept = MLP(self.concept_size, self.embed_size, args.embed_hidden)
        self.to_label = MLP(self.label_size, self.embed_size, args.embed_hidden)
        self.aggregator = MLP(self.embed_size * 2, args.aggregator_size, args.aggregator_hidden)
        self.f_params = MLP(args.aggregator_size, args.f_size * 2, args.func_param_hidden)
        self.func = MLP(self.f_size + self.embed_size, self.concept_size, args.func_hidden)
        self.device = device

    def get_f_params(self, z, label):
        """
        :param:
            z: FloatTensor, size (b * n * cs)
            label: LongTensor, size (b * n * ls)
        :return:
            f_params: FloatTensor, size (b * fs * 2)
        """
        z = self.to_concept(z)
        label = self.to_label(label)
        h = self.aggregator(torch.cat([z, label], -1))
        f_params = self.f_params(h.mean(1))
        f_params = f_params.reshape(-1, self.f_size, 2)
        return f_params

    def get_z_params(self, f, zc, yt, yc):
        """
        :param:
            f: FloatTensor, size (b * fs)
            zc: FloatTensor, size (b * nc * cs)
            yt: FloatTensor, size (b * nt * ls)
            yc: FloatTensor, size (b * nc * ls)
        :return:
            c: FloatTensor, size (b * nr)
            z_params: FloatTensor, size (b * n * cs * 2)
        """
        yt = self.to_label(yt)
        f = f[:, None].expand(-1, yt.size(1), -1)
        z_params = self.func(torch.cat([f, yt], -1))
        z_params = z_params.reshape(-1, yt.size(1), self.concept_size)
        z_params = cat_sigma(z_params, self.z_sigma)
        return z_params


class MainNet(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, pool_size=512):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.image_size = args.image_size
        self.image_channel = args.image_channel
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.num_missing = args.num_missing
        self.z_size = self.concept_size * self.num_concept
        self.f_size = args.f_size
        self.label_size = args.label_size
        self.z_dist = dist.Normal()
        self.f_dist = dist.Normal()
        self.x_dist = dist.Normal()
        self.num_node = args.num_node
        self.device = device
        self.pool_size = pool_size
        self.baseNet = EncoderDecoder(args, device, global_step=global_step)
        self.register_buffer(
            'f_params', torch.zeros(self.pool_size, self.num_concept, self.f_size, 2).to(device)
        )
        self.functions = nn.ModuleList([
            CompositionalFunction(args, device, global_step=global_step) for _ in range(self.num_concept)
        ])

    @staticmethod
    def batch_apply(x, dim, nets, cat_dim, func_name, args):
        results = []
        for d in range(len(nets)):
            local_args = []
            for sub_x, sub_dim in zip(x, dim):
                local_args.append(
                    sub_x.index_select(sub_dim, torch.tensor([d], device=sub_x.device)).squeeze(sub_dim)
                )
            func = getattr(nets[d], func_name)
            x_dim = func(*local_args, *args)
            if type(x_dim) is tuple:
                x_dim = [x_dim_element.unsqueeze(cat_dim) for x_dim_element in x_dim]
            else:
                x_dim = [x_dim.unsqueeze(cat_dim)]
            results.append(x_dim)
        results = [[row[i] for row in results] for i in range(len(results[0]))]
        results = tuple([torch.cat(row, cat_dim) for row in results])
        if len(results) == 1:
            return results[0]
        else:
            return results

    def save_to_pool(self, upd_f):
        self.f_params = torch.cat([upd_f, self.f_params], 0)[:self.pool_size]

    def random_read_from_pool(self, bs):
        random_index = list(range(self.pool_size))
        random.shuffle(random_index)
        mask = torch.tensor(random_index, device=self.device).long()
        return self.f_params.index_select(0, mask)[:bs]

    def forward(self, images, labels):
        # concat input
        b, n, zs, ims, imc = images.size(0), self.num_node, self.z_size, self.image_size, self.image_channel
        cs, nc = self.concept_size, self.num_concept
        results = self.baseNet.encode(images.reshape(-1, imc, ims, ims))

        # build mask
        num_missing = random.choice(list(range(self.num_missing[0], self.num_missing[1] + 1)))
        mask = images.new_zeros(n).long()
        random_index = list(range(n))
        random.shuffle(random_index)
        mask[random_index[:num_missing]] = 1

        # gnp regression
        label_t, label_c = split_by_mask(labels, mask, 1)
        results['z'] = results['z'].reshape(b, n, nc, cs)
        results['zt'], results['zc'] = split_by_mask(results['z'], mask, 1)
        results['z_post'] = results['z_post'].reshape(b, n, nc, cs, 2)
        results['f_post'] = self.batch_apply([results['z']], [2], self.functions, 1, 'get_f_params', (labels,))
        results['f'] = self.f_dist.sample(results['f_post'])
        results['f_prior'] = self.batch_apply([results['zc']], [2], self.functions, 1, 'get_f_params', (label_c,))
        self.save_to_pool(results['f_post'].detach().clone())
        results['zt_prior'] = self.batch_apply(
            [results['f'], results['zc']], [1, 2], self.functions, 2, 'get_z_params', (label_t, label_c)
        )
        results['zt_post'], results['zc_post'] = split_by_mask(results['z_post'], mask, 1)
        results_dec = self.baseNet.decode(results['zt'].reshape(-1, nc, cs))
        x_recon = results_dec['x_mean']
        x_param = results_dec['x_param'].reshape(b, -1, imc, ims, ims, 2)
        ground_truth = split_by_mask(images, mask, 1)[0]

        # vae
        loss_dicts = {
            **results, 'gt': ground_truth, 'x_param': x_param
        }
        elbo_vae, logger_vae = self.baseNet.loss(loss_dicts)
        elbo = elbo_vae
        logger_other = {
            'elbo': elbo.detach()
        }
        return elbo, {**logger_vae, **logger_other}, x_recon

    def evaluate(self, images, labels, num_pred=1):
        # concat input
        b, n, zs, ims, imc = images.size(0), self.num_node, self.z_size, self.image_size, self.image_channel
        cs, nc = self.concept_size, self.num_concept
        results = self.baseNet.encode(images.reshape(-1, imc, ims, ims))

        # build mask
        mask = images.new_zeros(n).long()
        random_index = list(range(n))
        random.shuffle(random_index)
        mask[random_index[:num_pred]] = 1

        # gnp regression
        label_t, label_c = split_by_mask(labels, mask, 1)
        results['z'] = results['z'].reshape(b, n, nc, cs)
        results['zt'], results['zc'] = split_by_mask(results['z'], mask, 1)
        results['z_post'] = results['z_post'].reshape(b, n, nc, cs, 2)
        results['f_post'] = self.batch_apply([results['z']], [2], self.functions, 1, 'get_f_params', (labels,))
        results['f'] = self.f_dist.sample(results['f_post'])
        results['f_prior'] = self.batch_apply([results['zc']], [2], self.functions, 1, 'get_f_params', (label_c,))
        results['zt_prior'] = self.batch_apply(
            [results['f'], results['zc']], [1, 2], self.functions, 2, 'get_z_params', (label_t, label_c)
        )
        results['zt_post'], results['zc_post'] = split_by_mask(results['z_post'], mask, 1)
        results_dec = self.baseNet.decode(results['zt'].reshape(-1, nc, cs))
        x_param = results_dec['x_param'].reshape(b, -1, imc, ims, ims, 2)
        ground_truth = split_by_mask(images, mask, 1)[0]

        # vae
        loss_dicts = {
            **results, 'gt': ground_truth, 'x_param': x_param
        }
        elbo_vae, logger_vae = self.baseNet.loss(loss_dicts)
        elbo = elbo_vae
        logger_other = {
            'elbo': elbo.detach()
        }
        return {**logger_vae, **logger_other}

    def metric(self, images, labels, num_pred=1):
        # concat input
        b, n, zs, ims, imc = images.size(0), self.num_node, self.z_size, self.image_size, self.image_channel
        cs, nc = self.concept_size, self.num_concept
        results = self.baseNet.encode(images.reshape(-1, imc, ims, ims))

        # build mask
        mask = images.new_zeros(n).long()
        random_index = list(range(n))
        random.shuffle(random_index)
        mask[random_index[:num_pred]] = 1

        # gnp regression
        label_t, label_c = split_by_mask(labels, mask, 1)
        results['z'] = results['z'].reshape(b, n, nc, cs)
        results['zt'], results['zc'] = split_by_mask(results['z'], mask, 1)
        results['z_post'] = results['z_post'].reshape(b, n, nc, cs, 2)
        results['f_post'] = self.batch_apply([results['zc']], [2], self.functions, 1, 'get_f_params', (label_c,))
        results['f'] = self.f_dist.sample(results['f_post'])
        results['zt_prior'] = self.batch_apply(
            [results['f'], results['zc']], [1, 2], self.functions, 2, 'get_z_params', (label_t, label_c)
        )
        results['zt_post'], results['zc_post'] = split_by_mask(results['z_post'], mask, 1)
        results['zt'] = self.z_dist.sample(results['zt_prior'])
        # results['zt'] = results['zt_prior'][..., 0]
        results_dec = self.baseNet.decode(results['zt'].reshape(-1, nc, cs))
        x_recon = results_dec['x_mean'].reshape(b, -1, imc, ims, ims)
        ground_truth = split_by_mask(images, mask, 1)[0]

        logger_other = {
            'MSE': (ground_truth - x_recon).pow(2).reshape(b, -1).sum(-1).mean().detach()
        }
        return logger_other

    def test(self, images, labels, pred_num=1, mask=None):
        bs, n, ds, ims, imc = images.size(0), images.size(1), self.z_size, self.image_size, self.image_channel
        cs, nc = self.concept_size, self.num_concept
        results_enc = self.baseNet.encode(images.reshape(-1, imc, ims, ims))
        results_dec = self.baseNet.decode(results_enc['z'])

        # build mask
        if mask is None:
            mask = images.new_zeros(n).long()
            random_index = list(range(n))
            random.shuffle(random_index)
            for i in range(pred_num):
                mask[random_index[i]] = 1
            results_enc['mask'] = mask
        else:
            results_enc['mask'] = mask

        # gp regression
        label_t, label_c = split_by_mask(labels, mask, 1)
        z = results_enc['z'].reshape(bs, n, nc, cs)
        results_enc['z_post'] = results_enc['z_post'].reshape(bs, n, nc, cs, -1)
        results_enc['zt'], results_enc['zc'] = split_by_mask(z, mask, 1)
        results_enc['f_post'] = self.batch_apply(
            [results_enc['zc']], [2], self.functions, 1, 'get_f_params', (label_c,)
        )
        results_enc['f'] = self.f_dist.sample(results_enc['f_post'])
        results_enc['zt_post'] = self.batch_apply(
            [results_enc['f'], results_enc['zc']], [1, 2], self.functions, 2, 'get_z_params', (label_t, label_c)
        )
        results_enc['zt'] = self.z_dist.sample(results_enc['zt_post'])
        results_gen_dec = self.baseNet.decode(results_enc['zt'].reshape(-1, nc, cs))

        # deal output
        gt_recon = results_dec['x_mean'].reshape(bs, n, imc, ims, ims)
        gt_recon_t, gt_recon_c = split_by_mask(gt_recon, mask, 1)
        images_t, images_c = split_by_mask(images, mask, 1)
        gt_pred_t = results_gen_dec['x_mean'].reshape(bs, -1, imc, ims, ims)
        gt_pred = combine_by_mask(gt_pred_t, images_c, mask, 1)

        return gt_recon, gt_pred, results_enc

    def disentangle(self, images, labels):
        bs, n, ds, ims, imc = images.size(0), images.size(1), self.z_size, self.image_size, self.image_channel
        cs, nc = self.concept_size, self.num_concept
        results_enc = self.baseNet.encode(images.reshape(-1, imc, ims, ims))
        z = results_enc['z'].reshape(bs, n, nc, cs)
        results_enc['f_post'] = self.batch_apply(
            [z], [2], self.functions, 1, 'get_f_params', (labels,)
        )
        results_enc['f'] = self.f_dist.sample(results_enc['f_post'])
        return results_enc['f'], results_enc['f_post']

    def encode(self, x, label):
        b, n = x.size(0), x.size(1)
        x = x.reshape(-1, self.image_channel, self.image_size, self.image_size)
        z_mean = self.baseNet.encoder(x).reshape(b, n, -1)
        return z_mean

    def generate(self, images, label_c, label_range, fps=24):
        b, n = images.size(0), self.num_node
        nc, cs, nf, ims, imc = self.num_concept, self.concept_size, self.f_size, self.image_size, self.image_channel
        results = self.baseNet.encode(images.reshape(-1, imc, ims, ims))
        zc = results['z'].reshape(b, n, nc, cs)
        label_start, label_end = label_range[..., 0], label_range[..., 1]
        labels = [label_start + i * (label_end - label_start) / (fps - 1) for i in range(fps)]
        labels = torch.cat([label[None] for label in labels], 0)[None].expand(b, -1, -1)
        f = self.f_dist.sample(self.random_read_from_pool(b))
        zs_post = self.batch_apply([f, zc], [1, 2], self.functions, 2, 'get_z_params', (labels, label_c))
        zs = self.z_dist.sample(zs_post)
        results_dec = self.baseNet.decode(zs.reshape(-1, nc, cs))
        gen = results_dec['x_mean'].reshape(b, fps, imc, ims, ims)
        return gen
