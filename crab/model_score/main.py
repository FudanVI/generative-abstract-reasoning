import random
import torch
import torch.nn as nn
from lib.building_block import BaseNetwork
import lib.dist as dist
from lib.utils import split_by_mask, combine_by_mask
from model_score.predictor import Predictor
from model_score.autoencoder import AutoEncoder


class MainNet(nn.Module, BaseNetwork):
    def __init__(self, args, device, global_step=0, logger=None):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.image_size = args.image_size
        self.image_channel = args.image_channel
        self.image_shape = (self.image_channel, self.image_size, self.image_size)
        self.concept_size = args.concept_size
        self.num_concept = args.num_concept
        self.num_rule = args.num_rule
        self.rule_size = args.rule_size
        self.total_size = self.concept_size * self.num_concept
        self.z_dist = dist.Normal(std_fix=self.z_sigma)
        self.num_node = args.num_node
        self.device = device
        self.logger = logger
        self.baseNet = AutoEncoder(args, device, global_step=global_step)
        self.predictor = Predictor(args, device, global_step=global_step)

    def state_dict_base(self):
        return {**self.baseNet.state_dict_base(), **self.predictor.state_dict_base()}

    def load_state_dict_base(self, state_dict):
        self.baseNet.load_state_dict_base(state_dict)
        self.predictor.load_state_dict_base(state_dict)

    def make_repr_fn(self):

        def represent_fn(x):
            x = x.to(self.device)
            encode_results = self.baseNet.encode(x)
            encode_results = {k: v.detach().cpu() for k, v in encode_results.items()}
            encode_results = {**encode_results, 'z_dist': self.z_dist}
            return encode_results

        return represent_fn

    def forward(self, p):
        # concat input
        bs, imc, ims = p.size(0), self.image_channel, self.image_size
        n, nc, cs = p.size(1), self.num_concept, self.concept_size
        results_enc = self.baseNet.encode(p)

        # build mask
        mask = p.new_zeros(n).long()
        random_index = list(range(n))
        random.shuffle(random_index)
        mask[random_index[:1]] = 1

        # prediction
        results_pred = self.predictor(results_enc['z'], mask)
        results_enc['zt'], _ = split_by_mask(results_enc['z'], mask, 1)
        results_enc['zt_post'], _ = split_by_mask(results_enc['z_post'], mask, 1)
        results_dec = self.baseNet.decode(results_enc['zt'].reshape(-1, nc, cs))
        pt, _ = split_by_mask(p, mask, 1)
        results_dec['x_param'] = results_dec['x_param'].reshape(bs, -1, imc, ims, ims, 2)

        # vae
        loss_dicts = {**results_enc, **results_pred, **results_dec, 'gt': pt}
        elbo_vae, logger_vae = self.baseNet.loss(loss_dicts)
        elbo = elbo_vae
        logger_other = {'elbo': elbo.detach()}
        return elbo, {**logger_vae, **logger_other}, loss_dicts

    def metric(self, p, s, a, pred_num=1):
        # concat input
        bs, imc, ims = p.size(0), self.image_channel, self.image_size
        n, nc, cs = p.size(1), self.num_concept, self.concept_size
        results_enc = self.baseNet.encode(p)
        results_enc_s = self.baseNet.encode(s)

        # MSE
        mask = p.new_zeros(n).long()
        random_index = list(range(n))
        random.shuffle(random_index)
        mask[random_index[:pred_num]] = 1
        results_pred = self.predictor.predict(results_enc['z_post'][..., 0], mask)
        results_dec = self.baseNet.decode(results_pred['zt_prior'][..., 0])
        pt, _ = split_by_mask(p, mask, 1)
        mse = (pt - results_dec['x_mean']).pow(2).reshape(bs, -1).sum(-1).mean()

        # ACC
        mask = p.new_zeros(self.num_node).long()
        mask[-1] = 1
        results_pred = self.predictor.predict(results_enc['z_post'][..., 0], mask)
        selected_index = (results_pred['zt_prior'][..., 0] - results_enc_s['z_post'][..., 0])\
            .pow(2).sum([2, 3]).argmin(1)
        concept_acc = torch.eq(selected_index, a).float().mean()
        results_dec = self.baseNet.decode(results_pred['zt_prior'][..., 0])
        selected_index = (results_dec['x_mean'] - s).pow(2).sum([2, 3, 4]).argmin(1)
        pixel_acc = torch.eq(selected_index, a).float().mean()

        logger_metric = {
            'MSE': mse.detach(),
            'Concept ACC': concept_acc.detach(),
            'Pixel ACC': pixel_acc.detach()
        }

        return logger_metric

    def test(self, p, mask=None, pred_num=1):
        # concat input
        n, nc, cs = p.size(1), self.num_concept, self.concept_size
        results_enc = self.baseNet.encode(p)
        results_dec = self.baseNet.decode(results_enc['z_post'][..., 0])

        # build mask
        if mask is None:
            mask = p.new_zeros(n).long()
            random_index = list(range(n))
            random.shuffle(random_index)
            mask[random_index[:pred_num]] = 1
        results_enc['mask'] = mask

        # prediction
        results_pred = self.predictor.predict(results_enc['z_post'][..., 0], mask)
        results_gen_dec = self.baseNet.decode(results_pred['zt_prior'][..., 0])
        results_pred['indicator'] = self.baseNet.assign_indicator(results_pred['func'])

        # deal output
        gt_recon_t, gt_recon_c = split_by_mask(results_dec['x_mean'], mask, 1)
        gt_pred = combine_by_mask(results_gen_dec['x_mean'], gt_recon_c, mask, 1)

        return results_dec['x_mean'], gt_pred, {**results_enc, **results_pred}

    def predict(self, x, t, mask):
        pass

    def generate(self, bs):
        pass
