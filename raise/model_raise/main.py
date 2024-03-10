import random
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from lib.building_block import BaseNetwork
from lib.utils import split_by_mask, combine_by_mask, masked_mean
from model_raise.gat_predictor import NPPredictor
from model_raise.autoencoder import AutoEncoder


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
        self.total_size = self.concept_size * self.num_concept
        self.num_node = args.num_node
        self.device = device
        self.logger = logger
        self.baseNet = AutoEncoder(args, device, global_step=global_step)
        self.gnp = NPPredictor(args, device, global_step=global_step)

    def forward(self, panel, label=None):
        # concat input
        bs, imc, ims = panel.size(0), self.image_channel, self.image_size
        n, nc, cs = panel.size(1), self.num_concept, self.concept_size
        results_enc = self.baseNet.encode(panel)

        # build mask
        mask = panel.new_zeros(n).long()
        random_index = list(range(n))
        random.shuffle(random_index)
        mask[random_index[:1]] = 1

        # gnp regression
        results_gp = self.gnp(results_enc['z'], mask)
        results_enc['zt'], _ = split_by_mask(results_enc['z'], mask, 1)
        results_enc['zt_post'], _ = split_by_mask(results_enc['z_post'], mask, 1)
        results_dec = self.baseNet.decode(results_enc['zt'].reshape(-1, nc, cs))
        panel_tgt, _ = split_by_mask(panel, mask, 1)
        results_dec['x_param'] = results_dec['x_param'].reshape(bs, -1, imc, ims, ims, 2)

        # vae
        loss_dicts = {
            **results_enc, **results_gp, **results_dec,
            'gt': panel_tgt, 'rule_label': label
        }
        elbo_vae, logger_vae = self.baseNet.loss(loss_dicts)
        elbo = elbo_vae
        logger_other = {'elbo': elbo.detach()}
        return elbo, {**logger_vae, **logger_other}, loss_dicts

    def metric(self, p, s, a, l, pred_num=1):
        # concat input
        bs, imc, ims = p.size(0), self.image_channel, self.image_size
        n, nc, cs = p.size(1), self.num_concept, self.concept_size
        results_enc = self.baseNet.encode(p)
        results_enc_s = self.baseNet.encode(s)

        # build mask
        mask = p.new_zeros(self.num_node).long()
        mask[-1] = 1

        # predict results
        results_gp = self.gnp.predict(results_enc['z_post'][..., 0], mask)
        results_gp_full = self.gnp(results_enc['z_post'][..., 0], mask)
        results_dec = self.baseNet.decode(results_gp['zt_prior'][..., 0])

        # MSE
        pt, _ = split_by_mask(p, mask, 1)
        mse = (pt - results_dec['x_mean']).pow(2).reshape(bs, -1).sum(-1).mean()

        # Rule ACC
        l_mask = 1. - torch.eq(l, -1).float()
        log_pr_matrix = masked_mean(
            torch.eq(results_gp_full['f_post'].argmax(-1).unsqueeze(1), l.unsqueeze(2)).float(),
            l_mask.unsqueeze(2).expand(-1, -1, nc), dim=0
        )
        row_ind, col_ind = linear_sum_assignment(-log_pr_matrix.detach().cpu().numpy())
        rule_acc_post = log_pr_matrix[row_ind, col_ind].mean()
        log_pr_matrix = masked_mean(
            torch.eq(results_gp_full['f_prior'].argmax(-1).unsqueeze(1), l.unsqueeze(2)).float(),
            l_mask.unsqueeze(2).expand(-1, -1, nc), dim=0
        )
        row_ind, col_ind = linear_sum_assignment(-log_pr_matrix.detach().cpu().numpy())
        rule_acc_prior = log_pr_matrix[row_ind, col_ind].mean()

        # ACC
        selected_index = (results_gp['zt_prior'][..., 0] - results_enc_s['z_post'][..., 0])\
            .pow(2).sum([2, 3]).argmin(1)
        concept_acc = torch.eq(selected_index, a).float().mean()
        selected_index = (results_dec['x_mean'] - s).pow(2).sum([2, 3, 4]).argmin(1)
        pixel_acc = torch.eq(selected_index, a).float().mean()

        logger_metric = {
            'MSE': mse.detach(),
            'Concept ACC': concept_acc.detach(),
            'I-ACC': pixel_acc.detach(),
            'R-ACC-Post': rule_acc_post.detach(),
            'R-ACC-Prior': rule_acc_prior.detach()
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

        # gp regression
        results_gp = self.gnp.predict(results_enc['z_post'][..., 0], mask)
        results_gen_dec = self.baseNet.decode(results_gp['zt_prior'][..., 0])

        # deal output
        gt_recon_t, gt_recon_c = split_by_mask(p, mask, 1)
        gt_pred = combine_by_mask(results_gen_dec['x_mean'], gt_recon_c, mask, 1)

        return results_dec['x_mean'], gt_pred, {**results_enc, **results_gp}

    def test_rule_acc(self, p, labels):
        # concat input
        n, nc, cs = p.size(1), self.num_concept, self.concept_size
        results_enc = self.baseNet.encode(p)
        mask = p.new_zeros(self.num_node).long()
        mask[-1] = 1
        results_gp_full = self.gnp(results_enc['z_post'][..., 0], mask)

        # Rule ACC
        l_mask = 1. - torch.eq(labels, -1).float()
        log_pr_matrix = masked_mean(
            torch.eq(results_gp_full['f_post'].argmax(-1).unsqueeze(1), labels.unsqueeze(2)).float(),
            l_mask.unsqueeze(2).expand(-1, -1, nc), dim=0
        )
        row_ind, col_ind = linear_sum_assignment(-log_pr_matrix.detach().cpu().numpy())
        rule_acc_post = log_pr_matrix[row_ind, col_ind].detach().clone()
        log_pr_matrix = masked_mean(
            torch.eq(results_gp_full['f_prior'].argmax(-1).unsqueeze(1), labels.unsqueeze(2)).float(),
            l_mask.unsqueeze(2).expand(-1, -1, nc), dim=0
        )
        row_ind, col_ind = linear_sum_assignment(-log_pr_matrix.detach().cpu().numpy())
        rule_acc_prior = log_pr_matrix[row_ind, col_ind].detach().clone()

        logger_metric = {
            'rule_acc_post_mat': rule_acc_post.detach(),
            'rule_acc_post': rule_acc_post.mean().detach(),
            'rule_acc_prior_mat': rule_acc_prior.detach(),
            'rule_acc_prior': rule_acc_prior.mean().detach()
        }
        return logger_metric

    def predict(self, x, t, mask):
        pass

    def generate(self, bs):
        pass
