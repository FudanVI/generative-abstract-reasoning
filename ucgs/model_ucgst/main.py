import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from backbone_vqvae.vqvae import VQVAE
from function_anp.hanp import DeterHANP
from lib.pe import LearnedPositionalEmbedding1D
from lib.building_block import BaseNetwork
from lib.utils import split_by_mask, combine_by_mask


class MainNet(nn.Module, BaseNetwork):

    def __init__(self, args, device, global_step=0):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.device = device
        self.backbone = self.backbone = VQVAE(
            channel=args.channel, n_res_block=args.n_res_block,
            n_res_channel=args.n_res_channel, embed_dim=args.embed_dim, n_embed=args.n_embed)
        self.backbone.load_state_dict(torch.load(
            args.model_root, weights_only=False, map_location='cpu')['model'])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.h_pe = args.latent_size
        self.pe = LearnedPositionalEmbedding1D(16, self.h_pe, dropout=0.)
        self.reasoner = DeterHANP(
            args.rule_size, args.latent_size, args.latent_size,
            self.backbone, args.n_embed, 4, image_layer=12, image_head=8,
            panel_layer=12, panel_head=8, dec_layer=12, dec_head=8, dropout=0.,
            num_slot=8, dec_hidden=args.dec_size
        )
        self.args = args

    def gen_labels(self, b, n):
        tensor_zero = torch.zeros(1, n, self.h_pe).float().to(self.device)
        pe = self.pe(tensor_zero).reshape(1, n, self.h_pe).expand(b, -1, -1)
        return pe

    def generate_pos(self, cpt_prob, id_prob, pe_prob, cpt_ans, id_ans, pe_ans):
        cpt = torch.cat([cpt_prob, cpt_ans], dim=1)
        id = torch.cat([id_prob, id_ans], dim=1)
        pe = torch.cat([pe_prob, pe_ans], dim=1)
        target_y, context_y, target_x, context_x, target_id = [], [], [], [], []
        masks = torch.eye(cpt.size(1)).to(self.device)
        for mask in masks:
            tid, cid = split_by_mask(id, mask, 1)
            target_id.append(tid)
            ty, cy = split_by_mask(cpt, mask, 1)
            tx, cx = split_by_mask(pe, mask, 1)
            target_y.append(ty)
            context_y.append(cy)
            target_x.append(tx)
            context_x.append(cx)
        target_y = torch.stack(target_y, dim=1).flatten(0, 1)
        context_y = torch.stack(context_y, dim=1).flatten(0, 1)
        target_x = torch.stack(target_x, dim=1).flatten(0, 1)
        context_x = torch.stack(context_x, dim=1).flatten(0, 1)
        target_id = torch.stack(target_id, dim=1).flatten(0, 1)
        return target_y, context_y, target_x, context_x, target_id

    def generate_neg(self, cpt_prob, id_prob, pe_prob, cpt_ans, id_ans, pe_ans):
        cpt = torch.cat([cpt_prob, cpt_ans], dim=1)
        id = torch.cat([id_prob, id_ans], dim=1)
        pe = torch.cat([pe_prob, pe_ans], dim=1)
        mask = torch.zeros(cpt.size(1)).long().to(self.device)
        mask[-1] = 1
        target_id, _ = split_by_mask(id, mask, 1)
        target_y, context_y = split_by_mask(cpt, mask, 1)
        target_x, context_x = split_by_mask(pe, mask, 1)
        return target_y, context_y, target_x, context_x, target_id

    def backbone_enc(self, images):
        bs, ims = images.shape[:-3], images.shape[-3:]
        images = images.reshape(-1, *ims)
        with torch.no_grad():
            z, _, zid = self.backbone.encode(images)
        z = z.detach().permute(0, 2, 3, 1).contiguous()
        z = z.reshape(*bs, *z.shape[1:])
        zid = zid.detach().reshape(*bs, *zid.shape[1:])
        return z, zid

    def backbone_dec_code(self, zid):
        bs = zid.shape
        zid = zid.reshape(-1)
        with torch.no_grad():
            z = self.backbone.quantize_b.embed_code(zid)
        z = z.reshape(*bs, -1)
        return z

    def backbone_dec(self, zid):
        z = self.backbone_dec_code(zid)
        bs, ims = z.shape[:-3], z.shape[-3:]
        z = z.reshape(-1, *ims)
        z = z.permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            recon = self.backbone.decode(z)
        recon = recon.reshape(*bs, *recon.shape[-3:])
        return recon

    def ce(self, logits, labels):
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        return cross_entropy(logits, labels)

    def forward(self, problem, answer, distractors):
        b, nc = problem.size(0), problem.size(1)
        n, nd = nc + 1, distractors.shape[1]
        panel = torch.cat([problem, answer, distractors], dim=1)
        cpt, id = self.backbone_enc(panel)
        pe = self.gen_labels(b, n)
        cpt_prob, cpt_ans, cpt_dis = cpt.split([nc, 1, nd], dim=1)
        id_prob, id_ans, id_dis = id.split([nc, 1, nd], dim=1)
        pe_prob, pe_ans = pe.split([nc, 1], dim=1)
        pos_ty, pos_cy, pos_tx, pos_cx, pos_tid = \
            self.generate_pos(cpt_prob, id_prob, pe_prob, cpt_ans, id_ans, pe_ans)
        prior_mu, prior_std, posterior_mu, posterior_std, logits \
            = self.reasoner(pos_cx, pos_cy, pos_tx, pos_ty)
        ll = - self.ce(logits, pos_tid)
        kl = self.reasoner.kl_div(prior_mu, prior_std, posterior_mu, posterior_std).mean()
        elbo = ll - kl
        scores = torch.tensor(0.).to(self.device)
        loss = - elbo + scores
        logger = {
            'loss': loss.detach(),
            'elbo': elbo.detach(),
            'log-likelihood': ll.detach(),
            'kl-divergence': kl.detach(),
            'score': scores.detach()
        }
        return loss, logger

    def metric(self, problem, answer, distractors):
        b, nc = problem.size(0), problem.size(1)
        n, nd = nc + 1, distractors.shape[1]
        full_panel = torch.cat([problem, answer, distractors], dim=1)
        concepts, cid = self.backbone_enc(full_panel)
        concepts_ctx, concepts_ans, concepts_dis = concepts.split([nc, 1, nd], dim=1)
        cid_ctx, cid_ans, cid_dis = cid.split([nc, 1, nd], dim=1)
        mask = concepts.new_zeros(n).long()
        mask[-1] = 1
        labels = self.gen_labels(b, n)
        target_y = concepts_ans
        context_y = concepts_ctx
        target_x, context_x = split_by_mask(labels, mask, 1)
        concepts_cad = torch.cat([concepts_ans, concepts_dis], 1)
        logits = self.reasoner.metric(context_x, context_y, target_x, concepts_cad)
        cid_cad = torch.nn.functional.one_hot(
            torch.cat([cid_ans, cid_dis], 1),
            num_classes=self.args.n_embed)
        scores = (logits.softmax(dim=-1) * cid_cad).sum([2, 3, 4])
        selected_index = scores.argmax(1)
        acc = torch.eq(selected_index, 0).float().sum() / b
        logger_metric = {
            'ACC': acc.detach()
        }
        return logger_metric

    def metric_o3_raven(self, problem, answer, distractors):
        problem = torch.cat([problem, distractors[:, 0:1]], dim=1)
        answer = torch.Tensor([8] * problem.size(0)).to(problem.device)
        return self.metric_o3(problem, answer)

    def metric_vap_raven(self, problem, answer, distractors):
        problem = problem[:, 3:]
        return self.metric(problem, answer, distractors)

    def metric_bp_raven(self, problem, answer, distractors):
        problem = torch.cat([problem, answer], dim=1)
        full1, full2 = problem[:, 0:6], problem[:, 3:9]
        problem1, problem2 = full1[:, :-1], full2[:, :-1]
        answer1, answer2 = full1[:, -1:], full2[:, -1:]
        acc1 = self.metric(problem1, answer1, answer2)['ACC']
        acc2 = self.metric(problem2, answer2, answer1)['ACC']
        results = {'ACC': (acc1 + acc2) / 2}
        return results

    def metric_bp(self, problem, answer, distractors):
        full1, full2 = problem[:, 0:9], problem[:, 9:18]
        problem1, problem2 = full1[:, :-1], full2[:, :-1]
        answer1, answer2 = full1[:, -1:], full2[:, -1:]
        acc1 = self.metric(problem1, answer1, answer2)['ACC']
        acc2 = self.metric(problem2, answer2, answer1)['ACC']
        results = {'ACC': (acc1 + acc2) / 2}
        return results

    def metric_o3(self, problem, answer):
        b, n = problem.size(0), problem.size(1)
        concepts, cid = self.backbone_enc(problem)
        full_panel = torch.cat([problem], dim=1)
        concepts, cid = self.backbone_enc(full_panel)
        pe = self.gen_labels(b, n)
        target_y, context_y, target_x, context_x, target_id = [], [], [], [], []
        masks = torch.eye(concepts.size(1)).to(self.device)
        for mask in masks:
            _tid, _cid = split_by_mask(cid, mask, 1)
            target_id.append(_tid)
            ty, cy = split_by_mask(concepts, mask, 1)
            tx, cx = split_by_mask(pe, mask, 1)
            target_y.append(ty)
            context_y.append(cy)
            target_x.append(tx)
            context_x.append(cx)
        target_y = torch.stack(target_y, dim=1).flatten(0, 1)
        context_y = torch.stack(context_y, dim=1).flatten(0, 1)
        target_x = torch.stack(target_x, dim=1).flatten(0, 1)
        context_x = torch.stack(context_x, dim=1).flatten(0, 1)
        target_id = torch.stack(target_id, dim=1).flatten(0, 1)
        logits = self.reasoner(context_x, context_y, target_x, target_y)[-1]
        target_oh = torch.nn.functional.one_hot(target_id, num_classes=self.args.n_embed)
        scores = (logits.softmax(dim=-1) * target_oh).sum([1, 2, 3, 4])
        scores = scores.reshape(b, n)
        selected_index = scores.argmin(1)
        acc = torch.eq(selected_index, answer).float().sum() / b
        logger_metric = {
            'ACC': acc.detach()
        }
        return logger_metric

    def test(self, problem, answer, distractors):
        b, nc, ci, hi, wi = problem.size()
        n = nc + 1
        full_panel = torch.cat([problem, answer], dim=1)
        concepts, cid = self.backbone_enc(full_panel)
        panel_recon = self.backbone_dec(cid)
        h, w, d = concepts.shape[-3:]
        concepts_ctx, concepts_ans = concepts.split([nc, 1], dim=1)
        mask = concepts.new_zeros(n).long()
        mask[-1] = 1
        context_recon = split_by_mask(panel_recon, mask, 1)[1]
        labels = self.gen_labels(b, n)
        target_y = concepts_ans
        context_y = concepts_ctx
        target_x, context_x = split_by_mask(labels, mask, 1)
        logits = self.reasoner.predict(context_x, context_y, target_x)
        logits = logits.argmax(dim=-1).reshape(b, h, w)
        target_pred = self.backbone_dec(logits)
        logits = self.reasoner.predict_via_post(context_x, context_y, target_x, target_y)
        logits = logits.argmax(dim=-1).reshape(b, h, w)
        target_pred_post = self.backbone_dec(logits)
        panel_pred_vis = combine_by_mask(
            target_pred.unsqueeze(1), context_recon, mask, 1)
        panel_pred_vis_post = combine_by_mask(
            target_pred_post.unsqueeze(1), context_recon, mask, 1)
        results = {
            'panel': full_panel,
            'panel_recon': panel_pred_vis,
            'panel_recon_post': panel_pred_vis_post
        }
        return results

    def test_o3(self, problem, answer):
        b, n, ci, hi, wi = problem.size()
        concepts, cid = self.backbone_enc(problem)
        panel_recon = self.backbone_dec(cid)
        h, w, d = concepts.shape[-3:]
        pe = self.gen_labels(b, n)
        target_y, context_y, target_x, context_x, target_id = [], [], [], [], []
        context_recons = []
        masks = torch.eye(concepts.size(1)).to(self.device)
        for mask in masks:
            _tid, _cid = split_by_mask(cid, mask, 1)
            target_id.append(_tid)
            ty, cy = split_by_mask(concepts, mask, 1)
            tx, cx = split_by_mask(pe, mask, 1)
            context_recon = split_by_mask(panel_recon, mask, 1)[1]
            target_y.append(ty)
            context_y.append(cy)
            target_x.append(tx)
            context_x.append(cx)
            context_recons.append(context_recon)
        target_y = torch.cat(target_y, dim=0)
        context_y = torch.cat(context_y, dim=0)
        target_x = torch.cat(target_x, dim=0)
        context_x = torch.cat(context_x, dim=0)
        target_id = torch.cat(target_id, dim=0)
        context_recons = torch.cat(context_recons, dim=0)
        logits = self.reasoner.predict(context_x, context_y, target_x)
        logits = logits.argmax(dim=-1).reshape(-1, h, w)
        target_pred = self.backbone_dec(logits)
        panel_pred_vis = torch.cat([context_recons, target_pred.unsqueeze(1)], 1)
        results = {
            'panel': problem,
            'panel_recon': panel_pred_vis
        }
        return results
