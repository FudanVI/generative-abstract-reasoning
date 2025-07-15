import torch
import torch.nn as nn
from backbone_vqvae.vqvae import VQVAE
from lib.building_block import BaseNetwork


class MainNet(nn.Module, BaseNetwork):

    def __init__(self, args, device, global_step=0):
        nn.Module.__init__(self)
        BaseNetwork.__init__(self, args, global_step=global_step)
        self.device = device
        self.criterion = nn.MSELoss()
        self.latent_loss_weight = 0.25
        self.backbone = VQVAE(
            channel=args.channel, n_res_block=args.n_res_block,
            n_res_channel=args.n_res_channel, embed_dim=args.embed_dim,
            n_embed=args.n_embed)

    def forward(self, problem, answer, distractors):
        full_panel = torch.cat([problem, answer, distractors], dim=1)
        full_panel = full_panel.flatten(0, 1)
        out, latent_loss = self.backbone(full_panel)
        recon_loss = self.criterion(out, full_panel)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        logger = {
            'loss': loss.detach()
        }
        return loss, logger

    def metric(self, problem, answer, distractors):
        full_panel = torch.cat([problem, answer, distractors], dim=1)
        b, n, c, h, w = full_panel.size()
        full_panel = full_panel.flatten(0, 1)
        out, _ = self.backbone(full_panel)
        mse = (full_panel - out).pow(2).sum() / b
        logger_metric = {
            'MSE': mse.detach()
        }
        return logger_metric

    def test(self, problem, answer, distractors):
        full_panel = torch.cat([problem, answer, distractors], dim=1)
        full_panel = full_panel.flatten(0, 1)
        out, _ = self.backbone(full_panel)
        results = {
            'panel': full_panel,
            'panel_recon': out,
        }
        return results
