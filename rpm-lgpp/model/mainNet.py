import torch
import torch.nn as nn
from model.axisVAE import AxisVAE
from lib.GP import DGP


class ReasonNet(nn.Module):
    def __init__(self, args, device):
        super(ReasonNet, self).__init__()
        self.image_size = args.image_size
        self.z_dim = args.latent_dim
        self.axis_dim = args.axis_dim
        self.device = device
        self.baseNet = AxisVAE(args, device)
        self.gp = DGP(args, device)

    def forward(self, x, t, dataset_size):
        # concat input
        x_view_panel = x.view(x.size(0) // 8, 8, 1, self.image_size, self.image_size)
        t_view_answer = t.view(t.size(0), 1, 1, self.image_size, self.image_size)
        panels = torch.cat((x_view_panel, t_view_answer), 1).view(-1, 1, self.image_size, self.image_size)
        batch_size = panels.size(0)
        ground_truth = panels.view(-1, self.image_size, self.image_size)
        results_enc = self.baseNet.encode(panels, x_view_panel.view(-1, 1, self.image_size, self.image_size))

        # gp regression
        zs_grid = results_enc['zs'].view(batch_size // 9, 9, self.z_dim)
        results_gp = self.gp(zs_grid, results_enc['axis_x'], results_enc['axis_y'], mode='train', use_diag=True)
        zs_grid = torch.cat((zs_grid[:, :-1, :], results_gp['z_cons'].unsqueeze(1)), 1)
        results_enc['zs'] = zs_grid.view(-1, self.z_dim)
        z_params_panel = results_enc['z_params'].view(batch_size // 9, 9, self.z_dim, 2)
        z33_params = torch.cat((results_gp['mu'][..., None], torch.log(results_gp['sigma'])[..., None]), -1)
        z_params_panel = torch.cat((z_params_panel[:, :-1, :, :], z33_params.unsqueeze(1)), 1)
        results_enc['z_params'] = z_params_panel.view(-1, self.z_dim, 2)
        x_recon, x_params = self.baseNet.decode(results_enc['zs'])

        # vae
        loss_dicts = {
            **results_enc, **results_gp,
            **{'gt': ground_truth, 'x_params': x_params, 'dataset_size': dataset_size}
        }
        elbo_vae, logger_vae = self.baseNet.loss(loss_dicts)

        elbo = elbo_vae
        return elbo, x_params[:, :, :, 0].unsqueeze(1)

    def evaluate(self, x, t, dataset_size):
        # concat input
        x_view_panel = x.view(x.size(0) // 8, 8, 1, self.image_size, self.image_size)
        t_view_answer = t.view(t.size(0), 1, 1, self.image_size, self.image_size)
        panels = torch.cat((x_view_panel, t_view_answer), 1).view(-1, 1, self.image_size, self.image_size)
        batch_size = panels.size(0)
        ground_truth = panels.view(-1, self.image_size, self.image_size)
        results_enc = self.baseNet.encode(panels, x_view_panel.view(-1, 1, self.image_size, self.image_size))

        # gp regression
        zs_grid = results_enc['zs'].view(batch_size // 9, 9, self.z_dim)
        results_gp = self.gp(zs_grid, results_enc['axis_x'], results_enc['axis_y'], mode='eval', use_diag=True)
        zs_grid[:, -1, :] = results_gp['z_cons']
        results_enc['zs'] = zs_grid.view(-1, self.z_dim)
        z_params_panel = results_enc['z_params'].view(batch_size // 9, 9, self.z_dim, 2)
        z_params_panel[:, -1, :, 0] = results_gp['mu']
        z_params_panel[:, -1, :, 1] = torch.log(results_gp['sigma'])
        results_enc['z_params'] = z_params_panel.view(-1, self.z_dim, 2)
        x_recon, x_params = self.baseNet.decode(results_enc['zs'])

        loss_dicts = {
            **results_enc, **results_gp,
            **{'gt': ground_truth, 'x_params': x_params, 'dataset_size': dataset_size}
        }
        elbo_vae, logger_vae = self.baseNet.loss(loss_dicts, val=True)
        elbo = elbo_vae
        logger_other = {'elbo': elbo.detach()}

        return elbo, {**logger_vae, **logger_other}, x_params[:, :, :, 0].unsqueeze(1)

    def test(self, x, t):
        # concat input
        x_view_panel = x.view(x.size(0) // 8, 8, 1, self.image_size,  self.image_size)
        t_view_answer = t.view(t.size(0), 1, 1,  self.image_size,  self.image_size)
        panels = torch.cat((x_view_panel, t_view_answer), 1).view(-1, 1,  self.image_size,  self.image_size)
        batch_size = panels.size(0)
        results_enc = self.baseNet.encode(panels, x_view_panel.view(-1, 1, self.image_size, self.image_size))
        x_recon, x_params = self.baseNet.decode(results_enc['zs'])
        
        # gp regression
        zs_grid = results_enc['zs'].view(batch_size // 9, 9, self.z_dim)
        results_gp = self.gp(zs_grid.detach(), results_enc['axis_x'], results_enc['axis_y'], mode='test')
        x_recon_answer, _ = self.baseNet.decode(results_gp['z_cons'])

        return x_params[:, :, :, 0].unsqueeze(1), _[:, :, :, 0].unsqueeze(1)

    def predict(self, x, t, mask):
        # concat input
        x_view_panel = x.view(x.size(0) // 8, 8, 1, self.image_size, self.image_size)
        t_view_answer = t.view(t.size(0), 1, 1, self.image_size, self.image_size)
        panels = torch.cat((x_view_panel, t_view_answer), 1).view(-1, 1, self.image_size, self.image_size)
        x_view_panel = x_view_panel * mask.view(-1)[:-1].contiguous().view(1, -1, 1, 1, 1)
        batch_size = panels.size(0)
        results_enc = self.baseNet.encode(panels, x_view_panel.view(-1, 1, self.image_size, self.image_size))
        
        # gp regression
        zs_grid = results_enc['zs'].view(batch_size // 9, 9, self.z_dim)
        results_gp = self.gp.predict(zs_grid.detach(), mask, results_enc['axis_x'], results_enc['axis_y'])
        x_recon, _ = self.baseNet.decode(results_gp['z_full'].transpose(-2, -1).contiguous().view(-1, self.z_dim))

        return _[:, :, :, 0].unsqueeze(1)

    def generate(self):
        axis_x = torch.randn(1, self.z_dim, 3 * self.axis_dim).to(self.device)
        axis_y = torch.randn(1, self.z_dim, 3 * self.axis_dim).to(self.device)
        z_gen, k = self.gp.generate(axis_x, axis_y)
        x_recon, _ = self.baseNet.decode(z_gen.view(-1, self.z_dim))
        return x_recon, z_gen, k, _

    def interpolate(self, nx=3, ny=3):
        axis_x = torch.randn(1, self.z_dim, 2, self.axis_dim).to(self.device)
        axis_y = torch.randn(1, self.z_dim, 2, self.axis_dim).to(self.device)
        axis_x_interpolate = torch.zeros(1, self.z_dim, nx, self.axis_dim).to(self.device)
        for i in range(nx):
            axis_x_interpolate[0, :, i, :] = axis_x[0, :, 0, :] + i * (axis_x[0, :, 1, :] - axis_x[0, :, 0, :]) / (nx - 1)
        axis_y_interpolate = torch.zeros(1, self.z_dim, ny, self.axis_dim).to(self.device)
        for i in range(ny):
            axis_y_interpolate[0, :, i, :] = axis_y[0, :, 0, :] + i * (axis_y[0, :, 1, :] - axis_y[0, :, 0, :]) / (ny - 1)
        z_gen = self.gp.interpolate(axis_x_interpolate, axis_y_interpolate)
        x_recon, _ = self.baseNet.decode(z_gen.view(-1, self.z_dim))
        return _[:, :, :, 0].unsqueeze(1)

    def check_distribution(self, num_sample=128):
        axis_x = torch.randn(num_sample, self.z_dim, 3 * self.axis_dim).to(self.device)
        axis_y = torch.randn(num_sample, self.z_dim, 3 * self.axis_dim).to(self.device)
        z_gen, k = self.gp.generate(axis_x, axis_y)
        return z_gen
