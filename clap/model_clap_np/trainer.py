import torch
import torch.optim as optim
from torchvision.utils import make_grid
import lib.utils as utils
from model_clap_np.main import MainNet


def prepare(args):
    device = torch.device("cuda:%d" % args.gpu) if args.gpu >= 0 else torch.device("cpu")
    vae = MainNet(args, device)
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    return vae, optimizer


def train_epoch(model, optimizer, data_loader, args):
    model.train()
    for i, (panels, labels, classes) in data_loader.visit():
        optimizer.zero_grad()
        obj, elbo, recon = model(panels, labels)
        obj.mul(-1).backward()
        optimizer.step()
        data_loader.finish_task()


def eval_epoch(model, optimizer, logger, data_loader, args):
    model.eval()
    logger.reset()
    with torch.no_grad():
        for i, (panels, labels, classes) in data_loader.visit():
            loss_logs = model.evaluate(panels, labels, num_pred=args.num_missing[1])
            met_logs = model.metric(panels, labels, num_pred=args.num_missing[1])
            logs = {**loss_logs, **met_logs}
            logger.update(logs)
            logger.next()
            data_loader.finish_task()


def visualize(model, writer, graph_plotter, epoch, data_loader, args):
    device = torch.device("cuda:%d" % args.gpu) if args.gpu >= 0 else torch.device("cpu")
    with torch.no_grad():
        panels, labels, classes = data_loader.visit_one()
        bs = panels.size(0)
        num_sample = min(args.log_image_num, bs)
        recon, recon_pred, result_enc = model.test(panels, labels, pred_num=args.num_missing[1])
        for sample_ind in range(num_sample):
            stacks = graph_plotter.plot_prediction(
                recon_pred[sample_ind].detach().cpu().numpy(),
                panels[sample_ind].cpu().numpy(),
                args.dataset_params, result_enc['mask']
            )
            writer.add_image('sample/%d' % sample_ind, stacks, epoch)
        z_disentangle = result_enc['z'].reshape(-1, args.num_concept, args.concept_size)
        z_disentangle = utils.shuffle(z_disentangle)
        disentanglement_list = []
        upper, lower = z_disentangle.max(0)[0], z_disentangle.min(0)[0]
        for dim in range(args.num_concept):
            params = {'lower': lower, 'upper': upper, 'num': 12}
            recon, num = model.baseNet.check_disentangle_contiguous(z_disentangle[:num_sample], dim, params=params)
            disentanglement_list.append(recon)
        disentanglement_result = torch.cat(disentanglement_list, 1)
        for sample_ind in range(num_sample):
            writer.add_image(
                'disentanglement/continuous/%d' % sample_ind,
                make_grid(disentanglement_result[sample_ind], nrow=num), epoch
            )


def saver(model, optimizer, args, epoch, best):
    if best:
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args, 'epoch': epoch
        }, args.save_path)
    utils.save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args, 'epoch': epoch
    }, args.save_path, name='latest.pth')
