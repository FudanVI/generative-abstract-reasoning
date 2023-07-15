import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import lib.utils as utils
from model_score.main import MainNet
from dataset.graph_utils import GraphPlotter
from lib.writer import AsyncWriter


def get_saved_params(context):
    return {
        'state_dict': context['model'].state_dict(),
        'optimizer': context['optimizer'],
        'args': context['args'], 'epoch': context['epoch']
    }


def prepare(context):
    args = context['args']
    device = torch.device("cuda:%d" % args.gpu) if args.gpu >= 0 else torch.device("cpu")
    vae = MainNet(args, device)
    vae = vae.to(device)
    optimizer = optim.RMSprop(vae.parameters(), lr=args.learning_rate)
    context['model'] = vae
    context['optimizer'] = optimizer
    context['logger'] = utils.RunningAverageMeter()
    context['writer'] = SummaryWriter(
        log_dir='{}/{}/{}/{}'.format(args.run_path, args.dataset, args.image_type, args.exp_name)
    )
    context['executor'] = AsyncWriter()
    context['best_metric'] = -1e6
    context['best_epoch'] = -1
    context['best_dict'] = {}
    return context


def epoch_start(context):
    context['start_time'] = time.time()


def train_epoch(context):
    model, optimizer = context['model'], context['optimizer']
    model.train()
    for i, (sample, panel, selection, answer, label) in context['train_loader'].visit():
        optimizer.zero_grad()
        obj, elbo, recon = model(sample)
        obj.mul(-1).backward()
        optimizer.step()
        context['train_loader'].finish_task()


def eval_epoch(context):
    model, logger = context['model'], context['logger']
    model.eval()
    logger.reset()
    with torch.no_grad():
        for i, (sample, panel, selection, answer, label) in context['valid_loader'].visit():
            _, loss_logger, __ = model(sample)
            metric_logger = model.metric(sample, selection, answer)
            logger.update(loss_logger, metric_logger)
            logger.next()
            context['valid_loader'].finish_task()
    losses, metrics = context['logger'].get_avg()
    if metrics['Concept ACC'] > context['best_metric']:
        context['best_dict'] = get_saved_params(context)
        context['best_epoch'] = context['epoch']
        context['best_metric'] = metrics['Concept ACC']
        saver(context)
    for k, v in losses.items():
        context['writer'].add_scalar('loss/%s' % k, v, context['epoch'])
    for k, v in metrics.items():
        context['writer'].add_scalar('metric/%s' % k, v, context['epoch'])


def visualize(context):
    model, writer, data_loader, epoch, args = \
        context['model'], context['writer'], \
        context['valid_loader'], context['epoch'], context['args']
    model.eval()
    with torch.no_grad():
        sample, panel, selection, answer, label = data_loader.visit_one()
        bs = sample.size(0)
        num_sample = min(args.log_image_num, bs)
        recon, recon_pred, result_enc = model.test(sample)

        # register task to plot samples
        context['executor'].add_writer_task(
            write_samples, recon_pred[:num_sample].detach().cpu().numpy(),
            sample[:num_sample].detach().cpu().numpy(),
            result_enc['mask'].detach().cpu(), args, writer, epoch
        )


def epoch_end(context):
    args, epoch = context['args'], context['epoch']
    if epoch % args.log_freq == 0:
        context['logger'].log(
            context['epoch'], args.num_epochs, context['best_epoch'],
            time.time() - context['start_time'], end='\t'
        )
        context['executor'].log()
    context['model'].update_step()
    context['writer'].flush()


def train_abort(context):
    context['writer'].flush()
    context['writer'].close()


def write_samples(pred, gt, mask, args, writer, epoch):
    graph_plotter = GraphPlotter(args.dataset_type)
    for sample_ind in range(len(pred)):
        stacks = graph_plotter.plot_prediction(
            pred[sample_ind], gt[sample_ind],
            args.dataset_params, mask
        )
        writer.add_image('sample/%d' % sample_ind, stacks, epoch)


def saver(context):
    context['executor'].add_writer_task(
        utils.save_checkpoint, context['best_dict'], context['args'].save_path,
        name='best.pth'
    )
