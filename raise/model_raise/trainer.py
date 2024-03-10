import os.path as path
import time
import itertools
import math
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import lib.utils as utils
from dataset.get_dataset import get_dataset
from dataset.multiprocess import QueueDataLoader
from torch.utils.data import DataLoader
from model_raise.main import MainNet
from dataset.graph_utils import GraphPlotter
from lib.writer import AsyncWriter


def get_saved_params(context):
    return {
        'state_dict': context['model'].state_dict(),
        'optimizer': context['optimizer'].state_dict(),
        'args': context['args'], 'epoch': context['epoch']
    }


def prepare(context):
    args = context['args']
    device = torch.device("cuda:%d" % args.gpu) if args.gpu >= 0 else torch.device("cpu")

    # data loader
    print("[*] Loading dataset ...", end='\r')
    valid_set, size_valid = get_dataset(
        args.image_type, cache_root=args.cache_path,
        dataset_name=args.dataset, part='val', kargs={}
    )
    context['valid_loader'] = QueueDataLoader(
        DataLoader(dataset=valid_set, batch_size=args.batch_size,
                   shuffle=False, num_workers=0),
        utils.gpu_id_to_device(args.gpu)
    )
    if args.label_ratio < 1.0:
        train_set, size_train = get_dataset(
            args.image_type, cache_root=args.cache_path,
            dataset_name=args.dataset, part='train', kargs={'label': None}
        )
        context['train_loader'] = QueueDataLoader(
            DataLoader(dataset=train_set, batch_size=args.batch_size,
                       shuffle=True, num_workers=0),
            utils.gpu_id_to_device(args.gpu)
        )
        train_set_annotation, _ = get_dataset(
            args.image_type, cache_root=args.cache_path,
            dataset_name=args.dataset, part='train',
            kargs={'label': args.label_ratio}
        )
        context['train_loader_annotation'] = QueueDataLoader(
            DataLoader(dataset=train_set_annotation, batch_size=args.batch_size,
                       shuffle=True, num_workers=0),
            utils.gpu_id_to_device(args.gpu)
        )
    else:
        train_set_annotation, _ = get_dataset(
            args.image_type, cache_root=args.cache_path,
            dataset_name=args.dataset, part='train',
            kargs={'label': args.label_ratio}
        )
        context['train_loader_annotation'] = QueueDataLoader(
            DataLoader(dataset=train_set_annotation, batch_size=args.batch_size,
                       shuffle=True, num_workers=0),
            utils.gpu_id_to_device(args.gpu)
        )
    print("[*] Loading dataset ... Done")

    # setup the VAE and optimizer
    print("[*] Initializing model and optimizer ...", end='\r')
    vae = MainNet(args, device)
    vae = vae.to(device)
    params = [vae.parameters()]
    optimizer = optim.RMSprop(itertools.chain(*params), lr=args.learning_rate)
    print("[*] Initializing model and optimizer ... Done")

    context['model'] = vae
    context['optimizer'] = optimizer
    context['logger'] = utils.RunningAverageMeter()
    context['writer'] = SummaryWriter(log_dir=args.run_path)
    context['executor'] = AsyncWriter()
    context['best_metric'] = -1e6
    context['best_epoch'] = -1
    context['best_dict'] = {}
    return context


def epoch_start(context):
    context['start_time'] = time.time()


def train_epoch(context):
    model, optimizer = context['model'], context['optimizer']
    args = context['args']
    model.train()
    if args.label_ratio < 1.0:
        for i, (sample, panel, selection, answer) in context['train_loader'].visit():
            for j in range(4):
                sample_a, _, __, ___, label_a = \
                    context['train_loader_annotation'].visit_one()
                optimizer.zero_grad()
                obj, elbo, recon = model(sample_a, label=label_a)
                try:
                    obj.mul(-1).backward()
                    optimizer.step()
                except RuntimeError as e:
                    print(e)
                context['train_loader_annotation'].finish_task()
            optimizer.zero_grad()
            obj, elbo, recon = model(sample)
            try:
                obj.mul(-1).backward()
                optimizer.step()
            except RuntimeError as e:
                print(e)
            context['train_loader'].finish_task()
    else:
        for i, (sample, panel, selection, answer, label) \
                in context['train_loader_annotation'].visit():
            optimizer.zero_grad()
            obj, elbo, recon = model(sample, label=label)
            try:
                obj.mul(-1).backward()
                optimizer.step()
            except RuntimeError as e:
                print(e)
            context['train_loader_annotation'].finish_task()


def eval_epoch(context):
    model, logger = context['model'], context['logger']
    model.eval()
    logger.reset()
    with torch.no_grad():
        for i, (sample, panel, selection, answer, label) in context['valid_loader'].visit():
            _, loss_logger, __ = model(sample, label=label)
            metric_logger = model.metric(sample, selection, answer, label)
            logger.update(loss_logger, metric_logger)
            logger.next()
            context['valid_loader'].finish_task()
    losses, metrics = context['logger'].get_avg()
    if metrics['Concept ACC'] > context['best_metric']:
        context['best_dict'] = get_saved_params(context)
        context['best_epoch'] = context['epoch']
        context['best_metric'] = metrics['Concept ACC']
    if context['epoch'] % context['args'].save_freq_base == 0:
        saver(context)
    for k, v in losses.items():
        context['writer'].add_scalar('loss/%s' % k, v, context['epoch'])
    for k, v in metrics.items():
        context['writer'].add_scalar('metric/%s' % k, v, context['epoch'])


def visualize(context):
    model, writer, data_loader, epoch, args = \
        context['model'], context['writer'], \
        context['valid_loader'], context['epoch'], context['args']
    if context['epoch'] % args.save_freq_base != 0:
        return
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

        z_disentangle = result_enc['z'].reshape(-1, args.num_concept, args.concept_size)
        z_disentangle = utils.shuffle(z_disentangle)
        disentanglement_list = []
        upper, lower = z_disentangle.max(0)[0], z_disentangle.min(0)[0]
        for dim in range(args.num_concept):
            params = {
                'start': lower, 'end': upper,
                'num_split': 12, 'param_size': args.concept_size,
                'num_concept': args.num_concept
            }
            recon, num = model.baseNet.check_disentangle(z_disentangle[:num_sample], dim, params=params)
            disentanglement_list.append(recon)
        disentanglement_result = torch.cat(disentanglement_list, 1).detach().cpu()

        # register task to plot disentanglement results
        context['executor'].add_writer_task(
            write_disentangle, disentanglement_result, writer, epoch,
            num=num
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


def write_disentangle(disentanglement_result, writer, epoch, num=12):
    for sample_ind in range(len(disentanglement_result)):
        writer.add_image(
            'disentanglement/continuous/%d' % sample_ind,
            make_grid(disentanglement_result[sample_ind], nrow=num), epoch
        )


def saver(context):
    context['executor'].add_writer_task(
        utils.save_checkpoint, context['best_dict'], context['args'].save_path,
        name='best.pth'
    )
