import os
import time
import tqdm
import torch
from torch.optim import Adam
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from model_ucgst.main import MainNet
from dataset.get_dataset import get_dataset
from lib.writer import AsyncWriter
from lib.utils import gpu_id_to_device
from lib.building_block import update_dict


def prepare(context):
    args = context['args']
    context['epoch'] = 0
    device = torch.device(gpu_id_to_device(args.gpu))

    # setup data loader
    print("[*] Loading dataset ...", end='\r')
    train_loader, valid_loader = get_dataset(args)
    context['train_loader'] = train_loader
    context['valid_loader'] = valid_loader
    print("[*] Loading dataset ... Done")

    # setup the model and optimizer
    print("[*] Initializing model and optimizer ...", end='\r')
    model = MainNet(args, device, global_step=0)
    model = model.to(device)
    optimizer = Adam(model.parameters(), args.lr)
    context['model'] = model
    context['optimizer'] = optimizer
    print("[*] Initializing model and optimizer ... Done")

    # setup training environment
    context['writer'] = SummaryWriter(
        log_dir='{}/{}'.format(args.run_path, args.exp_name)
    )
    context['executor'] = AsyncWriter()
    context['best_metric'] = 0
    context['best_epoch'] = -1
    context['best_dict'] = {}
    return context


def epoch_start(context):
    context['start_time'] = time.time()


def train_iteration(context, index, data):
    args, epoch = context['args'], context['epoch']
    optimizer, model, writer = context['optimizer'], context['model'], context['writer']
    device = gpu_id_to_device(args.gpu)
    problem = data[0].to(device)
    answer = data[1].to(device)
    distractors = data[2].to(device)
    optimizer.zero_grad()
    loss, loss_logs = model(problem, answer, distractors)
    loss.backward()
    optimizer.step()
    return loss_logs


def train_epoch(context):
    model, args = context['model'], context['args']
    model.train()
    train_log_loss = {}
    with tqdm.tqdm(total=args.train_iter) as progress_bar:
        for i in range(args.train_iter):
            data = next(context['train_loader'])
            log_loss = train_iteration(context, i, data)
            progress_bar.update(1)
            train_log_loss = update_dict(train_log_loss, log_loss)
        model.update_step()
    for key, val in train_log_loss.items():
        context['executor'].add_writer_task(
            context['writer'].add_scalar, 'train_loss/{}'.format(key),
            sum(val) / len(val), context['epoch'] + 1)


def eval_epoch(context):
    model, optimizer, writer = context['model'], context['optimizer'], context['writer']
    args, epoch = context['args'], context['epoch']
    device = gpu_id_to_device(args.gpu)
    model.eval()
    val_loss, val_log_loss, val_log_metric = [], {}, {}
    with torch.no_grad():
        with tqdm.tqdm(total=args.test_iter) as progress_bar:
            for i in range(args.test_iter):
                data = next(context['valid_loader'])
                problem = data[0].to(device)
                answer = data[1].to(device)
                distractors = data[2].to(device)
                loss, log_loss = model(problem, answer, distractors)
                log_metric = model.metric(problem, answer, distractors)
                val_loss += [loss.item()]
                val_log_loss = update_dict(val_log_loss, log_loss)
                val_log_metric = update_dict(val_log_metric, log_metric)
                progress_bar.update(1)
    val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_log_metric['ACC']) / len(val_log_metric['ACC'])
    context['executor'].add_writer_task(
        writer.add_scalar, 'eval_loss/metric', val_loss, epoch + 1)
    for key, val in val_log_loss.items():
        context['executor'].add_writer_task(
            writer.add_scalar, 'eval_loss/{}'.format(key), sum(val) / len(val), epoch + 1)
    for key, val in val_log_metric.items():
        context['executor'].add_writer_task(
            writer.add_scalar, 'eval_loss/{}'.format(key), sum(val) / len(val), epoch + 1)
    print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch + 1, val_loss))
    if val_acc > context['best_metric']:
        context['best_metric'] = val_acc
        context['best_epoch'] = epoch + 1
        checkpoint = {
            'epoch': epoch + 1,
            'best_metric': context['best_metric'],
            'best_epoch': context['best_epoch'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'checkpoint.pt.tar'))
    checkpoint = {
        'epoch': epoch + 1,
        'best_metric': context['best_metric'],
        'best_epoch': context['best_epoch'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args
    }
    torch.save(checkpoint, os.path.join(args.save_path, 'last.pt.tar'))
    context['executor'].add_writer_task(
        writer.add_scalar, 'eval_loss/best_metric', context['best_metric'], epoch + 1)
    print('====> [{}] Best Loss = {:F} @ Epoch {}'.format(
        args.exp_name, context['best_metric'], context['best_epoch']))


def visualize(context):
    model, writer = context['model'], context['writer']
    args, epoch = context['args'], context['epoch']
    device = gpu_id_to_device(args.gpu)
    model.eval()
    with torch.no_grad():
        data = next(context['valid_loader'])
        problem = data[0].to(device)
        answer = data[1].to(device)
        distractors = data[2].to(device)
        results = model.test(problem, answer, distractors)
        panel_recon = results['panel_recon'][:4]
        panel = results['panel'][:4]
        panel_recon_post = results['panel_recon_post'][:4]
        n = panel.size(1)
        vis_panel = torch.cat([panel, panel_recon, panel_recon_post], dim=1).flatten(0, 1)
        context['executor'].add_writer_task(
            writer.add_image, 'recon_panels', make_grid(vis_panel, nrow=n), epoch + 1)


def epoch_end(context):
    context['writer'].flush()
    context['epoch'] += 1


def train_abort(context):
    context['writer'].flush()
    context['writer'].close()
