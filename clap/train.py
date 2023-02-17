import importlib
import sys
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.config import get_parser, get_config
import lib.utils as utils
from dataset.get_dataset import get_dataset
from dataset.graph_utils import GraphPlotter
from dataset.multiprocess import QueueDataLoader


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


def main():

    # parse config
    args = get_parser(sys.argv)
    args = get_config(args)
    config_module = importlib.import_module('{}.config'.format(args.model))
    train_module = importlib.import_module('{}.trainer'.format(args.model))
    args = config_module.get_parser(sys.argv, args)
    args = config_module.get_config(args)
    if args.seed > 0:
        utils.set_seed(args.seed)

    # prepare path
    print("[*] Prepare paths for experiments ...", end='')
    utils.clear_dir(args.tmp_path)
    utils.clear_dir(args.save_path)
    tb_path = '{}/{}/{}/{}'.format(args.run_path, args.dataset, args.image_type, args.exp_name)
    utils.clear_dir(tb_path, create=False)
    print(" Done")

    print("[*] Copy codes for experiments ...", end='\r')
    shutil.copytree('lib', os.path.join(args.save_path, 'lib'))
    shutil.copytree(args.model, os.path.join(args.save_path, args.model))
    print("[*] Copy codes for experiments ... Done")

    # data loader
    print("[*] Loading dataset ...", end='\r')
    train_set, valid_set, size_train, size_valid = get_dataset(
        args.image_type, cache_root=args.cache_path, dataset_name=args.dataset, size=args.image_size
    )
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_loader = QueueDataLoader(train_loader, utils.gpu_id_to_device(args.gpu))
    valid_loader = QueueDataLoader(valid_loader, utils.gpu_id_to_device(args.gpu))
    print("[*] Loading dataset ... Done")

    # setup the VAE and optimizer
    print("[*] Initializing optimizer ...", end='\r')
    vae, optimizer = train_module.prepare(args)
    print("[*] Initializing optimizer ... Done")

    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    # initialize tensorboard
    writer = SummaryWriter(log_dir=tb_path)
    # record best elbo
    best_elbo = -1e6
    best_epoch = -1
    plotter = GraphPlotter(args.dataset_type)
    try:
        for epoch in range(args.num_epochs):
            train_module.train_epoch(vae, optimizer, train_loader, args)
            train_module.eval_epoch(vae, optimizer, elbo_running_mean, valid_loader, args)
            avg_elbo = elbo_running_mean.get_avg()['elbo']
            train_module.saver(vae, optimizer, args, epoch, avg_elbo > best_elbo)
            best_epoch = epoch if avg_elbo > best_elbo else best_epoch
            best_elbo = avg_elbo if avg_elbo > best_elbo else best_elbo
            for k, v in elbo_running_mean.get_avg().items():
                writer.add_scalar('loss/%s' % k, v, epoch)
            if epoch % args.save_freq == 0:
                train_module.visualize(vae, writer, plotter, epoch, valid_loader, args)
            if epoch % args.log_freq == 0:
                elbo_running_mean.log(epoch, args.num_epochs, best_epoch)
            vae.update_step()
            writer.flush()
    except KeyboardInterrupt:
        writer.flush()
        writer.close()
        val = input('If delete the training data? (type \'yes\' or \'y\' to confirm): ')
        if val in ['y', 'yes']:
            utils.clear_dir(tb_path, create=False)
            utils.clear_dir(args.tmp_path, create=False)
            utils.clear_dir(args.save_path, create=False)
            print('[*] Experiment path clear ... Done')
        exit(-1)


if __name__ == '__main__':
    main()
