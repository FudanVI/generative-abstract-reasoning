import importlib
import sys
import os
import shutil
import torch
from torch.utils.data import DataLoader
from config.config import get_parser, get_config
import lib.utils as utils
from dataset.get_dataset import get_dataset
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
    utils.set_seed(args.seed)
    context = {'args': args}

    # prepare path
    print("[*] Prepare paths for experiments ...", end='')
    utils.clear_dir(args.tmp_path)
    utils.clear_dir(args.save_path)
    tb_path = args.run_path
    utils.clear_dir(tb_path, create=False)
    print(" Done")

    print("[*] Copy codes for experiments ...", end='\r')
    shutil.copytree('lib', os.path.join(args.save_path, 'lib'))
    shutil.copytree(args.model, os.path.join(args.save_path, args.model))
    print("[*] Copy codes for experiments ... Done")
    
    context = train_module.prepare(context)

    try:
        for epoch in range(args.num_epochs):
            context['epoch'] = epoch
            train_module.epoch_start(context)
            train_module.train_epoch(context)
            train_module.eval_epoch(context)
            train_module.visualize(context)
            train_module.epoch_end(context)
    except KeyboardInterrupt:
        train_module.train_abort(context)
        val = input('If delete the training data? (type \'yes\' or \'y\' to confirm): ')
        if val in ['y', 'yes']:
            utils.clear_dir(tb_path, create=False)
            utils.clear_dir(args.tmp_path, create=False)
            utils.clear_dir(args.save_path, create=False)
            print('[*] Experiment path clear ... Done')
        exit(-1)


if __name__ == '__main__':
    main()
