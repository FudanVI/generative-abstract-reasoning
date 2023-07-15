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
    context = {'args': args}
    utils.set_seed(seed=0)

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
        args.image_type, cache_root=args.cache_path, dataset_name=args.dataset
    )
    context['train_loader'] = QueueDataLoader(
        DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0),
        utils.gpu_id_to_device(args.gpu)
    )
    context['valid_loader'] = QueueDataLoader(
        DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0),
        utils.gpu_id_to_device(args.gpu)
    )
    print("[*] Loading dataset ... Done")

    # setup the VAE and optimizer
    print("[*] Initializing model and optimizer ...", end='\r')
    context = train_module.prepare(context)
    print("[*] Initializing model and optimizer ... Done")

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
