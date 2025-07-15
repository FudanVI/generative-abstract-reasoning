import importlib
import sys
from config.config import get_parser, get_config
import lib.utils as utils


def main():

    # parse config
    args = get_parser(sys.argv)
    args = get_config(args)
    config_module = importlib.import_module('{}.config'.format(args.model))
    train_module = importlib.import_module('{}.trainer'.format(args.model))
    args = config_module.get_parser(sys.argv, args)
    args = config_module.get_config(args)
    context = {'args': args}

    # check experiment paths
    print("[*] Prepare paths for experiments ...", end='')
    utils.clear_dir(args.save_path)
    tb_path = '{}/{}'.format(args.run_path, args.exp_name)
    utils.clear_dir(tb_path, create=False)
    print(" Done")

    # setup the data loader, model, optimizer and environment
    context = train_module.prepare(context)
    try:
        for epoch in range(args.num_epochs):
            train_module.epoch_start(context)
            train_module.train_epoch(context)
            train_module.eval_epoch(context)
            train_module.visualize(context)
            train_module.epoch_end(context)
            context['epoch'] = context['epoch'] + 1
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
