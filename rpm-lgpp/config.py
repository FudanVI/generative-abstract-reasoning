import os
import yaml
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/main.yaml', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--beta', type=int)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    assert args.exp_name is not None
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if val is not None:
            config[key] = val
    config['save_path'] = os.path.join(config['save_path'], args.exp_name)
    config['tmp_path'] = os.path.join(config['tmp_path'], args.exp_name)
    args.__dict__ = config
    return args
