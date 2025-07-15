import os
import yaml
from config.parser import ArgsParser


def get_parser(args_list):
    parser = ArgsParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--config_path', default='config/main.yaml', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--size', type=int)
    parser.add_argument('--dataset', type=str)
    args = parser.get_args(args_list)
    assert args.model is not None
    assert args.exp_name is not None
    return args


def get_config(args):
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if val is not None:
            config[key] = val
    config['save_path'] = os.path.join(config['save_path'], args.exp_name)
    config['tmp_path'] = os.path.join(config['tmp_path'], args.exp_name)
    for k, v in config.items():
        setattr(args, k, v)
    return args
