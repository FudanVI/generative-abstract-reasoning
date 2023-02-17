import os
import yaml
from config.parser import ArgsParser


def get_parser(args_list):
    parser = ArgsParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--config_path', default='config/main.yaml', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--image_type', type=str)
    args = parser.get_args(args_list)
    assert args.model is not None
    assert args.exp_name is not None
    assert args.dataset is not None
    assert args.image_type is not None
    return args


def get_config(args):
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if val is not None:
            config[key] = val
    config['dataset_params'] = config[config['dataset']][config['image_type']]
    for k, v in config['dataset_params'].items():
        config[k] = v
    config['save_path'] = os.path.join(config['save_path'], config['dataset'], config['image_type'], args.exp_name)
    config['tmp_path'] = os.path.join(config['tmp_path'], config['dataset'], config['image_type'], args.exp_name)
    for k, v in config.items():
        setattr(args, k, v)
    return args
