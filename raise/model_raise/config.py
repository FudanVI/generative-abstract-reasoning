import yaml
from config.parser import ArgsParser


def get_parser(args_list, pre_args):
    parser = ArgsParser()
    parser.add_argument(
        '--config_path',
        default='model_raise/config_files/{}/{}.yaml'.format(pre_args.dataset, pre_args.image_type),
        type=str
    )
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    args = parser.get_args(args_list, pre_args)
    return args


def get_config(args):
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if val is not None:
            config[key] = val
    for k, v in config.items():
        setattr(args, k, v)
    return args
