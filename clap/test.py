import importlib.util
import os
import random
import pickle
import argparse
import yaml
import copy
import torch
from torch.utils.data import DataLoader
from dataset.get_dataset import get_dataset


CFG = {
    'balls': {
        'num_pred': [4, 6]
    },
    'MPI3D': {
        'num_pred': [4, 6, 10, 20]
    },
    'CRPM': {
        'num_pred': [2, 3]
    }
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/main.yaml')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--image_type', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_epoch', type=int, default=10)
    args = parser.parse_args()
    assert args.exp_name is not None
    assert args.model is not None
    assert args.dataset is not None
    assert args.image_type is not None
    return args


def get_config(pre_args):
    args = copy.deepcopy(pre_args)
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
    args.__dict__ = config
    return args


def mean_var(a):
    var, mean = torch.var_mean(a)
    return mean.item(), var.sqrt().item()


def mse(args, model, loader, save_path, utils, num_loop=1, num_pred=1):
    device = torch.device('cuda:%d' % args.gpu) if args.gpu >= 0 else torch.device('cpu')
    mse_list = []
    for rounds in range(num_loop):
        with torch.no_grad():
            mse_inner, count = 0, 0
            for i, (panels, labels, classes) in enumerate(loader):
                if args.dataset == 'MPI3D' and num_pred < 10:
                    panels, labels, classes = loader.dataset.process_input(panels, labels, classes)
                panels = panels.to(device)
                labels = labels.to(device)
                recon, recon_pred, result_enc = model.test(panels, labels, pred_num=num_pred)
                mask = result_enc['mask']
                count += panels.size(0)
                targets_gt, _ = utils.split_by_mask(panels, mask, 1)
                targets_pred, _ = utils.split_by_mask(recon_pred, mask, 1)
                mse_inner += (targets_gt - targets_pred).pow(2).reshape(panels.size(0), -1).sum(-1).sum().item()
            mse_inner /= count
            mse_list.append(mse_inner)
        print('Test epoch [%d/%d]' % (rounds, num_loop), end='\r')
    mean, std = mean_var(torch.tensor(mse_list).float())
    print('MSE-%d: mean %.4f \t standard deviation %.4f' % (num_pred, mean, std))
    return mean, std


def select_acc(args, model, loader, save_path, utils, num_loop=1, num_pred=1, num_fusion=7):
    device = torch.device('cuda:%d' % args.gpu) if args.gpu >= 0 else torch.device('cpu')
    mse_list = []
    for rounds in range(num_loop):
        with torch.no_grad():
            sa_inner, count = 0, 0
            for i, (panels, labels, classes) in enumerate(loader):
                if args.dataset == 'MPI3D' and num_pred < 10:
                    panels, labels, classes = loader.dataset.process_input(panels, labels, classes)
                panels = panels.to(device)
                labels = labels.to(device)
                bs = panels.size(0)
                recon, recon_pred, result_enc = model.test(panels, labels, pred_num=num_pred)
                mask = result_enc['mask']
                count += panels.size(0)
                targets_gt, _ = utils.split_by_mask(panels, mask, 1)
                targets_pred, _ = utils.split_by_mask(recon_pred, mask, 1)
                labels_gt, _ = utils.split_by_mask(labels, mask, 1)
                r_gt = model.encode(targets_gt, labels_gt)
                r_pred = model.encode(targets_pred, labels_gt)
                candidates = torch.zeros(bs, num_fusion + 1, num_pred, r_gt.size(-1)).to(device)
                for j in range(bs):
                    indexes = list(range(bs))
                    indexes = indexes[:j] + indexes[j+1:]
                    random.shuffle(indexes)
                    fusion_size = min(num_fusion, bs - 1)
                    indexes = indexes[:fusion_size]
                    if fusion_size < num_fusion:
                        indexes = indexes + random.choices(indexes, k=num_fusion - fusion_size)
                    indexes.append(j)
                    indexes = torch.tensor(indexes).long().to(device)
                    candidates[j] = r_gt[indexes]
                scores = (candidates - r_pred.unsqueeze(1)).pow(2).reshape(bs, num_fusion + 1, -1).sum(-1)
                select_index = scores.argmin(-1)
                sa_inner += (select_index == num_fusion).int().sum().item()
            sa_inner /= count
            mse_list.append(sa_inner)
        print('Test epoch [%d/%d]' % (rounds, num_loop), end='\r')
    mean, std = mean_var(torch.tensor(mse_list).float())
    print('SA-%d-%d: mean %.4f \t standard deviation %.4f' % (num_pred, num_fusion, mean, std))
    return mean, std


def main():
    # parse config
    pre_args = get_config(get_parser())
    print('Experiment on {}-{}-{}'.format(pre_args.dataset, pre_args.image_type, pre_args.exp_name))
    device = torch.device('cuda:%d' % pre_args.gpu) if pre_args.gpu >= 0 else torch.device('cpu')
    save_file = torch.load(os.path.join(pre_args.save_path, 'checkpoint.pth'), map_location=device)
    args = save_file['args']
    for key in ['gpu', 'batch_size', 'test_epoch']:
        if getattr(pre_args, key) is not None:
            setattr(args, key, getattr(pre_args, key))

    spec = importlib.util.spec_from_file_location('trainer', os.path.join(args.model, 'trainer.py'))
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    spec = importlib.util.spec_from_file_location('utils', os.path.join('lib', 'utils.py'))
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    spec = importlib.util.spec_from_file_location('main', os.path.join(args.model, 'main.py'))
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    # data loader
    print("[*] Loading dataset ...", end='\r')
    test_set, test_size = get_dataset(
        args.image_type, cache_root=args.cache_path, dataset_name=args.dataset, size=args.image_size, train=False
    )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("[*] Loading dataset ... Done")

    # setup the VAE and optimizer
    print("[*] Loading model ...", end='\r')
    start_epoch = save_file['epoch']
    vae = main.MainNet(args, device, global_step=start_epoch)
    vae = vae.to(device)
    vae.load_state_dict(save_file['state_dict'])
    vae = vae.eval()
    print("[*] Loading model ... Done. Start epoch {}".format(start_epoch))

    results = {}
    for num_pred in CFG[args.dataset]['num_pred']:
        results['mse-{}'.format(num_pred)] = mse(
            args, vae, test_loader, args.save_path, utils,
            num_loop=args.test_epoch, num_pred=num_pred
        )
        for k in [1, 2, 4, 8, 16]:
            results['sa-{}-{}'.format(num_pred, k)] = select_acc(
                args, vae, test_loader, args.save_path, utils,
                num_loop=args.test_epoch, num_pred=num_pred, num_fusion=k
            )
    with open(os.path.join(args.save_path, 'test_results'), 'wb') as output_f:
        pickle.dump(results, output_f)


if __name__ == '__main__':
    main()
