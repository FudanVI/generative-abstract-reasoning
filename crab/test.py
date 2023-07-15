import importlib.util
import os
import pickle
import argparse
import random
import yaml
import copy
import torch
from torch.utils.data import DataLoader
from dataset.get_dataset import get_dataset
import lib.utils as utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/main.yaml')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--image_type', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=512)
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
    config['save_path'] = os.path.join(config['save_path'], 'RAVEN', config['image_type'], args.exp_name)
    config['tmp_path'] = os.path.join(config['tmp_path'], 'RAVEN', config['image_type'], args.exp_name)
    args.__dict__ = config
    return args


def mean_var(a):
    var, mean = torch.var_mean(a)
    return mean.item(), var.sqrt().item()


def acc(args, model, loader, save_path, num_loop=1):
    device = torch.device('cuda:%d' % args.gpu) if args.gpu >= 0 else torch.device('cpu')
    acc_list = []
    for rounds in range(num_loop):
        with torch.no_grad():
            acc_inner, count = 0, 0
            for i, (samples, panels, selections, answers, classes) in enumerate(loader):
                samples = samples.to(device)
                panels = panels.to(device)
                selections = selections.to(device)
                answers = answers.to(device)
                classes = classes.to(device)
                mask = torch.tensor([0] * 8 + [1]).to(device)
                _, __, results = model.test(samples, mask=mask)
                count += panels.size(0)
                selections_z = model.baseNet.encode(selections)['z_post'][..., 0]
                pred_zt = results['zt_prior'][..., 0]
                selected_index = (pred_zt - selections_z).pow(2).sum([2, 3]).argmin(1)
                acc_inner += torch.eq(selected_index, answers).float().sum().item()
            acc_inner /= count
            acc_list.append(acc_inner)
        print('Test epoch [%d/%d]' % (rounds, num_loop), end='\r')
    mses = torch.tensor(acc_list).float()
    print('ACC: mean %.4f \t standard deviation %.4f' % (mean_var(mses)[0], mean_var(mses)[1]))
    with open(os.path.join(save_path, 'test_results_{}_acc.pkl'.format(args.dataset)), 'wb') as output_f:
        pickle.dump({'acc': mean_var(mses)}, output_f)


def g_acc(args, model, loader, save_path, num_preds, num_fusions, num_loop=1):
    results = {}
    for num_pred in num_preds:
        for k in num_fusions:
            mean, std = _g_acc(args, model, loader, num_pred, k, num_loop=num_loop)
            results['g_acc-{}-{}'.format(num_pred, k + 1)] = (mean, std)
    with open(os.path.join(save_path, 'test_results_{}_g_acc.pkl'.format(args.dataset)), 'wb') as output_f:
        pickle.dump(results, output_f)


def _g_acc(args, model, loader, num_pred, num_fusion, num_loop=1):
    device = torch.device('cuda:%d' % args.gpu) if args.gpu >= 0 else torch.device('cpu')
    acc_list = []
    for rounds in range(num_loop):
        with torch.no_grad():
            acc_inner, count = 0, 0
            for i, (samples, panels, selections, answers, classes) in enumerate(loader):
                samples = samples.to(device)
                panels = panels.to(device)
                selections = selections.to(device)
                answers = answers.to(device)
                classes = classes.to(device)
                bs = panels.size(0)
                _, __, results = model.test(samples, pred_num=num_pred)
                mask = results['mask']
                count += panels.size(0)
                targets_gt, _ = utils.split_by_mask(results['z_post'][..., 0], mask, 1)
                pred_zt = results['zt_prior'][..., 0].unsqueeze(1)
                candidates = torch.zeros(bs, num_fusion + 1, num_pred, *pred_zt.size()[-2:]).to(device)
                for j in range(bs):
                    indexes = list(range(bs))
                    indexes = indexes[:j] + indexes[j + 1:]
                    random.shuffle(indexes)
                    fusion_size = min(num_fusion, bs - 1)
                    indexes = indexes[:fusion_size]
                    if fusion_size < num_fusion:
                        indexes = indexes + random.choices(indexes, k=num_fusion - fusion_size)
                    indexes.append(j)
                    indexes = torch.tensor(indexes).long().to(device)
                    candidates[j] = targets_gt[indexes]
                selected_index = (pred_zt - candidates).pow(2).sum([2, 3, 4]).argmin(1)
                acc_inner += torch.eq(selected_index, num_fusion).float().sum().item()
            acc_inner /= count
            acc_list.append(acc_inner)
        print('Test epoch [%d/%d]' % (rounds, num_loop), end='\r')
    mses = torch.tensor(acc_list).float()
    print('G_ACC num_pred %d num_fusion %d: mean %.4f \t standard deviation %.4f' % (
        num_pred, num_fusion, mean_var(mses)[0], mean_var(mses)[1]
    ))
    return mean_var(mses)


def main():
    # parse config
    pre_args = get_config(get_parser())
    save_path = pre_args.save_path
    print('Experiment on {}-{}-{}'.format(pre_args.dataset, pre_args.image_type, pre_args.exp_name))
    device = torch.device('cuda:%d' % pre_args.gpu) if pre_args.gpu >= 0 else torch.device('cpu')
    save_file = torch.load(os.path.join(save_path, 'best.pth'), map_location=device)
    args = save_file['args']
    for key in ['gpu', 'batch_size', 'test_epoch', 'dataset', 'cache_path']:
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
        args.image_type, cache_root=args.cache_path, dataset_name=args.dataset, train=False
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

    acc(args, vae, test_loader, save_path, num_loop=args.test_epoch)
    g_acc(args, vae, test_loader, save_path, [1, 2, 3], [1, 3, 7, 15], num_loop=args.test_epoch)


if __name__ == '__main__':
    main()
