import os
import time
import pickle
import argparse
import yaml
import torch
from dataset.factorVAE_score import compute_factor_vae
from dataset.SAP_score import compute_sap
from model.mainNet import ReasonNet


def evaluate(output_dir, represent_fn, evaluation_fn, dataset, filename):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    experiment_timer = time.time()

    print('model: %s \t metric: %s \t dataset: %s' % (output_dir, filename, dataset))
    mean, std = evaluation_fn(dataset, represent_fn)

    results_dir = os.path.join(output_dir, "results_disentangle_%s.pkl" % filename)
    print('Execution time: %f' % (time.time() - experiment_timer))
    print('Mean: %.4f \t Var: %f' % (mean, std))
    saver(results_dir, {
        'mean': mean,
        'std': std
    })


def saver(path, obj):
    output = open(path, 'wb')
    pickle.dump(obj, output)
    output.close()


def loader(path):
    pkl_file = open(path, 'rb')
    output = pickle.load(pkl_file)
    pkl_file.close()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/main.yaml', type=str)
    parser.add_argument('--model_root', default='saves', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_root', default='saves', type=str)
    parser.add_argument('--eval_fn_list', default=['factorvae', 'sap'], type=list)
    parser.add_argument('--latent_dim', default=5, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    output_dir = os.path.join(args.model_root, args.exp_name)
    model_path = os.path.join(output_dir, 'checkpoint.pth')

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if val is not None:
            config[key] = val
    args.__dict__ = config

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    vae = ReasonNet(args, device)
    vae = vae.to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    vae = vae.eval()

    def represent_fn(x):
        x = torch.from_numpy(x).float() / 255
        x = x.to(device)
        z_params = vae.baseNet.encoder.forward(x).view(x.size(0), vae.baseNet.z_dim, vae.baseNet.q_dist.nparams)
        zs = vae.baseNet.q_dist.sample(params=z_params)
        return zs.detach().cpu().numpy()

    for eval_name in args.eval_fn_list:
        print('Evaluate by %s' % eval_name)
        eval_fn = None
        if eval_name == 'factorvae':
            eval_fn = compute_factor_vae
        elif eval_name == 'sap':
            eval_fn = compute_sap
        evaluate(output_dir, represent_fn, eval_fn, args.dataset, eval_name)

