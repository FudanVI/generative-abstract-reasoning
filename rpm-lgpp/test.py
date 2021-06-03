import os
import argparse
import yaml
import pickle
import torch
from torch.utils.data import DataLoader
from dataset.PolyMR import PolyMRDataset
from model.mainNet import ReasonNet


def mean_var(l):
    var, mean = torch.var_mean(l)
    return mean.item(), var.sqrt().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/main.yaml', type=str)
    parser.add_argument('--model_root', default='saves', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_root', default='saves', type=str)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--latent_dim', default=5, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_exp', default=10, type=int)
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
        device_id = "cuda:%d" % args.gpu
    else:
        device_id = 'cpu'
    device = torch.device(device_id)

    test_set = PolyMRDataset(size=args.image_size, type=args.dataset, set='test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=False, drop_last=False, num_workers=0)
    img_size = test_set.size

    vae = ReasonNet(args, device)
    vae = vae.to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device_id)['state_dict'])
    vae = vae.eval()

    num_loop = args.num_exp
    mse_list = []
    for loop_iter in range(num_loop):
        sum_mse, num_sample = 0, 0
        with torch.no_grad():
            for i, (x, t) in enumerate(test_loader):
                bs = x.size(0)
                x = x.view(-1, 1, img_size, img_size).to(device)
                t = t.view(-1, 1, img_size, img_size).to(device)
                panel = torch.cat((x.view(bs, 8, 1, img_size, img_size), t.unsqueeze(1)), 1).contiguous()
                mask_origin = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).to(device)
                mse_inner = 0
                for row in range(3):
                    for col in range(3):
                        mask = mask_origin.clone()
                        mask[row, col] = 0
                        recon = vae.predict(x, t, mask).view(bs, -1, 1, img_size, img_size)
                        zero_index = (torch.ones(9).to(device) - mask.view(-1)).nonzero().view(-1)
                        recon_answer = torch.index_select(recon, 1, zero_index)
                        recon_answer_gt = torch.index_select(panel, 1, zero_index)
                        mse = (recon_answer_gt - recon_answer).pow(2).view(bs, -1).mean(-1)
                        mse_inner += mse
                sum_mse += (mse_inner / 9).sum().item()
                num_sample += bs
                print('[%d/%d][%d/%d]' % (loop_iter + 1, num_loop, num_sample, len(test_set)), end='\r')
        mean_mse = sum_mse / num_sample
        mse_list.append(mean_mse)
    mses = torch.FloatTensor(mse_list)
    print('MSE: mean %.4f \t var %f' % mean_var(mses))
    output_f = open(os.path.join(output_dir, 'result_mse.pkl'), 'wb')
    pickle.dump({
        'mse': mean_var(mses)
    }, output_f)
    output_f.close()


if __name__ == '__main__':
    main()
