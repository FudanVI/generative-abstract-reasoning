import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import lib.utils as utils
from dataset.PolyMR import PolyMRDataset
from model.mainNet import ReasonNet
from config import get_config


def main():
    args = get_config()

    # prepare path
    utils.clear_dir(args.tmp_path)
    utils.clear_dir(args.save_path)
    print('[*] Clear path')

    # choose device
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
        print('[*] Choose cuda:%d as device' % args.gpu)
    else:
        device = torch.device('cpu')
        print('[*] Choose cpu as device')

    # load training set and evaluation set
    train_set = PolyMRDataset(size=args.image_size, type=args.dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_set = PolyMRDataset(size=args.image_size, type=args.dataset, set='val')
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('[*] Load datasets')
    img_size = train_set.size
    dataset_size_train = len(train_set)
    dataset_size_valid = len(valid_set)

    # setup the model
    vae = ReasonNet(args, device)
    vae = vae.to(device)
    print('[*] Load model')

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # training loop
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    # record best elbo and epoch
    best_elbo = - 1e6
    best_epoch = -1
    for epoch in range(args.num_epochs):
        vae.train()
        for i, (x, t) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.view(-1, 1, img_size, img_size).to(device)
            t = t.view(-1, 1, img_size, img_size).to(device)
            obj, recon = vae(x, t, dataset_size_train)
            obj.mul(-1).backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
            optimizer.step()
            iteration += 1

        vae.eval()
        elbo_running_mean.reset()
        with torch.no_grad():
            for i, (x, t) in enumerate(valid_loader):
                x = x.view(-1, 1, img_size, img_size).to(device)
                t = t.view(-1, 1, img_size, img_size).to(device)
                obj, elbo, recon = vae.evaluate(x, t, dataset_size_valid)
                elbo_running_mean.update(elbo)

            avg_elbo = elbo_running_mean.get_avg()['elbo']
            if avg_elbo > best_elbo:
                best_epoch = epoch
                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args,
                    'epoch': epoch}, args.save_path)
                best_elbo = avg_elbo

            if epoch % args.log_freq == 0:
                elbo_running_mean.log(epoch, args.num_epochs, best_epoch)


if __name__ == '__main__':
    main()
