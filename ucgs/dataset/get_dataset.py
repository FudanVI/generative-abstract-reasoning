import random
import torch
from torch.utils.data import DataLoader
from dataset.RAVEN.dataset import RAVENDataset
from dataset.PGM.dataset import PGMDataset
from dataset.VAP.dataset import VAPDataset
from dataset.G1SET.dataset import G1SETDataset
from dataset.SVRT.dataset import SVRTDataset
torch.multiprocessing.set_sharing_strategy('file_system')


DATA_CFG = {
    'RAVEN': RAVENDataset,
    'PGM': PGMDataset,
    'VAP': VAPDataset,
    'G1SET': G1SETDataset,
    'SVRT': SVRTDataset,
}


class CircleDataLoader:

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.data_iters = [iter(dataloader) for dataloader in dataloaders]
        self.len = len(dataloaders)

    def __iter__(self):
        return self

    def __next__(self):
        index = random.randint(0, self.len - 1)
        try:
            data = next(self.data_iters[index])
        except StopIteration:
            self.data_iters[index] = iter(self.dataloaders[index])
            data = next(self.data_iters[index])
        return data


def get_dataset(args, test=False, vis=False):
    if not test:
        train_dataloaders, valid_dataloaders = [], []
        for dataset_name in args.dataset.split(','):
            dataset_info = args.dataset_info[dataset_name]
            instance_names = dataset_info['instances']
            assert dataset_name in DATA_CFG
            for instance_name in instance_names:
                train_dataset = DATA_CFG[dataset_name](
                    args.size, cache_root=dataset_info['src'],
                    name=instance_name, part='train'
                )
                train_dataloaders.append(DataLoader(
                    train_dataset, batch_size=args.batch,
                    shuffle=True, num_workers=8, collate_fn=None
                ))
                valid_dataset = DATA_CFG[dataset_name](
                    args.size, cache_root=dataset_info['src'],
                    name=instance_name, part='val'
                )
                valid_dataloaders.append(DataLoader(
                    valid_dataset, batch_size=args.batch,
                    shuffle=True, num_workers=8, collate_fn=None
                ))
        return CircleDataLoader(train_dataloaders), \
            CircleDataLoader(valid_dataloaders)
    else:
        test_dataloaders = []
        for dataset_name in args.dataset.split(','):
            dataset_info = args.dataset_info[dataset_name]
            instance_names = dataset_info['instances']
            assert dataset_name in DATA_CFG
            for instance_name in instance_names:
                test_dataset = DATA_CFG[dataset_name](
                    args.size, cache_root=dataset_info['src'],
                    name=instance_name, part='test'
                )
                test_dataloaders.append(DataLoader(
                    test_dataset, batch_size=args.batch,
                    shuffle=vis, num_workers=8, collate_fn=None
                ))
        return test_dataloaders
