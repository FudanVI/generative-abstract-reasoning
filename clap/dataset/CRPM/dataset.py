import os
import h5py
import torch
from torch.utils.data import Dataset


class CRPMDataset(Dataset):
    def __init__(self, name, cache_root='cache', part='train', size=64):
        super(CRPMDataset, self).__init__()
        self.cache_root = os.path.join(cache_root, 'CRPM')
        self.part = part
        self.size = size
        self.name = name
        assert self.check_file_exist()
        with h5py.File(os.path.join(self.cache_root, self.name, '{}_{}.hdf5'.format(self.part, self.size)), 'r') as f:
            self.panel = torch.from_numpy(f['panel'][:]).float()
            self.label = torch.from_numpy(f['label'][:]).long()
        self.num_data = self.panel.size(0)

    def process_input(self, panel, coord, classes):
        return panel, coord, classes

    def __getitem__(self, index):
        x = torch.tensor([-1., 0., 1.])[..., None].expand(-1, 3)
        y = torch.tensor([-1., 0., 1.])[None, ...].expand(3, -1)
        coord = torch.cat([x[..., None], y[..., None]], -1).reshape(9, 2)
        return self.panel[index] / 255., coord, self.label[index][..., 0]

    def __len__(self):
        return self.num_data

    def check_file_exist(self):
        return os.path.exists(os.path.join(self.cache_root, self.name, '{}_{}.hdf5'.format(self.part, self.size)))
