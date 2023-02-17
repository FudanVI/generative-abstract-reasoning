import os
import os.path as path
import random
import h5py
import torch
from torch.utils.data import Dataset


class MPI3DDataset(Dataset):
    def __init__(self, name='real_complex', part='train', size=64, cache_root='cache'):
        super(MPI3DDataset, self).__init__()
        self.name = name
        self.part = part
        self.size = size
        self.num_node = 8
        self.cond_range = torch.tensor([[0.0, 1.0]]).float()
        self.cache_root = os.path.join(cache_root, 'MPI3D')
        if not self.check_cache():
            raise FileNotFoundError('Cached dataset file not found.')
        self.image = None
        self.cls = None
        self.label = None
        self.num_data = None
        self.load_data_from_cache()

    def __getitem__(self, index):
        return self.image[index], self.label[index], self.cls[index]

    def __len__(self):
        return self.num_data

    def process_input(self, images, labels, classes):
        rand_index = list(range(40))
        random.shuffle(rand_index)
        rand_index = torch.tensor(rand_index[:self.num_node]).long()
        return images.index_select(1, rand_index), labels.index_select(1, rand_index), classes

    def data_cache_root(self):
        return path.join(self.cache_root, self.name, '{}_{}.hdf5'.format(self.part, self.size))

    def check_cache(self):
        return path.exists(self.data_cache_root())

    def load_data_from_cache(self):
        with h5py.File(self.data_cache_root(), 'r') as f:
            self.image = torch.from_numpy(f['image'][:]).float() / 255.
            self.cls = torch.from_numpy(f['classes'][:]).long()
        self.num_data = self.image.size(0)
        self.label = torch.arange(40).float()[None].expand(self.num_data, -1)[..., None] / 40.
