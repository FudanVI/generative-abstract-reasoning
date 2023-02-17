import os
import os.path as path
import random
import h5py
import torch


class Drawer:
    def __init__(self, name, part='test', size=64, cache_root='cache'):
        super(Drawer, self).__init__()
        self.name = name
        if self.name == 'real_complex':
            self.bounds = [4, 4, 2, 3, 3, 2, 2]
        else:
            self.bounds = [6, 6, 2, 3, 3, 2, 2]
        self.part = part
        self.size = size
        self.num_node = 8
        self.cond_range = torch.tensor([[0.0, 1.0]]).float()
        self.cache_root = os.path.join(cache_root, 'MPI3D')
        if not self.check_cache():
            raise FileNotFoundError('Cached dataset file not found.')
        self.image = None
        self.classes = None
        self.label = None
        self.num_data = None
        self.load_data_from_cache()

    def process_input(self, images, labels, classes):
        rand_index = list(range(40))
        rand_index = torch.tensor(rand_index[:self.num_node]).long()
        return images.index_select(1, rand_index), labels.index_select(1, rand_index), classes

    def data_cache_root(self):
        return path.join(self.cache_root, self.name, '{}_{}.hdf5'.format(self.part, self.size))

    def check_cache(self):
        return path.exists(self.data_cache_root())

    def load_data_from_cache(self):
        with h5py.File(self.data_cache_root(), 'r') as f:
            self.image = torch.from_numpy(f['image'][:]).float() / 255.
            self.classes = torch.from_numpy(f['classes'][:]).long()
        self.num_data = self.image.size(0)
        self.label = torch.arange(40).float()[None].expand(self.num_data, -1)[..., None] / 40.

    def get_attr(self):
        if self.name == 'real_complex':
            return ['Scene', 'Object\nAxis']
        else:
            raise NotImplementedError

    def draw_batch(self, batch_size, index=None, fix=True):
        if index is None:
            index_can_select = torch.randperm(self.num_data)
        elif index == 0:
            fix_cls = torch.tensor([
                random.randint(0, self.bounds[0] - 1),
                random.randint(0, self.bounds[1] - 1),
                random.randint(0, self.bounds[2] - 1),
                random.randint(0, self.bounds[3] - 1),
                random.randint(0, self.bounds[4] - 1)
            ])
            index_can_select = torch.nonzero((self.classes[:, 0:5] == fix_cls).float()).reshape(-1)
            index_can_select = index_can_select[torch.randperm(index_can_select.size(0))]
        elif index == 1:
            fix_cls = torch.tensor([
                random.randint(0, self.bounds[5] - 1)
            ])
            index_can_select = torch.nonzero((self.classes[:, 5] == fix_cls).float()).reshape(-1)
            index_can_select = index_can_select[torch.randperm(index_can_select.size(0))]
        num_select = min(batch_size, index_can_select.size(0))
        index_can_select = index_can_select[:num_select]
        images, labels = self.image[index_can_select], self.label[index_can_select]
        images, labels, _ = self.process_input(images, labels, None)
        images, labels = images.numpy(), labels.numpy()
        return images, labels
