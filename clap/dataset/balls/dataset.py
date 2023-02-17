import os
import h5py
import torch
from torch.utils.data import Dataset


class BallsDataset(Dataset):
    def __init__(self, name, cache_root, part='train'):
        super(BallsDataset, self).__init__()
        self.name = name
        self.cache_root = cache_root
        self.part = part
        self.store_dir = os.path.join(self.cache_root, 'balls', self.name)
        self.check_file_exist()
        self.part = part
        self.video, self.background, self.objs, self.classes = None, None, None, None
        self.fps, self.frame_interval = None, None
        self.time = None
        self.num_data = None
        self.cond_range = None
        self.load_file()

    def check_file_exist(self):
        if not os.path.exists(self.store_dir):
            raise FileExistsError
        for part in ['train', 'val', 'test']:
            if not os.path.exists(os.path.join(self.store_dir, '{}.hdf5'.format(part))):
                raise FileExistsError

    def load_file(self):
        file_path = os.path.join(self.store_dir, '{}.hdf5'.format(self.part))
        with h5py.File(os.path.abspath(file_path), 'r') as f:
            self.video = torch.from_numpy(f['video'][:]).float().permute(0, 1, 4, 2, 3).contiguous() / 255.
            self.background = torch.from_numpy(f['background'][:]).float().permute(0, 3, 1, 2).contiguous() / 255.
            self.objs = torch.from_numpy(f['objs'][:]).float().permute(0, 1, 2, 5, 3, 4).contiguous() / 255.
            self.classes = torch.from_numpy(f['classes'][:])
            self.fps = f['fps'][()]
            self.frame_interval = 1. / self.fps
            self.cond_range = torch.tensor([[0.0, self.frame_interval * self.video.size(1)]]).float()
        self.num_data = self.video.size(0)
        times = [self.frame_interval * i for i in range(self.video.size(1))]
        self.time = torch.tensor(times).float().reshape(1, -1, 1).expand(self.num_data, -1, -1)

    def process_input(self, video, time, classes):
        time = time + torch.rand(time.size(0), 1, 1) * self.frame_interval
        return video, time, classes

    def __getitem__(self, index):
        return self.video[index], self.time[index], self.classes[index, 1]

    def __len__(self):
        return self.num_data
