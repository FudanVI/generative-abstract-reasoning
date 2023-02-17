import os
import os.path as path
import argparse
import shutil
import h5py
import torch
import numpy as np
from torch.nn.functional import interpolate


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='MPI3D', type=str)
    parser.add_argument('--output_path', default='cache', type=str)
    parser.add_argument('--size', default=64, type=int)
    arguments = parser.parse_args()
    return arguments


class MPI3DDealer:
    def __init__(self, root='MPI3D', size=64, cache_root='cache'):
        self.name_list = ['real_complex']
        self.npz_list = [
            'real3d_complicated_shapes_ordered.npz'
        ]
        self.part_list = ['train', 'val', 'test']
        self.port_num = [0.7, 0.8, 1.0]
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.size = size
        self.cache_root = path.join(cache_root, 'MPI3D')
        if os.path.exists(self.cache_root):
            shutil.rmtree(self.cache_root)
        os.makedirs(self.cache_root)

    def save_data(self):
        print('Start dealing MPI3D dataset')
        for name, npz in zip(self.name_list, self.npz_list):
            print('Deal {}'.format(name))
            os.makedirs(os.path.join(self.cache_root, name))
            images, classes = self.get_npz(name, npz)
            split_pos1, split_pos2 = \
                int(images.shape[0] * self.port_num[0]), int(images.shape[0] * self.port_num[1])
            name_path = path.join(self.cache_root, name)
            with h5py.File(path.join(name_path, 'train_{}.hdf5'.format(self.size)), 'w') as f:
                f['image'] = images[:split_pos1]
                f['classes'] = classes[:split_pos1]
            with h5py.File(path.join(name_path, 'val_{}.hdf5'.format(self.size)), 'w') as f:
                f['image'] = images[split_pos1:split_pos2]
                f['classes'] = classes[split_pos1:split_pos2]
            with h5py.File(path.join(name_path, 'test_{}.hdf5'.format(self.size)), 'w') as f:
                f['image'] = images[split_pos2:]
                f['classes'] = classes[split_pos2:]

    def get_npz(self, name, npz):
        # get npz file set
        full_path = os.path.join(self.root, npz)
        objs = np.load(full_path)['images']
        if name == 'real_complex':
            objs = objs.reshape([4, 4, 2, 3, 3, 40, 40, 64, 64, 3])
            labels_horizontal = np.stack(np.meshgrid(
                np.array(range(4)), np.array(range(4)), np.array(range(2)),
                np.array(range(3)), np.array(range(3)), np.array([1]), np.array([0] * 40),
                indexing='ij'
            ), axis=-1)
            labels_vertical = np.stack(np.meshgrid(
                np.array(range(4)), np.array(range(4)), np.array(range(2)),
                np.array(range(3)), np.array(range(3)), np.array([0] * 40), np.array([1]),
                indexing='ij'
            ), axis=-1)
        else:
            objs = objs.reshape([6, 6, 2, 3, 3, 40, 40, 64, 64, 3])
            labels_horizontal = np.stack(np.meshgrid(
                np.array(range(6)), np.array(range(6)), np.array(range(2)),
                np.array(range(3)), np.array(range(3)), np.array([1]), np.array([0] * 40),
                indexing='ij'
            ), axis=-1)
            labels_vertical = np.stack(np.meshgrid(
                np.array(range(6)), np.array(range(6)), np.array(range(2)),
                np.array(range(3)), np.array(range(3)), np.array([0] * 40), np.array([1]),
                indexing='ij'
            ), axis=-1)
        horizontal_objs = np.transpose(objs, (0, 1, 2, 3, 4, 6, 5, 9, 7, 8)).reshape([-1, 40, 3, 64, 64])
        horizontal_classes = np.transpose(labels_horizontal, (0, 1, 2, 3, 4, 6, 5, 7)).reshape([-1, 7])
        vertical_objs = np.transpose(objs, (0, 1, 2, 3, 4, 5, 6, 9, 7, 8)).reshape([-1, 40, 3, 64, 64])
        vertical_classes = np.transpose(labels_vertical, (0, 1, 2, 3, 4, 5, 6, 7)).reshape([-1, 7])
        objs = np.concatenate([horizontal_objs, vertical_objs], axis=0)
        classes = np.concatenate([horizontal_classes, vertical_classes], axis=0)
        num_data = objs.shape[0]
        print("Find %d panels" % num_data)
        if self.size != 64:
            objs = interpolate(
                torch.from_numpy(objs).reshape(-1, 3, 64, 64).float(),
                self.size, mode='bilinear', align_corners=False
            ).reshape(num_data, 40, 3, self.size, self.size).numpy()
        index = np.array(list(range(num_data)))
        np.random.shuffle(index)
        objs = objs[index]
        classes = classes[index]
        return objs, classes


if __name__ == '__main__':
    args = get_config()
    np.random.seed(0)
    dealer = MPI3DDealer(root=args.dataset_root, size=args.size, cache_root=args.output_path)
    dealer.save_data()
