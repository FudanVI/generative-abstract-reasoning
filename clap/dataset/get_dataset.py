from dataset.balls.dataset import BallsDataset
from dataset.MPI3D.dataset import MPI3DDataset
from dataset.CRPM.dataset import CRPMDataset


def get_dataset(name, dataset_name, cache_root='cache', train=True, size=64):
    if train:
        if dataset_name == 'CRPM':
            train_set = CRPMDataset(name, cache_root=cache_root, size=size)
            val_set = CRPMDataset(name, cache_root=cache_root, part='val', size=size)
        elif dataset_name == 'balls':
            train_set = BallsDataset(name, cache_root=cache_root, part='train')
            val_set = BallsDataset(name, cache_root=cache_root, part='val')
        elif dataset_name == 'MPI3D':
            train_set = MPI3DDataset(name, cache_root=cache_root, part='train')
            val_set = MPI3DDataset(name, cache_root=cache_root, part='val')
        else:
            train_set, val_set = None, None
        return train_set, val_set, len(train_set), len(val_set)
    else:
        if dataset_name == 'CRPM':
            test_set = CRPMDataset(name, cache_root=cache_root, part='test', size=size)
        elif dataset_name == 'balls':
            test_set = BallsDataset(name, cache_root=cache_root, part='test')
        elif dataset_name == 'MPI3D':
            test_set = MPI3DDataset(name, cache_root=cache_root, part='test')
        else:
            test_set = None
        return test_set, len(test_set)
