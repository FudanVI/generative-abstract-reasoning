from dataset.RAVEN.dataset import RavenDataset
from dataset.IRAVEN.dataset import IRavenDataset


def get_dataset(name, dataset_name, cache_root='cache', train=True):
    if train:
        if dataset_name == 'RAVEN':
            train_set = RavenDataset(name, cache_root=cache_root, part='train')
            val_set = RavenDataset(name, cache_root=cache_root, part='val')
        elif dataset_name == 'I-RAVEN':
            train_set = IRavenDataset(name, cache_root=cache_root, part='train')
            val_set = IRavenDataset(name, cache_root=cache_root, part='val')
        else:
            train_set, val_set = None, None
        return train_set, val_set, len(train_set), len(val_set)
    else:
        if dataset_name == 'RAVEN':
            test_set = RavenDataset(name, cache_root=cache_root, part='test')
        elif dataset_name == 'I-RAVEN':
            test_set = IRavenDataset(name, cache_root=cache_root, part='train')
        else:
            test_set = None
        return test_set, len(test_set)
