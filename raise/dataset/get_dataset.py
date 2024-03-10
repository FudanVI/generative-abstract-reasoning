from dataset.RAVEN.dataset import RavenDataset
from dataset.IRAVEN.dataset import IRavenDataset


def get_dataset(name, dataset_name, cache_root='cache', part='train', kargs=None):
    if dataset_name == 'RAVEN':
        dataset = RavenDataset(name, cache_root=cache_root, part=part, **kargs)
    elif dataset_name == 'I-RAVEN':
        dataset = IRavenDataset(name, cache_root=cache_root, part=part, **kargs)
    else:
        dataset = None
    return dataset, len(dataset)
