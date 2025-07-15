import os.path as path
import lmdb
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
import torchvision.transforms as transforms
import six
from PIL import Image
import numpy as np


def load_from_lmdb(txn, k):
    content_buf = txn.get(k.encode())
    buf = six.BytesIO()
    buf.write(content_buf)
    buf.seek(0)
    content = np.load(buf)
    buf.close()
    return content


class SVRTDataset(Dataset):

    def __init__(self, size, cache_root='cache', name='base', part='train'):
        super(SVRTDataset, self).__init__()
        self.name = name
        self.part = part
        self.size = size
        self.cache_root = cache_root
        if not self.check_cache():
            raise FileNotFoundError('Cached dataset file not found.')
        self.env = lmdb.open(
            self.data_cache_root(), max_readers=1, readonly=True,
            lock=False, readahead=False, meminit=False
        )
        if not self.env:
            raise FileNotFoundError('cannot find dataset file: {}'.format(self.data_cache_root()))
        self.num_data = None
        with self.env.begin(write=False) as txn:
            self.num_data = int(txn.get('num-samples'.encode()))

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index < len(self), 'index range error index: %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'problem-%09d' % index
            img_buf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            try:
                imgs = Image.open(buf).convert('RGB')
                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            imgs = transforms.ToTensor()(imgs)
            imgs = [imgs[:, :, i * 128:(i + 1) * 128] for i in range(18)]
            imgs = torch.stack(imgs, dim=0)
            imgs = interpolate(imgs, self.size, mode='bilinear', align_corners=False)
            panel, answer, distractors = imgs, imgs[-1:], imgs[-1:]
        return panel, answer, distractors

    def __len__(self):
        return self.num_data

    def data_cache_root(self):
        return path.join(self.cache_root, self.name, self.part)

    def check_cache(self):
        return path.exists(self.data_cache_root())
