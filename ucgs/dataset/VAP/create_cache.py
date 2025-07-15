import os
import os.path as path
import io
import random
import argparse
import shutil
import lmdb
from tempfile import TemporaryFile
from PIL import Image
import numpy as np


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def write_numpy_arr_to_cache(cache, key, arr):
    outfile = TemporaryFile()
    np.save(outfile, arr)
    outfile.seek(0)
    cache[key] = outfile.read()
    outfile.close()


def mkdir_with_check(p, delete=False):
    if delete:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)
    else:
        if not os.path.exists(p):
            os.makedirs(p)


class VAPDealer:

    name_to_src = {
        'transfer': ('novel.domain.transfer', 'analogy_novel.domain.transfer_{}'),
        'transfer-lbc': ('novel.domain.transfer', 'analogy_novel.domain.transfer_{}_lbc'),
        'transfer-normal': ('novel.domain.transfer', 'analogy_novel.domain.transfer_{}_normal'),
        'interpolation': ('interpolation', 'analogy_interpolation_{}'),
        'interpolation-lbc': ('interpolation', 'analogy_interpolation_{}_lbc'),
        'interpolation-normal': ('interpolation', 'analogy_interpolation_{}_normal'),
        'extrapolation': ('extrapolation', 'analogy_extrapolation_{}'),
        'extrapolation-lbc': ('extrapolation', 'analogy_extrapolation_{}_lbc'),
        'extrapolation-normal': ('extrapolation', 'analogy_extrapolation_{}_normal'),
        'line': ('novel.target.domain.line.type', 'analogy_novel.target.domain.line.type_{}'),
        'line-lbc': ('novel.target.domain.line.type', 'analogy_novel.target.domain.line.type_{}_lbc'),
        'line-normal': ('novel.target.domain.line.type', 'analogy_novel.target.domain.line.type_{}_normal'),
        'shape': ('novel.target.domain.shape.color', 'analogy_novel.target.domain.shape.color_{}'),
        'shape-lbc': ('novel.target.domain.shape.color', 'analogy_novel.target.domain.shape.color_{}_lbc'),
        'shape-normal': ('novel.target.domain.shape.color', 'analogy_novel.target.domain.shape.color_{}_normal'),
    }

    def __init__(self, root, cache_root):
        self.name_list = [
            'transfer', 'transfer-lbc', 'transfer-normal',
            'interpolation', 'interpolation-lbc', 'interpolation-normal',
            'extrapolation', 'extrapolation-lbc', 'extrapolation-normal',
            'line', 'line-lbc', 'line-normal', 'shape', 'shape-lbc', 'shape-normal'
        ]
        self.part_list = ['test']
        self.data_used = {'train': -1, 'val': 5000, 'test': -1}
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.cache_root = path.join(cache_root, 'VAP')
        mkdir_with_check(self.cache_root)

    def save_data(self):
        print('Start dealing VAP dataset')
        for name in self.name_list:
            print('Deal {}'.format(name))
            mkdir_with_check(os.path.join(self.cache_root, name))
            for part in self.part_list:
                mkdir_with_check(os.path.join(self.cache_root, name, part), delete=True)
                self.deal(name, part)

    def deal(self, name, part):
        # get npz file set
        full_path = os.path.join(self.root, self.name_to_src[name][0])
        env_path = os.path.join(self.cache_root, name, part)
        env = lmdb.open(env_path, map_size=1099511627776)
        cache = {}
        cnt = 1
        npz_path = []
        print(full_path)
        for (dir_root, dirs, files) in os.walk(full_path):
            for file in files:
                if file.startswith(self.name_to_src[name][1].format(part)) and file.endswith('.npz'):
                    npz_path.append(os.path.join(dir_root, file))
        num_data = len(npz_path)
        if self.data_used[part] > 0:
            num_data = self.data_used[part]
        random.shuffle(npz_path)
        npz_path = npz_path[:num_data]
        print("Find %d npz file to load" % num_data)
        # load data
        for index, npz in enumerate(npz_path):
            panel = np.zeros((160, 160 * 9), dtype=np.uint8)
            obj = np.load(npz)
            problem = obj["image"].reshape(9, 160, 160).astype(np.uint8)
            img_key = 'problem-%09d' % cnt
            for i in range(9):
                panel[:, i * 160:(i + 1) * 160] = problem[i]
            panel = Image.fromarray(panel).convert('RGB')
            img_byte = io.BytesIO()
            panel.save(img_byte, format='JPEG')
            byte_res = img_byte.getvalue()
            cache[img_key] = byte_res
            cache['answer-%09d' % cnt] = str(int(obj["target"])).encode()
            if cnt % 1000 == 0:
                write_cache(env, cache)
                cache = {}
            cnt += 1
            print("Load %d/%d" % (index, num_data), end='\r')
        cache['num-samples'] = str(num_data).encode()
        write_cache(env, cache)
        env.close()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    dealer = VAPDealer(root=args.input_path, cache_root=args.output_path)
    dealer.save_data()
