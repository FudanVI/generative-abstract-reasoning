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


class PGMDealer:
    def __init__(self, root, cache_root):
        self.name_list = ['neutral']
        self.part_list = ['train', 'val', 'test']
        self.data_used = {'train': -1, 'val': 5000, 'test': -1}
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.cache_root = path.join(cache_root, 'PGM')
        mkdir_with_check(self.cache_root)

    def save_data(self):
        print('Start dealing PGM dataset')
        for name in self.name_list:
            print('Deal {}'.format(name))
            mkdir_with_check(os.path.join(self.cache_root, name))
            for part in self.part_list:
                mkdir_with_check(os.path.join(self.cache_root, name, part), delete=True)
                self.deal(name, part)

    def deal(self, name, part):
        # get npz file set
        full_path = os.path.join(self.root, 'neutral')
        env_path = os.path.join(self.cache_root, name, part)
        env = lmdb.open(env_path, map_size=1099511627776)
        cache = {}
        cnt = 1
        npz_path = []
        print(full_path)
        for (dir_root, dirs, files) in os.walk(full_path):
            for file in files:
                if part == "train" and file.startswith('PGM_neutral_train') and file.endswith('.npz'):
                    npz_path.append(os.path.join(dir_root, file))
                elif part == "val" and file.startswith('PGM_neutral_val') and file.endswith('.npz'):
                    npz_path.append(os.path.join(dir_root, file))
                elif part == "test" and file.startswith('PGM_neutral_test') and file.endswith('.npz'):
                    npz_path.append(os.path.join(dir_root, file))
        num_data = len(npz_path)
        if self.data_used[part] > 0:
            num_data = self.data_used[part]
        random.shuffle(npz_path)
        npz_path = npz_path[:num_data]
        print("Find %d npz file to load" % num_data)
        # load data
        for index, npz in enumerate(npz_path):
            panel = np.zeros((160, 160 * 16), dtype=np.uint8)
            obj = np.load(npz)
            problem = obj["image"].reshape(16, 160, 160).astype(np.uint8)
            img_key = 'problem-%09d' % cnt
            for i in range(16):
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
    dealer = PGMDealer(root=args.input_path, cache_root=args.output_path)
    dealer.save_data()
