import os
import os.path as path
import io
import argparse
import shutil
import lmdb
import glob
from tempfile import TemporaryFile
import numpy as np
from PIL import Image


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


class SVRTDealer:
    
    def __init__(self, root, cache_root):
        self.name = 'base'
        self.part_list = ['test']
        self.data_used = {'train': -1, 'val': 5000, 'test': -1}
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.cache_root = path.join(cache_root, 'SVRT')
        mkdir_with_check(self.cache_root)

    def save_data(self):
        print('Start dealing SVRT dataset')
        mkdir_with_check(os.path.join(self.cache_root, self.name))
        for part in self.part_list:
            mkdir_with_check(os.path.join(self.cache_root, self.name, part), delete=True)
            self.deal(part)

    def deal(self, part):
        # get npz file set
        env_path = os.path.join(self.cache_root, self.name, part)
        env = lmdb.open(env_path, map_size=1099511627776)
        cache = {}
        cnt = 1
        for tid in range(1, 24):
            problems_1 = glob.glob(path.join(self.root, 'results_problem_{}'.format(tid), 'sample_0_*.png'))
            problems_2 = glob.glob(path.join(self.root, 'results_problem_{}'.format(tid), 'sample_1_*.png'))
            problems_1 = sorted(problems_1)
            problems_2 = sorted(problems_2)
            num_prob = 11
            # load data
            for pid in range(num_prob):
                panel = np.zeros((128, 128 * 18, 3), dtype=np.uint8)
                problem = []
                for p in problems_1[pid * 9: (pid + 1) * 9]:
                    im = Image.open(p).convert('RGB')
                    problem.append(np.array(im.resize((128, 128))).astype(np.uint8))
                for p in problems_2[pid * 9: (pid + 1) * 9]:
                    im = Image.open(p).convert('RGB')
                    problem.append(np.array(im.resize((128, 128))).astype(np.uint8))
                img_key = 'problem-%09d' % cnt
                for i in range(18):
                    panel[:, i * 128:(i + 1) * 128] = problem[i]
                panel = Image.fromarray(panel).convert('RGB')
                img_byte = io.BytesIO()
                panel.save(img_byte, format='JPEG')
                byte_res = img_byte.getvalue()
                cache[img_key] = byte_res
                cache['task-%09d' % cnt] = str(tid).encode()
                if cnt % 1000 == 0:
                    write_cache(env, cache)
                    cache = {}
                cnt += 1
            print("Load %d/%d" % (tid, 23), end='\r')
        cache['num-samples'] = str(cnt - 1).encode()
        write_cache(env, cache)
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    dealer = SVRTDealer(root=args.input_path, cache_root=args.output_path)
    dealer.save_data()
