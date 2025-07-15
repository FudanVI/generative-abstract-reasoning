import os
import csv
import os.path as path
import io
import random
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


class G1SETDealer:
    def __init__(self, root, cache_root):
        self.name = 'base'
        self.part_list = ['test']
        self.data_used = {'test': -1}
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.cache_root = path.join(cache_root, 'G1SET')
        mkdir_with_check(self.cache_root)

    def save_data(self):
        print('Start dealing G1SET dataset')
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
        problems = glob.glob(path.join(self.root, '*.png'))
        problems = sorted(problems)
        csvfile = open(path.join(self.root, 'answers.csv'), newline='')
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        labels = []
        for row in spamreader:
            labels.append(int(row[0]))
        csvfile.close()
        if self.data_used[part] > 0:
            random.shuffle(problems)
            problems = problems[:self.data_used[part]]
        num_data = len(problems)
        print("Find %d npz file to load" % num_data)
        # load data
        for index, p in enumerate(problems):
            im_rgba = Image.open(p)
            im = Image.new('RGB', im_rgba.size, (255, 255, 255))
            im.paste(im_rgba, mask=im_rgba.split()[3])  # 3 is the alpha channel
            num_image = int(im.size[0] / 100)
            panel = np.zeros((128, 128 * num_image, 3), dtype=np.uint8)
            problem = np.stack(
                [np.array(im.crop((100 * i, 0, 100 + 100 * i, 100)).resize((128, 128))).astype(np.uint8)
                 for i in range(num_image)], axis=0
            )
            img_key = 'problem-%09d' % cnt
            for i in range(num_image):
                panel[:, i * 128:(i + 1) * 128] = problem[i]
            panel = Image.fromarray(panel).convert('RGB')
            img_byte = io.BytesIO()
            panel.save(img_byte, format='JPEG')
            byte_res = img_byte.getvalue()
            cache[img_key] = byte_res
            cache['answer-%09d' % cnt] = str(labels[index]).encode()
            if cnt % 1000 == 0:
                write_cache(env, cache)
                cache = {}
            cnt += 1
            print("Load %d/%d" % (index, num_data), end='\r')
        cache['num-samples'] = str(num_data).encode()
        write_cache(env, cache)
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    dealer = G1SETDealer(root=args.input_path, cache_root=args.output_path)
    dealer.save_data()
