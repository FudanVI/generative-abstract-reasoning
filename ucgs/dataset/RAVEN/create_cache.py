import os
import os.path as path
import io
import argparse
import shutil
import xml.etree.ElementTree as ET
import random
from PIL import Image
import lmdb
from tempfile import TemporaryFile
import numpy as np


CONFIG = {
    'center_single': {
        'concept': ['Type0', 'Size0', 'Color0'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'in_center_single_out_center_single': {
        'concept': ['Type0', 'Size0', 'Type1', 'Size1', 'Color1'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'left_center_single_right_center_single': {
        'concept': ['Type0', 'Size0', 'Color0', 'Type1', 'Size1', 'Color1'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'up_center_single_down_center_single': {
        'concept': ['Type0', 'Size0', 'Color0', 'Type1', 'Size1', 'Color1'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'in_distribute_four_out_center_single': {
        'concept': ['Type0', 'Position1', 'Type1', 'Size1', 'Color1'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'distribute_four': {
        'concept': ['Number0', 'Type0', 'Size0', 'Color0'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'distribute_nine': {
        'concept': ['Position0', 'Type0', 'Size0', 'Color0'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'in_distribute_four_out_center_single_uniform': {
        'concept': ['Type0', 'Position1', 'Type1', 'Size1', 'Color1'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'distribute_four_uniform': {
        'concept': ['Number0', 'Type0', 'Size0', 'Color0'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
    'distribute_nine_uniform': {
        'concept': ['Position0', 'Type0', 'Size0', 'Color0'],
        'rule': ['Constant', 'Progression', 'Arithmetic', 'Distribute_Three']
    },
}


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


class RavenDealer:
    def __init__(self, root, cache_root):
        self.name_list = [
            'center_single',
            'distribute_four',
            'distribute_nine',
            'in_center_single_out_center_single',
            'in_distribute_four_out_center_single',
            'left_center_single_right_center_single',
            'up_center_single_down_center_single'
        ]
        self.name = 'base'
        self.part_list = ['train', 'val', 'test']
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.cache_root = path.join(cache_root, 'RAVEN')
        mkdir_with_check(self.cache_root)

    def save_data(self):
        print('Start dealing RAVEN dataset')
        mkdir_with_check(os.path.join(self.cache_root, self.name))
        for part in self.part_list:
            mkdir_with_check(os.path.join(self.cache_root, self.name, part), delete=True)
            self.deal(part)

    @staticmethod
    def deal_xml(p, name):
        tree = ET.parse(p)
        root = tree.getroot()
        concepts = CONFIG[name]['concept']
        rules = CONFIG[name]['rule']
        label = np.zeros(len(concepts)).astype(np.uint8)
        for rule_group in root[1]:
            group_id = rule_group.attrib['id']
            for rule in rule_group:
                attr = rule.attrib['attr'] + group_id
                if attr not in concepts:
                    continue
                name = rule.attrib['name']
                label[concepts.index(attr)] = rules.index(name)
        return label

    def deal(self, part):
        # get npz file set
        env_path = os.path.join(self.cache_root, self.name, part)
        env = lmdb.open(env_path, map_size=1099511627776)
        cache = {}
        cnt = 1
        npz_path = []
        for name in self.name_list:
            full_path = os.path.join(self.root, name)
            for (dir_root, dirs, files) in os.walk(full_path):
                for file in files:
                    if part == "train" and file.endswith('_train.npz'):
                        npz_path.append(os.path.join(dir_root, file))
                    elif part == "val" and file.endswith('_val.npz'):
                        npz_path.append(os.path.join(dir_root, file))
                    elif part == "test" and file.endswith('_test.npz'):
                        npz_path.append(os.path.join(dir_root, file))
        num_data = len(npz_path)
        print("Find %d npz file to load" % num_data)
        # load data
        for index, npz in enumerate(npz_path):
            panel = np.zeros((160, 160 * 16), dtype=np.uint8)
            obj = np.load(npz)
            problem = obj["image"].astype(np.uint8)
            img_key = 'problem-%09d' % cnt
            for i in range(16):
                panel[:, i*160:(i+1)*160] = problem[i]
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
    dealer = RavenDealer(root=args.input_path, cache_root=args.output_path)
    dealer.save_data()
