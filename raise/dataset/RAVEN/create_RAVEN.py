import os
import os.path as path
import argparse
import xml.etree.ElementTree as ET
import shutil
import h5py
import torch
import numpy as np
from torch.nn.functional import interpolate


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='RAVEN-10000', type=str)
    parser.add_argument('--output_path', default='cache', type=str)
    parser.add_argument('--size', default=64, type=int)
    arguments = parser.parse_args()
    return arguments


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


class RavenDealer:
    def __init__(self, root='RAVEN-10000', size=64, cache_root='cache'):
        self.name_list = [
            'center_single',
            'distribute_four',
            'distribute_four_uniform',
            'distribute_nine',
            'distribute_nine_uniform',
            'in_center_single_out_center_single',
            'in_distribute_four_out_center_single',
            'in_distribute_four_out_center_single_uniform',
            'left_center_single_right_center_single',
            'up_center_single_down_center_single'
        ]
        self.part_list = ['train', 'val', 'test']
        self.root = root
        assert os.path.exists(os.path.join(self.root))
        self.size = size
        self.cache_root = path.join(cache_root, 'RAVEN')
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)

    def save_data(self):
        print('Start dealing RAVEN dataset')
        for name in self.name_list:
            print('Deal {}'.format(name))
            if os.path.exists(os.path.join(self.cache_root, name)):
                shutil.rmtree(os.path.join(self.cache_root, name))
            os.makedirs(os.path.join(self.cache_root, name))
            for part in self.part_list:
                panels, selections, answers, labels = self.deal(name, part)
                with h5py.File(path.join(self.cache_root, name, '{}_{}.hdf5'.format(part, self.size)), 'w') as f:
                    f['panel'] = panels
                    f['selection'] = selections
                    f['answer'] = answers
                    f['label'] = labels

    @staticmethod
    def deal_xml(p, name):
        tree = ET.parse(p)
        root = tree.getroot()
        concepts = CONFIG[name]['concept']
        rules = CONFIG[name]['rule']
        label = torch.zeros(len(concepts)).long()
        for rule_group in root[1]:
            group_id = rule_group.attrib['id']
            for rule in rule_group:
                attr = rule.attrib['attr'] + group_id
                if attr not in concepts:
                    continue
                name = rule.attrib['name']
                label[concepts.index(attr)] = rules.index(name)
        return label

    def deal(self, name, part):
        # get npz file set
        full_path = os.path.join(self.root, name)
        npz_path = []
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
        panels = torch.zeros(num_data, 8, 1, self.size, self.size).float()
        selections = torch.zeros(num_data, 8, 1, self.size, self.size).float()
        answers = torch.zeros(num_data).long()
        cfg = CONFIG[name]
        labels = torch.zeros(num_data, len(cfg['concept'])).long()
        for index, npz in enumerate(npz_path):
            obj = np.load(npz)
            panels[index] = interpolate(
                torch.from_numpy(obj["image"][:8, :, :]).view(8, 1, 160, 160).float(),
                self.size, mode='bilinear', align_corners=False
            )
            selections[index] = interpolate(
                torch.from_numpy(obj["image"][8:, :, :]).view(8, 1, 160, 160).float(),
                self.size, mode='bilinear', align_corners=False
            )
            answers[index] = torch.from_numpy(obj["target"]).long()
            labels[index] = self.deal_xml(npz.replace('.npz', '.xml'), name)
            print("Load %d/%d" % (index, num_data), end='\r')
        return panels, selections, answers, labels


if __name__ == '__main__':
    args = get_config()
    dealer = RavenDealer(root=args.dataset_root, size=args.size, cache_root=args.output_path)
    dealer.save_data()
