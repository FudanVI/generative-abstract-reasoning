import numpy as np
from PIL import Image, ImageDraw
import argparse
import torch
import math
import random
import os
import h5py
import shutil
# from dataset.CRPM.utils import PanelRuleSampler
from utils import PanelRuleSampler


seed = 0
random.seed(seed)


class InstanceDrawer:
    def __init__(self, num_side, size_img):
        self.n = num_side
        self.s = size_img

    @ staticmethod
    def get_vertex(n, s, length, angle):
        def v2coord(theta, ms, r):
            return int(ms + r * math.cos(theta)), int(ms - r * math.sin(theta))
        inside_vertex = 2 * math.pi / n
        start_vertex = 1 / 2 * math.pi + ((n + 1) % 2) * inside_vertex / 2 + angle
        vertexes = []
        for i in range(n):
            vertexes.append(v2coord(start_vertex + i * inside_vertex, s / 2, length))
        vertexes.append(v2coord(start_vertex, s / 2, length))
        return vertexes

    def draw(self, attributions):
        fill_color = int(attributions['Color'] * 255)
        zoom_s = self.s * 2
        white = (255, 255, 255)
        black = (0, 0, 0)
        img = Image.new('RGB', (zoom_s, zoom_s), white)
        vertexes = self.get_vertex(self.n, zoom_s, attributions['Size'] * zoom_s / 2,
                                   2 * attributions['Rot'] * math.pi)
        ImageDraw.Draw(img).polygon(vertexes, outline=white, fill=(fill_color, fill_color, fill_color))
        ImageDraw.Draw(img).line(vertexes, fill=black, width=3, joint='curve')
        img = img.resize((self.s, self.s), Image.BILINEAR)
        img = img.convert('L')
        mask = np.array(img)
        return mask


def get_attributions():
    return ['Color', 'Size', 'Rot']


def get_rules():
    return {
        'Color': ['progress', 'constant'],
        'Size': ['progress', 'constant'],
        'Rot': ['constant']
    }


def get_bounds(n):
    return {
        'Color': (0.0, 1.0),
        'Size': (0.3, 0.9),
        'Rot': (-1/12, 1/12)
    }


def get_steps():
    return {
        'Color': (0.2, 0.3),
        'Size': (0.2, 0.25),
        'Rot': (0.0, 0.0)
    }


class PanelDrawer:
    def __init__(self, num_side, size_img):
        self.n = num_side
        self.s = size_img
        self.instanceDrawer = InstanceDrawer(self.n, self.s)
        self.ruleMaker = PanelRuleSampler(
            get_attributions(), get_rules(), get_bounds(self.n), get_steps()
        )

    def draw(self):
        panels, labels = self.ruleMaker.draw()
        pics = []
        for p in panels:
            pic = self.instanceDrawer.draw(p)
            pics.append(pic[None, None, ...])
        return np.concatenate(pics, axis=0), np.array(labels)

    def draw_batch(self, batch_size, index=None, fix=True):
        ref_panels, ref_labels = self.ruleMaker.draw()
        pics = np.zeros((batch_size, len(ref_panels), 1, self.s, self.s))
        for i in range(batch_size):
            panels = self.ruleMaker.draw()[0] if index is None else self.ruleMaker.redraw(ref_panels, index, fix=fix)
            for j in range(len(panels)):
                pics[i, j, 0] = self.instanceDrawer.draw(panels[j])
        x = np.tile(np.array([-1., 0., 1.]), (3, 1))[..., None]
        y = np.tile(np.array([[-1.], [0.], [1.]]), (1, 3))[..., None]
        coord = np.tile(np.concatenate([x, y], -1).reshape(9, 2), (batch_size, 1, 1))
        return pics / 255., coord


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='cache', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--num_train', default=10000, type=int)
    parser.add_argument('--num_val', default=1000, type=int)
    parser.add_argument('--num_test', default=2000, type=int)
    parser.add_argument('-s', default=64, type=int)
    parser.add_argument('-n', default=3, type=int)
    args = parser.parse_args()
    assert args.name is not None
    print('Generating {}'.format(args.name))
    panelDrawer = PanelDrawer(args.n, args.s)
    num_dataset = [args.num_train, args.num_val, args.num_test]
    base_path = os.path.join(args.output_path, 'CRPM')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    save_dir = os.path.join(base_path, args.name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    for index, phase in enumerate(['train', 'val', 'test']):
        data = np.zeros((num_dataset[index], 9, 1, args.s, args.s)).astype(np.uint8)
        data_label = np.zeros((num_dataset[index], 3)).astype(np.uint8)
        for i in range(num_dataset[index]):
            sample, label = panelDrawer.draw()
            data[i] = sample.astype(np.uint8)
            data_label[i] = label.astype(np.uint8)
            print('[%s]: %d/%d' % (phase, i + 1, num_dataset[index]), end='\r')
        with h5py.File(os.path.join(save_dir, '{}_{}.hdf5'.format(phase, args.s)), 'w') as f:
            f['panel'] = data
            f['label'] = data_label
