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


class InstanceDrawerComplexCircle:
    def __init__(self, size_img):
        self.size_out = 0.7
        self.size_in = 0.3
        self.color_out = 0
        self.s = size_img

    @staticmethod
    def center_radius_to_bbox(x, y, r):
        return x - r, y - r, x + r, y + r

    @staticmethod
    def degree_to_xy(x, y, r, a):
        return x + r * math.cos(a / 180 * math.pi), y + r * math.sin(a / 180 * math.pi)

    def draw(self, attributions):
        color_out = int(self.color_out * 255)
        color_inner = int(attributions['Inner\nColor'] * 255)
        zoom_s = self.s * 2
        white = (255, 255, 255)
        fill_color_out = (color_out, color_out, color_out)
        fill_color_inner = (color_inner, color_inner, color_inner)
        img = Image.new('RGB', (zoom_s, zoom_s), white)
        draw = ImageDraw.Draw(img)
        xc, yc = zoom_s / 2, zoom_s / 2
        radius_out = zoom_s * self.size_out / 2
        radius_in = zoom_s * self.size_in / 2

        draw_pad = zoom_s * (1 - self.size_out) / 2
        draw_region = (draw_pad, draw_pad, zoom_s - draw_pad, zoom_s - draw_pad)
        draw.pieslice(
            draw_region, attributions['Sector\nPos'], attributions['Sector\nPos'] + attributions['Sector\nSize'],
            fill=fill_color_out, outline=fill_color_out
        )
        draw.arc(draw_region, 0, 360, fill='black', width=3)
        draw.line(
            [(xc, yc), self.degree_to_xy(xc, yc, radius_out, attributions['Sector\nPos'])],
            fill='black', width=3
        )
        draw.line(
            [(xc, yc), self.degree_to_xy(xc, yc, radius_out, attributions['Sector\nPos'] + attributions['Sector\nSize'])],
            fill='black', width=3
        )
        draw.ellipse(
            self.center_radius_to_bbox(zoom_s / 2, zoom_s / 2, radius_in),
            fill=fill_color_inner, outline='black', width=3
        )

        img = img.resize((self.s, self.s), Image.BILINEAR)
        img = img.convert('L')
        mask = np.array(img)
        return mask


def get_attributions():
    return ['Inner\nColor', 'Sector\nPos', 'Sector\nSize']


def get_rules():
    return {
        'Inner\nColor': ['progress', 'constant'],
        'Sector\nPos': ['progress', 'constant'],
        'Sector\nSize': ['progress', 'constant']
    }


def get_bounds():
    return {
        'Inner\nColor': (0.2, 0.8),
        'Sector\nPos': (0, 360),
        'Sector\nSize': (30, 120)
    }


def get_steps():
    return {
        'Inner\nColor': (0.2, 0.3),
        'Sector\nPos': (30, 60),
        'Sector\nSize': (30, 60)
    }


class PanelDrawer:
    def __init__(self, size_img):
        self.s = size_img
        self.instanceDrawer = InstanceDrawerComplexCircle(self.s)
        self.ruleMaker = PanelRuleSampler(
            get_attributions(), get_rules(), get_bounds(), get_steps()
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
    args = parser.parse_args()
    assert args.name is not None
    print('Generating {}'.format(args.name))
    panelDrawer = PanelDrawer(args.s)
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
