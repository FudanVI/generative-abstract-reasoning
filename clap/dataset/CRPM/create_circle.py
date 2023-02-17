import numpy as np
from PIL import Image, ImageDraw
import argparse
import math
import random
import os
import shutil
import h5py
# from dataset.CRPM.utils import PanelRuleSampler, build_panel
from utils import PanelRuleSampler, build_panel


seed = 0
random.seed(seed)


class InstanceDrawerContinue:
    def __init__(self, size_img):
        self.size = 0.7
        self.color = 0.4
        self.s = size_img

    @staticmethod
    def degree_to_xy(x, y, r, a):
        return x + r * math.cos(a / 180 * math.pi), y + r * math.sin(a / 180 * math.pi)

    def draw(self, attributions):
        fill_color = (int(self.color * 255), int(self.color * 255), int(self.color * 255))
        zoom_s = self.s * 2
        white = (255, 255, 255)
        img = Image.new('RGB', (zoom_s, zoom_s), white)
        draw = ImageDraw.Draw(img)
        draw_pad = zoom_s * (1 - self.size) / 2
        draw_region = (draw_pad, draw_pad, zoom_s - draw_pad, zoom_s - draw_pad)
        xc, yc = zoom_s / 2, zoom_s / 2
        radius = zoom_s * self.size / 2

        draw.pieslice(
            draw_region, attributions['Sector\nPos'], attributions['Sector\nPos'] + attributions['Sector\nSize'],
            fill=fill_color, outline=fill_color
        )
        draw.arc(draw_region, 0, 360, fill='black', width=3)
        draw.line(
            [(xc, yc), self.degree_to_xy(xc, yc, radius, attributions['Sector\nPos'])],
            fill='black', width=3
        )
        draw.line(
            [(xc, yc), self.degree_to_xy(xc, yc, radius, attributions['Sector\nPos'] + attributions['Sector\nSize'])],
            fill='black', width=3
        )
        img = img.resize((self.s, self.s), Image.BILINEAR)
        img = img.convert('L')
        mask = np.array(img)
        return mask


def get_attributions():
    return ['Sector\nPos', 'Sector\nSize']


def get_rules():
    return {
        'Sector\nPos': ['progress', 'constant'],
        'Sector\nSize': ['progress', 'constant']
    }


def get_bounds():
    return {
        'Sector\nPos': (0, 360),
        'Sector\nSize': (30, 120)
    }


def get_steps():
    return {
        'Sector\nPos': (30, 60),
        'Sector\nSize': (30, 60)
    }


class PanelDrawer:
    def __init__(self, size_img):
        self.s = size_img
        self.instanceDrawer = InstanceDrawerContinue(self.s)
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
        data_label = np.zeros((num_dataset[index], 2)).astype(np.uint8)
        for i in range(num_dataset[index]):
            sample, label = panelDrawer.draw()
            data[i] = sample.astype(np.uint8)
            data_label[i] = label.astype(np.uint8)
            print('[%s]: %d/%d' % (phase, i + 1, num_dataset[index]), end='\r')
        with h5py.File(os.path.join(save_dir, '{}_{}.hdf5'.format(phase, args.s)), 'w') as f:
            f['panel'] = data
            f['label'] = data_label
