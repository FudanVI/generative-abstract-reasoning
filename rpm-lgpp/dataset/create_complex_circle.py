import numpy as np
from PIL import Image, ImageDraw
import argparse
import torch
import math
import random
import os
from utils import RuleProgressSingle, SelectionGenerator


seed = 0
random.seed(seed)


class InstanceDrawerComplexCircle:
    def __init__(self, size_img):
        self.size_out = 0.7
        self.size_in = 0.3
        self.color_out = 0
        self.s = size_img

    def draw(self, attributions):
        color_out = int(self.color_out * 255)
        color_inner = int(attributions['color_inner'] * 255)
        zoom_s = self.s * 2
        white = (255, 255, 255)
        fill_color_out = (color_out, color_out, color_out)
        fill_color_inner = (color_inner, color_inner, color_inner)
        black = (0, 0, 0)
        img = Image.new('RGB', (zoom_s, zoom_s), white)
        draw = ImageDraw.Draw(img)

        def center_radius_to_bbox(x, y, r):
            return x - r, y - r, x + r, y + r

        draw_pad = zoom_s * (1 - self.size_out) / 2
        draw_region = (draw_pad, draw_pad, zoom_s - draw_pad, zoom_s - draw_pad)
        draw.pieslice(draw_region, attributions['start'], attributions['start'] + attributions['theta'],
                      fill=fill_color_out, outline=black)
        draw.pieslice(draw_region, attributions['start'] + attributions['theta'], attributions['start'] + 360,
                      outline=black)
        draw.ellipse(center_radius_to_bbox(zoom_s / 2, zoom_s / 2, zoom_s * self.size_in / 2),
                     fill=fill_color_inner, outline='black')

        img = img.resize((self.s, self.s), Image.BILINEAR)
        img = img.convert('L')
        mask = np.array(img)
        return mask


def get_attributions():
    return ['color_inner', 'start', 'theta']


def get_changeable_attributions():
    return ['color_inner', 'start', 'theta']


def get_bounds():
    return {
        'color_inner': (0.2, 0.8),
        'start': (0, 360),
        'theta': (30, 120)
    }


def get_steps():
    return {
        'color_inner': (0.2, 0.3),
        'start': (30, 60),
        'theta': (30, 60)
    }


class RuleMaker:
    def __init__(self, rules):
        self.rules = rules
        self.bounds = get_bounds()
        self.attributions = get_changeable_attributions()

    def draw(self):
        rand_rule = random.choice(self.rules)
        rand_attribution = random.choice(self.attributions)
        return rand_rule.draw(rand_attribution)


class PanelDrawer:
    def __init__(self, size_img, rules, selection=False):
        self.s = size_img
        self.selection = selection
        self.r = []
        for rule in rules:
            if rule == 'progress':
                self.r.append(RuleProgressSingle(get_bounds(), get_attributions(), get_steps()))
        self.instanceDrawer = InstanceDrawerComplexCircle(self.s)
        self.ruleMaker = RuleMaker(self.r)
        self.selectionGenerator = SelectionGenerator(get_bounds(), get_attributions())

    def draw(self):
        rule_panel = self.ruleMaker.draw()
        pics = []
        for rule_pic in rule_panel:
            pic = self.instanceDrawer.draw(rule_pic)
            pics.append(pic[None, None, ...])
        if self.selection:
            rule_selections, answer = self.selectionGenerator.draw(rule_panel[-1])
            selection_pics = []
            for rule_selection in rule_selections:
                selection_pics.append(self.instanceDrawer.draw(rule_selection)[None, None, ...])
            return np.concatenate(pics, axis=0), np.concatenate(selection_pics, axis=0), np.array([answer])
        return np.concatenate(pics, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='cache', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--num_train', default=8000, type=int)
    parser.add_argument('--num_val', default=800, type=int)
    parser.add_argument('--num_test', default=1600, type=int)
    parser.add_argument('--selection', default='False', type=str)
    parser.add_argument('-s', default=64, type=int)
    parser.add_argument('-r', default=['progress'], type=list)
    args = parser.parse_args()
    assert args.name is not None
    panelDrawer = PanelDrawer(args.s, args.r, selection=args.selection == 'True')
    num_dataset = [args.num_train, args.num_val, args.num_test]
    for index, phase in enumerate(['train', 'val', 'test']):
        data = torch.zeros(num_dataset[index], 9, 1, args.s, args.s).float()
        data_selection = torch.zeros(num_dataset[index], 8, 1, args.s, args.s).float()
        data_answer = torch.zeros(num_dataset[index]).long()
        for i in range(num_dataset[index]):
            if args.selection == 'True':
                sample, selection, answer = panelDrawer.draw()
                data_selection[i] = torch.from_numpy(selection).float()
                data_answer[i] = torch.from_numpy(answer).long()
            else:
                sample = panelDrawer.draw()
            data[i] = torch.from_numpy(sample).float()
            print('[%s]: %d/%d' % (phase, i + 1, num_dataset[index]), end='\r')
        torch.save(data, os.path.join(args.output_path, '%s_%s_%d.pt' % (args.name, phase, args.s)))
        if args.selection == 'True':
            torch.save(data_selection, os.path.join(args.output_path, '%s_%s_%d.selection' % (args.name, phase, args.s)))
            torch.save(data_answer, os.path.join(args.output_path, '%s_%s_%d.answer' % (args.name, phase, args.s)))
