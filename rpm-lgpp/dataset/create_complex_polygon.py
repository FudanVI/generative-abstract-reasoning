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


class InstanceDrawerComplexPolygon:
    def __init__(self, num_side_in, num_side_out, color_out, size_in, size_img):
        self.n_in = num_side_in
        self.n_out = num_side_out
        self.color_out = color_out
        self.size_in = size_in
        self.rotation_out = 0
        self.s = size_img

    @staticmethod
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
        fill_color_in = int(attributions['color_in'] * 255)
        fill_color_out = int(self.color_out * 255)
        zoom_s = self.s * 2
        white = (255, 255, 255)
        black = (0, 0, 0)
        img = Image.new('RGB', (zoom_s, zoom_s), white)
        vertexes_out = self.get_vertex(self.n_out, zoom_s, attributions['size_out'] * zoom_s / 2,
                                       2 * self.rotation_out * math.pi)
        ImageDraw.Draw(img).polygon(vertexes_out, outline=white, fill=(fill_color_out, fill_color_out, fill_color_out))
        ImageDraw.Draw(img).line(vertexes_out, fill=black, width=3, joint='curve')
        vertexes_in = self.get_vertex(self.n_out, zoom_s, self.size_in * zoom_s / 2,
                                      2 * attributions['rotation_in'] * math.pi)
        ImageDraw.Draw(img).polygon(vertexes_in, outline=white, fill=(fill_color_in, fill_color_in, fill_color_in))
        ImageDraw.Draw(img).line(vertexes_in, fill=black, width=3, joint='curve')
        img = img.resize((self.s, self.s), Image.BILINEAR)
        img = img.convert('L')
        mask = np.array(img)
        return mask


def get_attributions():
    return ['color_in', 'size_out', 'rotation_in']


def get_changeable_attributions():
    return ['color_in', 'size_out']


def get_bounds(n):
    return {
        'color_in': (0.2, 0.8),
        'size_out': (0.6, 0.9),
        'rotation_in': (0.0, 1 / n)
    }


def get_steps():
    return {
        'color_in': (0.2, 0.3),
        'size_out': (0.1, 0.15)
    }


class RuleMaker:
    def __init__(self, rules, num_side):
        self.rules = rules
        self.n = num_side
        self.bounds = get_bounds(self.n)
        self.attributions = get_changeable_attributions()

    def draw(self):
        rand_rule = random.choice(self.rules)
        rand_attribution = random.choice(self.attributions)
        return rand_rule.draw(rand_attribution)


class PanelDrawer:
    def __init__(self, num_side, size_img, color_out, size_in, rules, selection=False):
        self.n = num_side
        self.s = size_img
        self.c = color_out
        self.sz = size_in
        self.selection = selection
        self.r = []
        for rule in rules:
            if rule == 'progress':
                self.r.append(RuleProgressSingle(get_bounds(self.n), get_attributions(), get_steps()))
        self.instanceDrawer = InstanceDrawerComplexPolygon(self.n, self.n, self.c, self.sz, self.s)
        self.ruleMaker = RuleMaker(self.r, self.n)
        self.selectionGenerator = SelectionGenerator(get_bounds(self.n), get_attributions())

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
    parser.add_argument('-n', default=3, type=int)
    parser.add_argument('-r', default=['progress'], type=list)
    args = parser.parse_args()
    assert args.name is not None
    panelDrawer = PanelDrawer(args.n, args.s, 1.0, 0.3, args.r, selection=args.selection == 'True')
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
