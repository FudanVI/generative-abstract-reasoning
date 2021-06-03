import numpy as np
from PIL import Image, ImageDraw
import argparse
import torch
import random
import os
from utils import RuleMultiProgress, RuleProgressSingle, SelectionGenerator


seed = 0
random.seed(seed)


class InstanceDrawerContinue:
    def __init__(self, size_img):
        self.size = 0.5
        self.color = 0.4
        self.s = size_img

    def draw(self, attributions):
        fill_color = (int(self.color * 255), int(self.color * 255), int(self.color * 255))
        zoom_s = self.s * 2
        position = (int(attributions['xpos'] * zoom_s), int(attributions['ypos'] * zoom_s))
        white = (255, 255, 255)
        black = (0, 0, 0)
        img = Image.new('RGB', (zoom_s, zoom_s), white)
        draw = ImageDraw.Draw(img)
        r = zoom_s * self.size / 2
        draw_region = (position[0] - r, position[1] - r, position[0] + r, position[1] + r)
        draw.pieslice(draw_region, attributions['start'], attributions['start'] + attributions['theta'], fill=fill_color, outline=black)
        draw.pieslice(draw_region, attributions['start'] + attributions['theta'], attributions['start'] + 360, outline=black)
        # draw.pieslice(draw_region, 0, 360, outline=black, width=4)
        img = img.resize((self.s, self.s), Image.BILINEAR)
        img = img.convert('L')
        mask = np.array(img)
        return mask


def get_attributions():
    return ['start', 'theta', 'xpos', 'ypos']


def get_changeable_attributions():
    return ['start', 'theta', 'xpos', 'ypos']


def get_bounds():
    return {
        'start': (0, 360),
        'theta': (30, 120),
        'xpos': (0.35, 0.65),
        'ypos': (0.35, 0.65)
    }


def get_steps():
    return {
        'start': (30, 60),
        'theta': (30, 60),
        'xpos': (0.1, 0.15),
        'ypos': (0.1, 0.15)
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
        self.r = []
        self.selection = selection
        for rule in rules:
            if rule == 'progress':
                self.r.append(RuleProgressSingle(get_bounds(), get_attributions(), get_steps()))
            elif rule == 'progress_multi':
                self.r.append(RuleMultiProgress(get_bounds(), get_attributions(),
                                                get_changeable_attributions(), get_steps()))
        self.instanceDrawer = InstanceDrawerContinue(self.s)
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
    parser.add_argument('-r', default=['progress_multi', 'progress'], type=list)
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
