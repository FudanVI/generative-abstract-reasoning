import random
import copy
import numpy as np


def rand(s, e):
    return random.random() * (e - s) + s


def yes():
    return rand(0, 1) > 0.5


def sample_progress_rule(attribute, steps, bounds, transpose=False):
    step = rand(steps[0], steps[1])
    inverse = random.random() > 0.5
    single_sample = []
    for si in range(3):
        first = rand(bounds[0], bounds[1] - step * 2)
        r = [first, first + step, first + 2 * step]
        if inverse:
            r = r[::-1]
        row = [{attribute: ri} for ri in r]
        single_sample.append(row)
    if transpose:
        single_sample = [[row[si] for row in single_sample] for si in range(len(single_sample[0]))]
    return [item for sublist in single_sample for item in sublist]


def sample_constant_rule(attribute, steps, bounds, transpose=False):
    single_sample = []
    for si in range(3):
        value = rand(bounds[0], bounds[1])
        row = [{attribute: value} for _ in range(3)]
        single_sample.append(row)
    if transpose:
        single_sample = [[row[si] for row in single_sample] for si in range(len(single_sample[0]))]
    return [item for sublist in single_sample for item in sublist]


def sample_permute_rule(attribute, steps, bounds, transpose=False):
    base_values = [rand(bounds[0], bounds[1]) for _ in range(3)]
    single_sample = []
    for si in range(3):
        random.shuffle(base_values)
        row = [{attribute: val} for val in base_values]
        single_sample.append(row)
    if transpose:
        single_sample = [[row[si] for row in single_sample] for si in range(len(single_sample[0]))]
    return [item for sublist in single_sample for item in sublist]


class PanelRuleSampler:
    def __init__(self, attributes, rules, bounds, steps):
        self.attributes = attributes
        self.rules = rules
        self.bounds = bounds
        self.steps = steps

    def choose_rule_function(self, rule):
        if rule == 'progress':
            return sample_progress_rule
        elif rule == 'constant':
            return sample_constant_rule
        elif rule == 'permute':
            return sample_permute_rule
        else:
            raise NotImplementedError

    def draw(self, transpose=False):
        sample = [{} for _ in range(9)]
        label = []
        for a in self.attributes:
            idx = random.randint(0, len(self.rules[a]) - 1)
            label.append(idx)
            attr_val = self.choose_rule_function(self.rules[a][idx])(
                a, self.steps[a], self.bounds[a], transpose=transpose
            )
            sample = [{**sample[ri], **attr_val[ri]} for ri in range(len(attr_val))]
        return sample, label

    def redraw(self, sample, index, fix=True, transpose=False):
        sample = copy.deepcopy(sample)
        base_sample = self.draw(transpose=transpose)[0]
        attr = self.attributes[index]
        if fix:
            for i in range(len(base_sample)):
                base_sample[i][attr] = sample[i][attr]
            return base_sample
        else:
            for i in range(len(sample)):
                sample[i][attr] = base_sample[i][attr]
            return sample


class SelectionGenerator:
    def __init__(self, bounds, attributes):
        self.bounds = bounds
        self.attributes = attributes

    def draw_change_rules(self):
        num = random.randint(1, min(len(self.attributes), 3))
        return random.sample(self.attributes, num)

    @ staticmethod
    def min_diff(ds, a, v):
        min_diff = 1e10
        for d in ds:
            min_diff = min(min_diff, abs(d[a] - v))
        return min_diff

    def draw(self, pic_rule):
        selection_panel = [copy.deepcopy(pic_rule)]
        for i in range(7):
            attributes = self.draw_change_rules()
            new_dict = copy.deepcopy(pic_rule)
            for attribute in attributes:
                diff, max_diff = 0, self.bounds[attribute][1] - self.bounds[attribute][0]
                while diff / max_diff < 0.06:
                    sample_v = rand(self.bounds[attribute][0], self.bounds[attribute][1])
                    diff = self.min_diff(selection_panel, attribute, sample_v)
                    new_dict[attribute] = sample_v
            selection_panel.append(new_dict)
        x = list(enumerate(selection_panel))
        random.shuffle(x)
        indices, selection_panel = zip(*x)
        answer = indices.index(0)
        return selection_panel, answer


def build_panel(panel, border=2):
    assert len(panel) == 9
    h, w = panel[0].shape
    base = np.ones((h * 3 + 4 * border, w * 3 + 4 * border))
    for idx, image in enumerate(panel):
        row, col = idx // 3, idx % 3
        ys, ye = border + row * (border + h), border + row * (border + h) + h
        xs, xe = border + col * (border + w), border + col * (border + w) + w
        base[ys:ye, xs:xe] = image
    return base
