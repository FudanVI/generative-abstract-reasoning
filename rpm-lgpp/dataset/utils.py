import random
import copy
from abc import abstractmethod


def rand(s, e):
    return random.random() * (e - s) + s


def yes():
    return rand(0, 1) > 0.5


class RuleSingle:
    def __init__(self, bounds, attributes, steps):
        self.bounds = bounds
        self.attributes = attributes
        self.steps = steps

    @ abstractmethod
    def draw_row(self, attribute, params):
        pass

    @ abstractmethod
    def params(self, attribute, bound):
        pass

    def draw(self, attribute):
        bound = self.bounds[attribute]
        single_sample = []
        params = self.params(attribute, bound)
        for si in range(3):
            base_dict = {}
            for key, val in self.bounds.items():
                if key != attribute:
                    base_dict[key] = rand(val[0], val[1])
            row = [{**base_dict, attribute: ri} for ri in self.draw_row(attribute, params)]
            single_sample.append(row)
        if yes():
            single_sample = [[row[si] for row in single_sample] for si in range(len(single_sample[0]))]
        return [item for sublist in single_sample for item in sublist]


class RuleProgressSingle(RuleSingle):
    def __init__(self, bounds, attributes, steps):
        super(RuleProgressSingle, self).__init__(bounds, attributes, steps)

    def draw_row(self, attribute, params):
        step, inverse = params
        bound = self.bounds[attribute]
        first = rand(bound[0], bound[1] - step * 2)
        r = [first, first + step, first + 2 * step]
        if inverse:
            r = r[::-1]
        return r

    def params(self, attribute, bound):
        step = rand(self.steps[attribute][0], self.steps[attribute][1])
        inverse = yes()
        return step, inverse


class RuleMulti:
    def __init__(self, bounds, attributes, changeable_attributes, steps, alternate=True):
        self.bounds = bounds
        self.attributions = attributes
        self.changeable_attributions = changeable_attributes
        self.steps = steps
        self.alternate = alternate

    @ abstractmethod
    def draw_row(self, attribute, params):
        pass

    @ abstractmethod
    def params(self, attribute, bound):
        pass

    def draw(self, attribute):
        bound = self.bounds[attribute]
        # init panel
        panel = [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]

        def combine_attr(s, a):
            return [[{**s[i][j], **a[i][j]} for j in range(3)] for i in range(3)]

        for key, val in self.bounds.items():
            if key not in self.changeable_attributions:
                append_attr = [[{key: rand(val[0], val[1])} for _ in range(3)] for __ in range(3)]
                panel = combine_attr(panel, append_attr)
        direction = yes()
        for attrs in self.changeable_attributions:
            single_sample = []
            params = self.params(attrs, bound)
            for si in range(3):
                row = [{attrs: ri} for ri in self.draw_row(attrs, params)]
                single_sample.append(row)
            if direction:
                single_sample = [[row[si] for row in single_sample] for si in range(len(single_sample[0]))]
            panel = combine_attr(panel, single_sample)
            if self.alternate:
                direction = not direction
            else:
                direction = yes()
        return [item for sublist in panel for item in sublist]


class RuleMultiProgress(RuleMulti):
    def __init__(self, bounds, attributes, changeable_attributes, steps, alternate=True):
        super(RuleMultiProgress, self).__init__(bounds, attributes, changeable_attributes, steps, alternate=alternate)

    def draw_row(self, attribute, params):
        step, inverse = params
        bound = self.bounds[attribute]
        first = rand(bound[0], bound[1] - 2 * step)
        r = [first, first + step, first + 2 * step]
        if inverse:
            r = r[::-1]
        return r

    def params(self, attribute, bound):
        step = rand(self.steps[attribute][0], self.steps[attribute][1])
        inverse = yes()
        return step, inverse


class RuleBiProgress:
    def __init__(self, bounds, attributes, steps):
        self.bounds = bounds
        self.attributes = attributes
        self.steps = steps

    def draw(self, attribute):
        bound = self.bounds[attribute]
        min_step = (bound[1] - bound[0]) * self.steps[attribute][0]
        sample = []
        base_dict = {}
        for key, val in self.bounds.items():
            if key != attribute:
                base_dict[key] = rand(val[0], val[1])
        first = rand(bound[0], bound[1] - min_step * 4)
        step_h = rand(min_step, (bound[1] - first - 2 * min_step) / 2)
        step_v = rand(min_step, (bound[1] - first - 2 * step_h) / 2)
        for i in range(3):
            begin = first + step_v * i
            row = [
                {**base_dict, attribute: begin},
                {**base_dict, attribute: begin + step_h},
                {**base_dict, attribute: begin + step_h * 2}
            ]
            sample.append(row)
        if rand(0, 1) > 0.5:
            sample = sample[::-1]
        if rand(0, 1) > 0.5:
            sample = [t[::-1] for t in sample]
        return [item for sublist in sample for item in sublist]


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
                    # diff = abs(sample_v - pic_rule[attribute])
                    diff = self.min_diff(selection_panel, attribute, sample_v)
                    new_dict[attribute] = sample_v
            selection_panel.append(new_dict)
        x = list(enumerate(selection_panel))
        random.shuffle(x)
        indices, selection_panel = zip(*x)
        answer = indices.index(0)
        return selection_panel, answer

