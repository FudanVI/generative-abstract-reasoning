from abc import ABC, abstractmethod
import math
import torch.nn as nn
import torch
import lib.utils as utils


class BaseNetwork(ABC):
    def __init__(self, args, global_step=0):
        super(BaseNetwork, self).__init__()
        self.global_step = global_step
        self.anneal_settings = args.anneal if hasattr(args, 'anneal') else {}
        self.anneal()

    def anneal(self):
        for k, v in self.anneal_settings.items():
            setattr(self, k, utils.anneal_by_milestones(
                self.global_step, v['milestones'], v['value'], mode=v['mode']
            ))

    def update_hook(self):
        pass

    def update_step(self):
        self.global_step += 1
        if self.update_hook is not None:
            self.update_hook()
        self.anneal()
        for k in dir(self):
            v = getattr(self, k)
            if isinstance(v, BaseNetwork):
                v.update_step()


class ReshapeBlock(nn.Module):
    def __init__(self, size):
        super(ReshapeBlock, self).__init__()
        self.size = size

    def forward(self, x):
        shape = x.size()[:1] + tuple(self.size)
        return x.reshape(shape)


class FCResBlock(nn.Module):
    def __init__(self, hidden_size):
        super(FCResBlock, self).__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, x):
        dim = x.size(-1)
        assert dim == self.hidden_size
        h = x.reshape(-1, dim)
        h = h + self.net(h)
        return h.reshape(x.size())
