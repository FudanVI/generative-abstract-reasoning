import torch.nn as nn
from lib.building_block import ReshapeBlock


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            ReshapeBlock([512]),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, x):
        return self.net(x)
