import torch.nn as nn
from lib.building_block import ReshapeBlock


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            ReshapeBlock([self.input_dim, 1, 1]),
            nn.ConvTranspose2d(self.input_dim, 128, (1, 1), stride=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_dim, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.net(z)
        return x


class DeepConvDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            ReshapeBlock([self.input_dim, 1, 1]),
            nn.ConvTranspose2d(self.input_dim, 512, (1, 1), stride=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_dim, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.net(z)
        return x
