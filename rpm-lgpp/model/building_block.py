import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 4)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv_z = nn.Conv2d(256, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class AxisEncoder(nn.Module):
    def __init__(self, dim_size, args):
        super(AxisEncoder, self).__init__()
        self.z_dim = dim_size
        self.axis_dim = args.axis_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 4)
        self.bn5 = nn.BatchNorm2d(64)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

        self.axis_nets_x = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 6 * self.axis_dim * self.z_dim)
        )
        self.axis_nets_y = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 6 * self.axis_dim * self.z_dim)
        )

    def forward(self, x):
        bs = x.size(0)
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        h = h.view(bs // 8, 8 * 64)
        axis_list_x = self.axis_nets_x(h).view(bs // 8, self.z_dim, 3 * self.axis_dim, 2)
        axis_list_y = self.axis_nets_y(h).view(bs // 8, self.z_dim, 3 * self.axis_dim, 2)
        return axis_list_x, axis_list_y


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 256, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 64, 4, 1, 0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        self.act_final = nn.Sigmoid()

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.act_final(self.conv_final(h))
        return mu_img
