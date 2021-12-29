import torch
import torch.nn as nn
from model.bin.birealnet import BinaryActivation, HardBinaryConv


def get_activation(activation='relu'):
    activation = activation.lower()
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'prelu':
        return nn.PReLU()


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, stride=1, pad=1,
                 bn=True, activation='', binary=False):

        super().__init__()

        self.activation = BinaryActivation(binary=binary)
        self.conv = HardBinaryConv(in_channels,
                                   out_channels,
                                   kernel_size=ks,
                                   padding=pad,
                                   stride=stride,
                                   binary=binary)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.nonlinear = get_activation(activation) if activation != '' else None

    def forward(self, x):
        x = self.activation(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.nonlinear is not None:
            x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, stride=1, pad=1,
                 bn=True, activation='', binary=False, down_sample=None):

        super().__init__()
        self.activation = BinaryActivation(binary=binary)
        self.conv = HardBinaryConv(in_channels,
                                   out_channels,
                                   kernel_size=ks,
                                   padding=pad,
                                   stride=stride,
                                   binary=binary)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.nonlinear = get_activation(activation) if activation != '' else None
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.activation(x)
        out = self.conv(out)
        if self.bn is not None:
            out = self.bn(out)
        if self.nonlinear is not None:
            out = self.nonlinear(out)
        if self.down_sample is not None:
            residual = self.down_sample(residual)
        out += residual
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, stride=1, pad=1,
                 bn=True, activation='', n=1, binary=False, last_float=False):

        super().__init__()

        modules = []
        for i in range(n):
            down_sample = None
            if (in_channels != out_channels or stride != 1) and i == 0:
                down_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                                            nn.BatchNorm2d(out_channels))
            if i == n - 1 and last_float:
                binary = False
            modules.append(BasicResBlock(in_channels,
                                         out_channels,
                                         ks,
                                         stride,
                                         pad,
                                         bn,
                                         activation,
                                         binary,
                                         down_sample))
            in_channels = out_channels
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        return self.convs(x)


class NormalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, stride=1, pad=1,
                 bn=True, activation='', n=1, binary=False, last_float=False):

        super().__init__()

        modules = []
        for i in range(n):
            if i == n - 1 and last_float:
                binary = False
            modules.append(BasicBlock(in_channels,
                                      out_channels,
                                      ks,
                                      stride,
                                      pad,
                                      bn,
                                      activation,
                                      binary))
            in_channels = out_channels
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        return self.convs(x)


class Down(nn.Module):
    """Downscaling with maxpool then convs"""

    def __init__(self, in_channels, out_channels, bn=True, n=1,
                 binary=False, activation='', residual=False, pool='max'):
        super().__init__()
        block = NormalBlock
        if residual:
            block = ResBlock

        self.maxpool = nn.MaxPool2d(2)
        if pool == 'avg':
            self.maxpool = nn.AvgPool2d(2)

        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(in_channels)
        self.activation = '' if activation == '' else get_activation(activation)
        self.block = block(in_channels, out_channels, ks=3, bn=bn, n=n,
                           activation=activation, binary=binary)

    def forward(self, x):
        out = self.maxpool(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation != '':
            out = self.activation(out)
        out = self.block.forward(out)
        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bn=True,
                 n=1, binary=False, activation='', residual=False, last_float=False):
        super().__init__()
        block = NormalBlock
        if residual:
            block = ResBlock
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(in_channels)
        self.activation = '' if activation == '' else get_activation(activation)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = block(in_channels, out_channels, ks=3, bn=bn,
                           activation=activation, n=n, binary=binary, last_float=last_float)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        '''
        buttom, right = x1.size(2) % 2, x1.size(3) % 2
        x2 = nn.functional.pad(x2, (0, -right, 0, -buttom))
        '''
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [int(diffX / 2), diffX - int(diffX / 2),
                                    int(diffY / 2), diffY - int(diffY / 2)])
        out = torch.cat([x1, x2], 1)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation != '':
            out = self.activation(out)
        return self.block(out)
