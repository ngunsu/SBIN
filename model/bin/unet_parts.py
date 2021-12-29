""" Parts of the U-Net model
Based on: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
from model.bin.birealnet import BinaryActivation, HardBinaryConv


class BasicBlock(nn.Module):
    """(convolution => [BN] ==> activation ) """

    def __init__(self, in_channels, out_channels, ks, stride=1, pad=1,
                 bn=True, activation='', binary=False, residual='False',
                 bias=False):
        super().__init__()
        modules = []
        self.residual = residual
        self.residual_down = None
        self.binary = binary
        self.stride = stride

        if bn and not binary:
            modules.append(nn.BatchNorm2d(in_channels))

        if activation.lower() == 'relu':
            modules.append(nn.ReLU(inplace=True))

        if binary:
            modules.append(BinaryActivation())
            modules.append(HardBinaryConv(in_channels,
                                          out_channels,
                                          kernel_size=ks,
                                          padding=pad,
                                          stride=stride))
        else:
            modules.append(nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=ks,
                                     padding=pad,
                                     stride=stride,
                                     bias=bias))
        if bn and binary:
            modules.append(nn.BatchNorm2d(out_channels))

        if self.residual:
            self.residual_down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                                               nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        residual = x

        out = self.conv(x)
        if self.residual_down is not None:
            residual = self.residual_down(residual)
            out += residual

        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bn=True, n=1, binary=False, activation='', residual=False):
        super().__init__()
        modules = []
        modules.append(nn.MaxPool2d(2))
        for i in range(n):
            modules.append(BasicBlock(in_channels, out_channels, ks=3, bn=bn,
                                      activation=activation, binary=binary, residual=residual))
            in_channels = out_channels
        self.maxpool_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bn=True, bilinear=True,
                 n=2, binary=False, activation='', residual=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        modules = []
        for i in range(n):
            modules.append(BasicBlock(in_channels, out_channels, ks=3, bn=bn,
                                      activation=activation, binary=binary, residual=residual))
            in_channels = out_channels
        self.conv = nn.Sequential(*modules)

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
        return self.conv(torch.cat([x1, x2], 1))
