# Base on code from https://github.com/liuzechun/Bi-Real-net
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BinaryActivation(nn.Module):
    def __init__(self, binary=True):
        super().__init__()
        self.binary = binary

    def forward(self, x):
        if self.binary:
            out_forward = torch.sign(x)
            mask1 = x < -1
            mask2 = x < 0
            mask3 = x < 1
            out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            # out3 is for the gradient approximation, out are the binary values (trick)
            out = out_forward.detach() - out3.detach() + out3
            return out
        else:
            return x


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, binary=True):
        super().__init__()
        self.stride = stride
        self.binary = binary
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)
        if not binary:
            n = kernel_size * kernel_size * out_chn
            self.weights.data.normal_(0, math.sqrt(2. / n))
        self.binactive = BinaryActivation(binary=binary)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        if self.binary:
            '''
            abs_mean = torch.mean(abs(real_weights), dim=3, keepdim=True)
            scaling_factor = torch.mean(torch.mean(abs_mean, dim=2, keepdim=True), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            # clipped_weights  for the gradient approximation, binary_weights the binary weights (trick)
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
            '''
            binary_weights = self.binactive(real_weights)
            y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
            return y
        else:
            y = F.conv2d(x, real_weights, stride=self.stride, padding=self.padding)
            return y
