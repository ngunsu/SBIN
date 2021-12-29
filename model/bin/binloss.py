import torch
from torch import nn
from model.bin.birealnet import HardBinaryConv


class BinLoss(nn.Module):

    def __init__(self, gamma=5e-7):
        super().__init__()
        self.gamma = gamma

    def forward(self, model):
        first = True
        for m in model.modules():
            if isinstance(m, HardBinaryConv):
                if first:
                    out = torch.pow(torch.sum(1 - torch.abs(m.weights)), 2)
                    first = False
                else:
                    out = out + torch.pow(torch.sum(1 - torch.abs(m.weights)), 2)
        return self.gamma * out
