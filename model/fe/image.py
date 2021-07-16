import torch
import torch.nn as nn


class FEA1(nn.Module):
    def __init__(self):
        super(FEA1, self).__init__()

    def forward(self, x):
        return x

    def info(self):
        return """info"""
