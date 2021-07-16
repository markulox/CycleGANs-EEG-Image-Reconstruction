from ..module.layer import *
from ..module.wavelet_func import *
from torch.nn import Module
import torch.nn as nn


class DisA1(Module):
    """
    Consist of Multidimensional MultiDimWN layers (Not flexible one)
    """

    def __init__(self, in_channel, in_len):
        super(DisA1, self).__init__()
        self.in_len = in_len
        # Wavelet Block
        self.block01 = nn.Sequential(
            MultiDimWN(in_channel, 20, mexican_hat),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=64)
        )

        # Linear block
        self.lin_f = in_channel * in_len
        self.block02 = nn.Sequential(
            nn.Linear(in_features=self.lin_f, out_features=self.lin_f / 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.lin_f / 2, out_features=self.lin_f / 3),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.lin_f / 3, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block01(x)
        x = x.reshape(x.shape[0], -1)  # Flatten input tensor
        x = self.block02(x)
        return x

    def info(self):
        return """EEG Discriminator Arch 1 - {\n
        \tMultiDimWN(20,Mexican Hat)\n
        \tLinear(%d,%d),LeakyReLU\n
        \tLinear(%d,%d),LeakyReLU\n
        \tLinear(%d,%d),Sigmoid\n
        }""" % (self.lin_f, self.lin_f / 2, self.lin_f / 2, self.lin_f / 3, self.lin_f / 3, 1)


class DisA2(Module):
    """
    Replace with FlexMultiDimWN
    """

    def __init__(self, in_channel, in_len):
        super(DisA2, self).__init__()
        self.in_len = in_len
        # Wavelet Block
        self.block01 = nn.Sequential(
            FlexMultiDimWN(in_channel, in_channel / 2, mexican_hat),
            FlexMultiDimWN(in_channel / 2, in_channel / 3, mexican_hat),
            FlexMultiDimWN(in_channel / 3, 1, mexican_hat)
        )

        # Linear block
        self.lin_f = in_channel * in_len
        self.block02 = nn.Sequential(
            nn.Linear(in_features=self.lin_f, out_features=self.lin_f / 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.lin_f / 2, out_features=self.lin_f / 3),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.lin_f / 3, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block01(x)
        x = x.reshape(x.shape[0], -1)  # Flatten input tensor
        x = self.block02(x)
        return x

    def info(self):
        return """EEG Discriminator Arch 1 - {\n
        \tFlexMultiDimWN(in_ch, in_ch/2, Mexican Hat)\n
        \tFlexMultiDimWN(in_ch/2, in_ch/3, Mexican Hat)\n
        \tFlexMultiDimWN(in_ch/3, 1, Mexican Hat)\n
        \tLinear(%d,%d),LeakyReLU\n
        \tLinear(%d,%d),LeakyReLU\n
        \tLinear(%d,%d),Sigmoid\n
        }""" % (self.lin_f, self.lin_f / 2, self.lin_f / 2, self.lin_f / 3, self.lin_f / 3, 1)


if __name__ == '__main__':
    pass