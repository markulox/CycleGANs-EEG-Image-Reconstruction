import torch
import torch.nn as nn
from model.module.wnn_layer import *
from model.module.wavelet_func import *


class GenA1(nn.Module):
    """
    Image[3,224,224] -> EEG[CHANNEL,LEN]
    len = 189
    """

    def __init__(self, out_channel):
        super(GenA1, self).__init__()
        # IMG --conv--> Latent map --deconv1D--WNN--> EEG
        # Try with no latent distance regularization first
        self.out_ch = out_channel

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=5)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=out_channel, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=2704, out_features=832),
            nn.ELU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channel, out_channels=out_channel, kernel_size=11),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=out_channel, out_channels=out_channel, kernel_size=19),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=out_channel, out_channels=out_channel, kernel_size=28),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=out_channel, out_channels=out_channel, kernel_size=42, dilation=2),
            nn.ELU()
        )

        self.wnn = FlexMultiDimWN(in_channel=out_channel, hidden_unit=out_channel, mother_wavelet=mexican_hat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.out_ch, -1)
        x = self.dconv1(x)
        x = self.wnn(x)
        return x

    def info(self):
        return """info"""


# a = torch.rand(1, 3, 224, 224)
# m = GenA1(out_channel=16)
# b = m(a)
