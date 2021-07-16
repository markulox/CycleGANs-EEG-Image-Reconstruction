import torch
import torch.nn as nn
from model.module.wnn_layer import *
from model.module.wavelet_func import *


class GenA1(nn.Module):
    """
    EEG[CHANNEL,LEN] -> Image[3,224,224]
    We can put dropout later
    """

    def __init__(self, in_channel):
        super(GenA1, self).__init__()
        # EEG --WNN--conv1d--> latent map --dconv2d--> IMG
        self.wnn = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channel),
            MultiDimWN(in_channel=in_channel, hidden_unit=int(in_channel * 1.5), mother_wavelet=mexican_hat),
            nn.ELU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=64, padding=15),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=208, out_features=10816),
            nn.LeakyReLU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=10, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=10, out_channels=8, kernel_size=7, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=7, stride=3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.wnn(x)
        x = self.conv1(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 64, 13, 13)

        x = self.dconv1(x)
        return x

    def info(self):
        return """info"""


if __name__ == '__main__':
    sample = torch.rand([2, 16, 189])
    m = GenA1(16)
    b = m(sample)
