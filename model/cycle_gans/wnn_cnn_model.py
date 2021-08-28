import torch
import torch.nn as nn
from model.module.wnn_layer import *
from model.module.wavelet_func import *


class EEG2IMG(nn.Module):
    """
    EEG[CHANNEL,LEN] -> Image[3,224,224]
    We can put dropout later
    """

    def __init__(self, in_channel):
        super(EEG2IMG, self).__init__()
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

        # TODO: Calculate the size of deconvolution
        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=10, kernel_size=5),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=10, out_channels=8, kernel_size=7, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=7, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=6, stride=2, padding=5, ),
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


class IMG2EEG(nn.Module):
    """
    Image[3,224,224] -> EEG[CHANNEL,LEN]
    len = 189
    """

    def __init__(self, out_channel):
        super(IMG2EEG, self).__init__()
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


class D_EEG1(nn.Module):
    """
    Consist of Multidimensional MultiDimWN layers (Not flexible one)
    """

    def __init__(self, in_channel, in_len):
        super(D_EEG1, self).__init__()
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


class D_EEG2(nn.Module):
    """
    Replace with FlexMultiDimWN
    """

    def __init__(self, in_channel, in_len):
        super(D_EEG2, self).__init__()
        self.in_len = in_len
        # Wavelet Block
        self.block01 = nn.Sequential(
            FlexMultiDimWN(in_channel, in_channel // 2, mexican_hat),
            FlexMultiDimWN(in_channel // 2, in_channel // 3, mexican_hat),
            FlexMultiDimWN(in_channel // 3, 1, mexican_hat)
        )

        # Linear block
        self.lin_f = 1 * in_len
        self.block02 = nn.Sequential(
            nn.Linear(in_features=self.lin_f, out_features=self.lin_f // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.lin_f // 2, out_features=self.lin_f // 3),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.lin_f // 3, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block01(x)
        x = x.reshape(x.shape[0], -1)  # Flatten input tensor
        x = self.block02(x)
        return x

    def info(self):
        return """EEG Discriminator Arch 1 - {\n
        \tFlexMultiDimWN(in_ch, in_ch//2, Mexican Hat)\n
        \tFlexMultiDimWN(in_ch//2, in_ch//3, Mexican Hat)\n
        \tFlexMultiDimWN(in_ch//3, 1, Mexican Hat)\n
        \tLinear(%d,%d),LeakyReLU\n
        \tLinear(%d,%d),LeakyReLU\n
        \tLinear(%d,%d),Sigmoid\n
        }""" % (self.lin_f, self.lin_f // 2, self.lin_f // 2, self.lin_f // 3, self.lin_f // 3, 1)


class D_IMG1(nn.Module):
    """
    Image[3,224,224] -> Label
    Current expected image size = 3x224x224
    """

    def __init__(self):
        super(D_IMG1, self).__init__()
        self.conv1 = nn.Sequential(  # Currently we input black and white img
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, kernel_size=3, padding_mode='replicate'),
            nn.LeakyReLU()
        )
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=46),
            nn.LeakyReLU(),
            nn.Linear(in_features=46, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  # 64x110x110
        x = self.conv2(x)  # 128x53x53
        x = self.conv3(x)  # 256x25x25
        x = self.conv4(x)  # 512x11x11
        x = x.reshape((x.shape[0], 512))  #
        x = self.final_fc(x)
        return x


if __name__ == '__main__':
    # Here will be testing forwarding model
    pass
