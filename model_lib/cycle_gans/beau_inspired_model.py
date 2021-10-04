import torch
from torch import nn as nn
from torch.nn import Module
from model.module.wnn_layer import *

"""
These models will be implemented using fully connected layers only since my friends model which using only FC seems to 
work for somehow.
"""


class EEG2IMG(Module):
    """
    Generate an image from EEG signal. Should we do a bottle neck design? May be yes
    """

    def __init__(self, in_channel, latent_size=100, dropout_p=0.5):
        """
        Expect input EEG size [16,189] output Image size [3,224,224]
        :param latent_size: We use default = 100 same as EEGChannelNet paper
        """
        super(EEG2IMG, self).__init__()
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=189 // 2),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=189 // 6),
            nn.BatchNorm1d(num_features=64),
            nn.ELU()
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(4224, 130),
            nn.Dropout(p=dropout_p),
            nn.ELU(),
            nn.Linear(130, latent_size),
            nn.ELU()
        )

        self.img_recon = nn.Sequential(
            nn.Linear(latent_size, (latent_size // 2) ** 2),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear((latent_size // 2) ** 2, 150528),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.eeg_conv(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        latent = self.fc_enc(x)
        x = self.img_recon(latent)
        x = x.reshape(x.shape[0], 3, 224, -1)
        return x


class D_IMG(Module):
    """
    Discriminate the generated image and real image. No need bottle neck
    """

    def __init__(self, dropout_p=0.5):
        """
        Expected image input size [3,224,224]
        """
        super(D_IMG, self).__init__()
        self.img_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2, dilation=2),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, dilation=2),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, dilation=2),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1),
            nn.ReLU(),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(6144, 157),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(157, 25),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.img_enc(x)
        x = x.flatten(start_dim=1)
        x = self.fc_block(x)
        return x


class IMG2EEG(Module):
    """
    Generate an EEG signal from image. Let's do bottle neck
    """

    def __init__(self, out_channel, latent_size=100, dropout_p=0.5):
        super(IMG2EEG, self).__init__()

        self.img_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2, dilation=2),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, dilation=2),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, dilation=2),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1),
            nn.ReLU(),
        )

        self.fc_latent = nn.Sequential(
            nn.Linear(6144, latent_size),
            nn.Dropout(p=dropout_p),
            nn.ReLU()
        )

        self.fc_reshape = nn.Sequential(
            nn.Linear(latent_size, 130),
            nn.Dropout(p=dropout_p),
            nn.ELU(),
            nn.Linear(130, 4224),
            nn.ELU()
        )

        self.img_recon = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=189 // 6),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=out_channel, kernel_size=189 // 2),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ELU()
        )

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = self.img_enc(x)
        x = x.flatten(start_dim=1)
        latent = self.fc_latent(x)
        x = self.fc_reshape(latent)
        x = x.reshape(x.shape[0], 64, -1)
        x = self.img_recon(x)
        return x


class D_EEG(Module):
    """
    Discriminate generated EEG signal and the real one
    """

    def __init__(self, in_channel, in_len, dropout_p=0.5):
        super(D_EEG, self).__init__()
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=189 // 2),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=189 // 6),
            nn.BatchNorm1d(num_features=64),
            nn.ELU()
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(4224, 130),
            nn.Dropout(p=dropout_p),
            nn.ELU(),
            nn.Linear(130, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.eeg_conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc_enc(x)
        return x


if __name__ == "__main__":
    BS = 2
    sample = torch.rand(BS, 3, 224, 224)
    md = D_IMG()
    out = md(sample)
