import torch
from torch import nn as nn
from torch.nn import Module
from model.module.wnn_layer import *
from model.module.layers import ConvBlockInv4


class EEG_LT2IMG(Module):
    """
    Generate an image from EEG signal
    """

    def __init__(self):
        super(EEG_LT2IMG, self).__init__()

    def forward(self, x):
        return x


class D_IMG(Module):
    """
    Discriminate the generated image and real image
    """

    def __init__(self):
        super(D_IMG, self).__init__()

    def forward(self, x):
        return x


class IMG2EEG_LT(Module):
    """
    Generate an EEG signal from image
    """

    def __init__(self, latent_size):
        """
        Map image domain to EEG latent.
        :param latent_size: Specify output latent size
        """
        super(IMG2EEG_LT, self).__init__()
        self.conv_block01 = nn.Sequential(
            ConvBlockInv4(stride_tuple=(2, 1, 1), channel_tuple=(3, 16, 32)),
            ConvBlockInv4(stride_tuple=(2, 1, 1), channel_tuple=(32, 64, 128)),
            ConvBlockInv4(stride_tuple=(2, 1, 1), channel_tuple=(128, 256, 512)),
            ConvBlockInv4(stride_tuple=(1, 1, 1), channel_tuple=(512, 1024, 2048))
        )

        self.fc = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1024, latent_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x: An stimuli image, expected shape [3,128,128]
        :return:
        """
        x = self.conv_block01(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class D_EEG_LT(Module):
    """
    Discriminate generated EEG signal and the real one
    """

    def __init__(self):
        super(D_EEG_LT, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    BS = 2
    sample = torch.rand(BS, 3, 128, 128)
    model = IMG2EEG_LT(latent_size=100)
    out = model(sample)
    print(out.shape)
