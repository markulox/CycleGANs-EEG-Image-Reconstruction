import torch
from torch import nn as nn
from model.module.container import ParallelModule
from model.module.layers import ResidualBlock, Flatten


class EEGChannelNetEncoder(nn.Module):
    """
    This encoder is implemented by follow the architecture from research paper : Decoding Brain Representations
    by Multimodal Learning of Neural Activity and Visual Features
    link : https://arxiv.org/abs/1810.10974
    """

    def __init__(self, in_channel, output_size):
        """
        Currently, this model limited to a fixed size of EEG Input data in shape of (BS, 1, 128 electrodes, 440 EEG Samples)
        :param output_size (Hyper parameter) The size of output latent vector
        """
        super(EEGChannelNetEncoder, self).__init__()
        self.temporal_block = ParallelModule(
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 1), padding=(0, 16), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 2), padding=(0, 32), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 4), padding=(0, 64), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 8), padding=(0, 128), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(1, 33), stride=(1, 2), dilation=(1, 16), padding=(0, 256), in_channels=1,
                          out_channels=10),
                nn.BatchNorm2d(num_features=10),
                nn.ReLU())
        )

        self.spatial_block = ParallelModule(
            nn.Sequential(
                nn.Conv2d(kernel_size=(in_channel, 1), stride=(2, 1), dilation=(1, 1), padding=(7, 0), in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(in_channel // 2, 1), stride=(2, 1), dilation=(1, 1), padding=(3, 0),
                          in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(in_channel // 4, 1), stride=(2, 1), dilation=(1, 1), padding=(1, 0),
                          in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(kernel_size=(in_channel // 8, 1), stride=(2, 1), dilation=(1, 1), padding=(0, 0),
                          in_channels=50,
                          out_channels=50),
                nn.BatchNorm2d(num_features=50),
                nn.ReLU())
        )

        self.residual_block = nn.Sequential(
            ResidualBlock(channel_num=200),
            ResidualBlock(channel_num=200),
            ResidualBlock(channel_num=200),
            ResidualBlock(channel_num=200)
        )

        self.reduced_residual_block = ResidualBlock(channel_num=200)

        self.output_block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 5), in_channels=200, out_channels=50),
            Flatten(),
            nn.Linear(in_features=2150, out_features=output_size)
        )

    def forward(self, x):
        x = self.temporal_block(x)  # Expected 4D tensor input in shape of : [BS, F, CH, LEN]
        x = self.spatial_block(x)
        x = self.reduced_residual_block(x)
        x = self.output_block(x)
        return x


def channelNetLoss(l_e1: torch.Tensor, l_v1: torch.Tensor, l_v2: torch.Tensor):
    """
    The input should be a latent vector only.
    """
    loss = (l_e1 @ l_v2.transpose(0, 1)) - (l_e1 @ l_v1.transpose(0, 1))
    loss = torch.diag(loss, diagonal=0).reshape(-1, 1).clamp(min=0)
    return loss  # shape should be : [bs, 1]


class ViT(nn.Module):

    def __init__(self):
        super(ViT, self).__init__()

    def forward(self, x):
        return x
