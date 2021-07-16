import torch
import torch.nn as nn


class DisA1(nn.Module):
    """
    Image[3,224,224] -> Label
    Current expected image size = 3x224x224
    """

    def __init__(self):
        super(DisA1, self).__init__()
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
        print(x.shape)
        x = self.conv2(x)  # 128x53x53
        print(x.shape)
        x = self.conv3(x)  # 256x25x25
        print(x.shape)
        x = self.conv4(x)  # 512x11x11
        print(x.shape)
        x = x.reshape((x.shape[0], 512))  #
        x = self.final_fc(x)
        return x


# sample = torch.rand(1, 3, 224, 224)
# a = DisA1()
# b = a(sample)
# print(b.shape)
