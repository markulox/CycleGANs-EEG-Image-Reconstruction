import torch.nn as nn


# Should D1 and D2 takes an real/gen image as an input?
# D1 : Image only
# D2 : Semantic features and label
class SimpleDiscrim(nn.Module):
    def __init__(self):
        super(SimpleDiscrim, self).__init__()
        self.conv1 = nn.Sequential(  # Currently we input black and white img
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=46),
            nn.LeakyReLU(),
            nn.Linear(in_features=46, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        x = self.final_fc(x)
        return x
