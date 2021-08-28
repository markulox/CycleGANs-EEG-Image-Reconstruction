import torch.nn as nn
import torch
import math


# from model.semi_supervised.loss_func import *


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=3, out_channels=6, stride=2),
            nn.Conv2d(kernel_size=5, in_channels=6, out_channels=12),
            nn.Conv2d(kernel_size=5, in_channels=12, out_channels=24),
            nn.Conv2d(kernel_size=5, in_channels=24, out_channels=48),
            nn.Conv2d(kernel_size=3, in_channels=48, out_channels=2),
        )

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        return x


def check_nan(tag='', chck=0, **log):
    if math.isnan(chck):
        print("<!>[%s] : NAN detected --------" % tag)
        for each_elm in log:
            print(each_elm, log[each_elm], sep='=')
        print("-------------------------------")


if __name__ == "__main__":
    l1 = 2
    l2 = 44
    check_nan("D_LOSS", 12, gl1=l1, gl2=l2)
