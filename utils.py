import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from torch import nn as nn
import numpy as np

from torchvision.utils import save_image


# custom weights initialization called on netG and netD
def weights_init(model):
    for m in model.modules():  # loop all layers in that model
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class WeightClipper(object):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(self.min, self.max)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def check_inf_nan_tensor(tensor):
    print("nan : {}".format(torch.sum(torch.isnan(tensor)).item()))
    print("inf : {}".format(torch.sum(torch.isinf(tensor)).item()))

# # Check the split dataset
# def check_split(X1, X2, y1, y2, name1, name2):
#     unique1, count1 = np.unique(y1, return_counts=True)
#     unique2, count2 = np.unique(y2, return_counts=True)
#
#     assert count1[0] == count1[1] == count1[2]
#     assert count2[0] == count2[1] == count2[2]
#
#     print('=' * 20, name1, '=' * 20)
#     print(f"Shape of X_{name1}: ", X1.shape)
#     print(f"Shape of y_{name1}: ", y1.shape)
#     print(f"Classes of y_{name1}: ", unique1)
#     print(f"Counts of y_{name1} classes: ", count1)
#     print('=' * 20, name2, '=' * 20)
#     print(f"Shape of X_{name2}: ", X2.shape)
#     print(f"Shape of y_{name2}: ", y2.shape)
#     print(f"Classes of y_{name2}: ", unique2)
#     print(f"Counts of y_{name2} classes: ", count2)
