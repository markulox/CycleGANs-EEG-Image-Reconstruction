from torch.nn import Module
from model.module.wnn_layer import *


class EEG2IMG(Module):
    """
    Generate an image from EEG signal
    """

    def __init__(self):
        super(EEG2IMG, self).__init__()

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


class IMG2EEG(Module):
    """
    Generate an EEG signal from image
    """

    def __init__(self):
        super(IMG2EEG, self).__init__()

    def forward(self, x):
        return x


class D_EEG(Module):
    """
    Discriminate generated EEG signal and the real one
    """

    def __init__(self):
        super(D_EEG, self).__init__()

    def forward(self, x):
        return x
