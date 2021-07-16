from torch.nn import Module
from model.module.wnn_layer import *


class G_IMG(Module):
    """
    Generate an image from EEG signal
    """
    def __init__(self):
        super(G_IMG, self).__init__()


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


class G_EEG(Module):
    """
    Generate an EEG signal from image
    """
    def __init__(self):
        super(G_EEG, self).__init__()

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


# Lost functions
l_gan = torch.nn.MSELoss()
l_cycle = torch.nn.L1Loss()
l_iden = torch.nn.L1Loss()
