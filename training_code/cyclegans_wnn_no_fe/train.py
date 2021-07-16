from param_config import *

import torch

from model.dis.eeg import DisA2 as D_EEG
from model.dis.image import DisA1 as D_IMG
from model.gen.eeg import GenA1 as G_EEG
from model.gen.image import GenA1 as G_IMG
from torch.utils.data import DataLoader
from dataset.cylinder_rgb import Cylinder_RBG_Dataset

# Init dataset
dataset = Cylinder_RBG_Dataset()
dat_loader = DataLoader(dataset, shuffle=True, batch_size=BS)

IN_CHNNL: int = dataset.get_data_shape()[0]
IN_LEN: int = dataset.get_data_shape()[1]

d_eeg = D_EEG(in_channel=IN_CHNNL, in_len=IN_LEN)  # In this work, we aren't going to chunk EEG.
d_img = D_IMG()  # No additional params required
g_eeg = G_EEG(out_channel=IN_CHNNL)
g_img = G_IMG()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()  # Is impossible due to domain A and domain B has not the same data's shape
