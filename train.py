import torch
from model.classics_cyclegan import *


# Check CUDA available
cuda = torch.cuda.is_available()

# Model declaration & Init
G_EEG_TO_IMG = G_EEG

if cuda:
    l_gan.cuda()
    l_iden.cuda()
    l_cycle.cuda()