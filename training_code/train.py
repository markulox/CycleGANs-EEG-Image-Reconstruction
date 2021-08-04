import torch
from model.template import *


# Check CUDA available
cuda = torch.cuda.is_available()

# Model declaration & Init
G_EEG_TO_IMG = G_EEG

