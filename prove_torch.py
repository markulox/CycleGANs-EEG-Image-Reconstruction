import torch
import torch.nn as nn

from dataset.mind_big_data import MindBigData
from dataset.cylinder_rgb import Cylinder_RBG_Dataset

a = MindBigData(dev="cpu")
b = Cylinder_RBG_Dataset(dev="cpu")
print(len(b))
