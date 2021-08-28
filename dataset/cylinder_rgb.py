import pandas as pd
import pickle
import numpy as np
import torchvision.transforms as transforms
import PIL
import PIL.Image as Image
from torch.utils.data.dataset import Dataset
import torch
import os
from sklearn.model_selection import train_test_split


class Cylinder_RBG_Dataset(Dataset):
    """
    Current data's shape : 16x189
    """
    __dirname__ = os.path.dirname(__file__)

    __FILE_TRAIN_LOC_X__ = os.path.join(__dirname__, 'content/cylinder_rgb/data/1-par1_perception_0.0_128.0_X.npy')
    __FILE_TRAIN_LOC_Y__ = os.path.join(__dirname__, 'content/cylinder_rgb/data/1-par1_perception_0.0_128.0_y.npy')
    __FILE_LOC_STIM__ = os.path.join(__dirname__, 'cylinder_rgb/stim/')

    __IMG_SIZE__ = 128

    __img_name__ = ["r.png", "g.png", "b.png"]
    __stim__ = []

    def __init__(self, validation=False, dev="cpu"):

        self.__DEV__ = dev

        X = np.load(self.__FILE_TRAIN_LOC_X__)
        y = np.load(self.__FILE_TRAIN_LOC_Y__)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if not validation:
            self.obj_data = X_train
            self.obj_label = y_train
        else:
            self.obj_data = X_test
            self.obj_label = y_test

        self.transformer = transforms.Compose([transforms.Resize(self.__IMG_SIZE__),
                                               transforms.CenterCrop(self.__IMG_SIZE__),
                                               transforms.ToTensor()])

        for each_stim in self.__img_name__:
            stim: PIL.PngImagePlugin.PngImageFile = Image.open(self.__FILE_LOC_STIM__ + each_stim)
            self.__stim__.append(self.transformer(stim.convert('RGB')))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: EEG Signals, Label and Stimuli Image
        """
        label_idx = self.obj_label[idx].item()
        eeg = torch.Tensor(self.obj_data[idx, :, :]).float()
        label = torch.Tensor([self.obj_label[idx]]).float()
        stim = torch.Tensor(self.__stim__[label_idx]).float()
        return eeg.to(self.__DEV__), label.to(self.__DEV__), stim.to(self.__DEV__)

    def __len__(self):
        return self.obj_data.shape[0]

    def get_image(self, idx):
        return self.__stim__[idx]

    def get_data_shape(self):
        return self.obj_data[0].shape

    def get_label_shape(self):
        return self.obj_label[0].shape

    def get_name(self):
        return "CylinderRGB"

# a = Cylinder_RBG_Dataset()
# z = a[14][0].shape
# print(z)
