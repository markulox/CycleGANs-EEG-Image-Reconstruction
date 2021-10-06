import os

import torch
from torch.utils.data import Dataset
import pickle
import random


class SixObject1KStimuli(Dataset):
    __dirname__ = os.path.dirname(__file__)
    __FILE_TRAIN_LOC__ = os.path.join(__dirname__, 'content/six_objects_1000stimuli/6class_1k_img_each_no_idle.dat')

    # __FILE_VAL_LOC__ = os.path.join(__dirname__, 'content/very_nice_dataset/')

    def __init__(self, dev, exclude_class: list = None):
        super(SixObject1KStimuli, self).__init__()
        self.exclude_class = exclude_class
        self.whole_data = pickle.load(open(self.__FILE_TRAIN_LOC__, "rb"))
        self.dev = dev

    def __getitem__(self, idx):
        curr_item = self.whole_data[idx]
        stim = curr_item[0].to(self.dev)
        label = curr_item[1].to(self.dev)
        # Normalize stim
        stim = (stim - 0.5) / 0.5
        return stim, label

    def __len__(self):
        return len(self.whole_data)

    def get_name(self):
        return "SixObject1KStimuli"


class SixObject1KStimuliv2(SixObject1KStimuli):
    def __init__(self, dev, exclude_class: list = None):
        super().__init__(dev=dev, exclude_class=exclude_class)

    def __getitem__(self, idx):
        stim, label = super().__getitem__(idx)
        while True:
            rng = random.randint(a=0, b=len(self.whole_data))
            stim_w, label_w = super().__getitem__(rng)
            if torch.argmax(label_w) != torch.argmax(label):
                break

        return stim, stim_w, label, label_w


if __name__ == "__main__":
    dataset = SixObject1KStimuliv2(dev="cpu")
    for i, (stim, stim_w, label, label_w) in enumerate(dataset):
        print(i, stim.shape, torch.argmax(label))
        print(stim_w.shape, torch.argmax(label_w))
