import os
import torch
from torch.utils.data import Dataset
import pickle
import random


class VeryNiceDataset(Dataset):
    __dirname__ = os.path.dirname(__file__)
    __FILE_TRAIN_LOC__ = os.path.join(__dirname__, 'content/very_nice_dataset/very_nice_dataset.dat')

    # __FILE_VAL_LOC__ = os.path.join(__dirname__, 'content/very_nice_dataset/')

    def __init__(self, dev, participant_id=1):
        super(VeryNiceDataset, self).__init__()
        self.whole_data = pickle.load(open(self.__FILE_TRAIN_LOC__, "rb"))
        self.curr_participant = self.whole_data[participant_id]
        self.dev = dev

    def __getitem__(self, idx):
        curr_entry = self.curr_participant[idx]
        eeg = curr_entry[0].to(self.dev)
        label = curr_entry[1].to(self.dev)
        stim = curr_entry[2].to(self.dev)
        # Normalize stim
        stim = (stim - 127.5) / 127.5
        return eeg, label, stim

    def __len__(self):
        return len(self.curr_participant)

    def get_name(self):
        return "VeryNiceDataset"

    def change_participant_id(self, participant_id=1):
        self.curr_participant = self.whole_data[participant_id]


class VeryNiceDatasetv2(VeryNiceDataset):

    def __init__(self, dev, participant_id=1):
        super().__init__(dev=dev, participant_id=participant_id)

    def __getitem__(self, idx):
        eeg, label, stim = super().__getitem__(idx)
        while True:
            rng = random.randint(a=0, b=len(self.whole_data))
            eeg_w, label_w, stim_w = super().__getitem__(rng)
            if torch.argmax(label_w) != torch.argmax(label):
                break

        return eeg, eeg_w, label, label_w, stim, stim_w


if __name__ == "__main__":
    dataset = VeryNiceDataset(dev="cpu", participant_id=1)
    for i, (eeg, label, stim) in enumerate(dataset):
        print(i, eeg.shape, label.shape, stim.shape)
