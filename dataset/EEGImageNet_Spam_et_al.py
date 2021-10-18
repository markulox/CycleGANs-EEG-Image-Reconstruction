import torch
import os
from torch.utils.data import Dataset

import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import DataLoader


# Dataset class
class EEGImageNetDataset_Spam(Dataset):
    __EEG_TRAIN_LOC__ = 'content/EEGImageNetDataset_Spampinato/eeg_5_95_std.pth'
    __STIM_LIB_LOC__ = 'content/EEGImageNetDataset_Spampinato/stim_lib.dat'

    #     __EEG_TRAIN_LOC__ = "content/EEGImageNetDataset_Spam/eeg_5_95_std.pth"
    #     __STIM_LIB_LOC__ = "content/EEGImageNetDataset_Spam/stim_lib.dat"

    TRAIN_MODE = True
    NUM_CLASSES = 40

    # Constructor
    def __init__(self, dev, subject=0, sample_min_idx=20, sample_max_idx=460, model_type='model10', train_ratio=0.8):
        # Load EEG signals
        eeg_path = os.path.join(os.path.dirname(__file__), self.__EEG_TRAIN_LOC__)
        loaded = torch.load(eeg_path)
        stim_path = os.path.join(os.path.dirname(__file__), self.__STIM_LIB_LOC__)
        self.img_lib = pickle.load(open(stim_path, "rb"))
        self.dev = dev
        if subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == subject]
        else:
            self.data = loaded['dataset']

        # Split train test
        self.data_train, self.data_test = train_test_split(self.data, test_size=1 - train_ratio)

        self.labels = loaded["labels"]
        self.images = loaded["images"]

        self.time_low = sample_min_idx
        self.time_high = sample_max_idx

        self.model_type = model_type

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        if self.TRAIN_MODE:
            return len(self.data_train)
        else:
            return len(self.data_test)

    def set_train(self):
        self.TRAIN_MODE = True

    def set_test(self):
        self.TRAIN_MODE = False

    # Get item
    def __getitem__(self, i):
        """
        :param i: Index of item
        :return: EEG signal(BS, 128, sample_min_idx - sample_max_idx), Stimuli image(BS,3,64,64), Label(BS, 40)
        """
        # Process EEG
        if self.TRAIN_MODE:
            item = self.data_train[i]
        else:
            item = self.data_test[i]

        eeg = item["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high, :]

        if self.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, self.time_high - self.time_low)
        # Get label
        label = item["label"]
        label_tensor = torch.tensor(label)
        # label_tensor = F.one_hot(label_tensor, num_classes=num_classes)

        # Get stim
        stim_index = item["image"]
        stim = self.img_lib[stim_index]

        return eeg.to(self.dev).squeeze(0), stim.to(self.dev), label_tensor.to(self.dev)

    @staticmethod
    def get_name():
        return "EEGImageNetDataset_Spam"


class UnpairedStimuliDataset(Dataset):
    __CONTENT_LOC__ = 'content/EEGImageNetDataset_Spampinato/unpaired_stim.dict'
    NUM_CLASSES = 40

    def __init__(self, dev):
        super(UnpairedStimuliDataset, self).__init__()
        load_path = os.path.join(os.path.dirname(__file__), self.__CONTENT_LOC__)
        self.loaded = pickle.load(open(load_path, "rb"))
        self.stim_list = self.loaded['stimuli']
        self.label_list = self.loaded['labels']
        self.dev = dev

    def __getitem__(self, idx):
        return self.stim_list[idx, :, :, :].to(self.dev), self.label_list[idx, :].to(self.dev)

    def __len__(self):
        return self.stim_list.shape[0]


# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

# train_iterator = DataLoader(train_split, batch_size=batch_size, shuffle=True)
