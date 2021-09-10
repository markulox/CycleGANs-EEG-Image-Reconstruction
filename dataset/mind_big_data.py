import pickle
import os

from torch.utils.data import Dataset, DataLoader


class MindBigData(Dataset):
    __dirname__ = os.path.dirname(__file__)

    __FILE_TRAIN_LOC__ = os.path.join(__dirname__, 'content/mind_big_data/mind_big_data_64x64_label_40class.dat')

    def __init__(self, dat_pth=__FILE_TRAIN_LOC__, dev='cpu'):
        """
        :param dat_pth: Path of .dat file (Expected in format [(EEG_1,STIM_1),...,(EEG_N,STIM_N)])
        """
        super(MindBigData, self).__init__()
        self.dev = dev
        print("<I> : Loading dataset")
        print(dat_pth)
        self.data_holder = pickle.load(open(dat_pth, "rb"))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: Tuple of EEG tensor and STIM tensor
        """
        eeg, label, stim = self.data_holder[idx]
        # Move the data to a given dev
        eeg = eeg.to(self.dev)
        stim = stim.to(self.dev)
        label = label.to(self.dev)
        return eeg, label, stim

    def __len__(self):
        return len(self.data_holder)

    def get_name(self):
        return "MindBigData_64x64_40_class"


if __name__ == '__main__':
    dataset = MindBigData()
    dat_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for eeg, stim in dat_loader:
        print(eeg.shape)
        print(stim.shape)
        print("-------")
