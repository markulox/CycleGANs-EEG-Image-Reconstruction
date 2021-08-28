import pickle

from torch.utils.data import Dataset, DataLoader

__PATH__DEFAULT__ = "/home/nopphon/Documents/AIT/Special " \
                    "study/code/WCycleGANs_slim/code/dataset/content/mind_big_data/mind_big_data_64x64_label_40class.dat"


class MindBigData(Dataset):
    def __init__(self, dat_pth=__PATH__DEFAULT__, dev='cpu'):
        """
        :param dat_pth: Path of .dat file (Expected in format [(EEG_1,STIM_1),...,(EEG_N,STIM_N)])
        """
        super(MindBigData, self).__init__()
        self.dev = dev
        print("<I> : Loading dataset")
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
        return "MindBigData_128x128"


if __name__ == '__main__':
    dataset = MindBigData()
    dat_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for eeg, stim in dat_loader:
        print(eeg.shape)
        print(stim.shape)
        print("-------")
