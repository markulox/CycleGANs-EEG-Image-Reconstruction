import os
from torch.utils.data import Dataset
import pickle


class SixObject1KStimuli(Dataset):
    __dirname__ = os.path.dirname(__file__)
    __FILE_TRAIN_LOC__ = os.path.join(__dirname__, 'content/six_objects_1000stimuli/6class_1k_img_each.dat')

    # __FILE_VAL_LOC__ = os.path.join(__dirname__, 'content/very_nice_dataset/')

    def __init__(self, dev):
        super(SixObject1KStimuli, self).__init__()
        self.whole_data = pickle.load(open(self.__FILE_TRAIN_LOC__, "rb"))
        self.dev = dev

    def __getitem__(self, idx):
        curr_item = self.whole_data[idx]
        stim = curr_item[0].to(self.dev)
        label = curr_item[1].to(self.dev)
        # Normalize stim
        stim = (stim - 127.5) / 127.5
        return stim, label

    def __len__(self):
        return len(self.whole_data)

    def get_name(self):
        return "SixObject1KStimuli"


if __name__ == "__main__":
    dataset = SixObject1KStimuli(dev="cpu")
    for i, (stim, label) in enumerate(dataset):
        print(i, stim.shape, label.shape)
