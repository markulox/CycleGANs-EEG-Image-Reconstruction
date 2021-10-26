import torch
import matplotlib.pyplot as plt
import os.path as pth
from dataset.EEGImageNet_Spam_et_al import EEGImageNetDataset_Spam
from model_lib.semi_supervised.model import *
from model_lib.extras.EEGNet import EEGNet_Extractor

from config import *

# from train import d1, d2, G, sx, sy

LOAD_AT_EPCH = "5000"

SAVED_MODEL_LOC = "saved_models/" + EEGImageNetDataset_Spam.get_name()
d1_name = LOAD_AT_EPCH + "_d1.pth"
d2_name = LOAD_AT_EPCH + "_d2.pth"
G_name = LOAD_AT_EPCH + "_G.pth"
sx_name = LOAD_AT_EPCH + "_sx.pth"
sy_name = LOAD_AT_EPCH + "_sy_EEGNet.pth"

load_path = {
    "d1": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, d1_name),
    "d2": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, d2_name),
    "G": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, G_name),
    "sx": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, sx_name),
    "sy": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, sy_name)
}

NUM_LIM_CLASS = 40
sample_len = 440
channel_num = 128

d1 = D1()
d2 = D2(num_classes=NUM_LIM_CLASS, embed_size=EMBEDDED_SIZE)
G = Generator(num_classes=NUM_LIM_CLASS, embed_size=EMBEDDED_SIZE, latent_size=LATENT_SIZE)
sx = SemanticImageExtractorV2(
    output_class_num=NUM_LIM_CLASS,
    feature_size=LATENT_SIZE,
    pretrain=True
)
sy = EEGNet_Extractor(
    in_channel=channel_num,
    samples=sample_len,
    kern_len=sample_len // 2,
    F1=10,
    F2=10,
    D=2,
    nb_classes=NUM_LIM_CLASS,
    latent_size=LATENT_SIZE
)

d1.load_state_dict(torch.load(load_path["d1"]))
# d2.load_state_dict(torch.load(load_path["d2"]))
G.load_state_dict(torch.load(load_path["G"]))
sx.load_state_dict(torch.load(load_path["sx"]))
sy.load_state_dict(torch.load(load_path["sy"]))


def simple_layer_info(model):
    for name, param in model.named_parameters():
        print("Layer name: ", name, "shape", param.shape)
        print("\t> Min:", torch.min(param).item(), "Max:", torch.max(param).item())
        print("\t> Mean:", torch.mean(param).item(), "Std:", torch.std(param).item())
        print("----------")


def show_heatmaps(w):
    w = w.detach().numpy()
    plt.imshow(w)
    plt.show()


pass
