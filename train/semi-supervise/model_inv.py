import torch
import os.path as pth
from dataset.EEGImageNet_Spam_et_al import EEGImageNetDataset_Spam
from model_lib.semi_supervised.model import *

LOAD_AT_EPCH = "5000"

SAVED_MODEL_LOC = "saved_models/" + EEGImageNetDataset_Spam.get_name()
d1_name = LOAD_AT_EPCH + "_d1.pth"
d2_name = LOAD_AT_EPCH + "_d2.pth"
G_name = LOAD_AT_EPCH + "_G.pth"
sx_name = LOAD_AT_EPCH + "_sx.pth"
sy_name = LOAD_AT_EPCH + "_sy.pth"

load_path = {
    "d1": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, d1_name),
    "d2": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, d2_name),
    "G": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, G_name),
    "sx": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, sx_name),
    "sy": pth.join(pth.dirname(__file__), SAVED_MODEL_LOC, sy_name)
}

d1 = D1()
d1.load_state_dict(torch.load(load_path["d1"]))

NUM_LIM_CLASS = 40
LATENT_SIZE = 200
EMBEDDED_SIZE = 40
G = Generator(num_classes=NUM_LIM_CLASS, embed_size=EMBEDDED_SIZE, latent_size=LATENT_SIZE)
G.load_state_dict(torch.load(load_path["G"]))


def simple_layer_info(model):
    for name, param in model.named_parameters():
        print("Layer name: ", name, "shape", param.shape)
        print("\t> Min:", torch.min(param).item(), "Max:", torch.max(param).item())
        print("\t> Mean:", torch.mean(param).item(), "Std:", torch.std(param).item())
        print("----------")


pass
