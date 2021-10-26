import os
import torch
from libs.utilities import get_freer_gpu

# Load model
load_at_epoch = 0
LOAD_GEN = True
LOAD_DIS = True

num_epochs_alex = 5
num_epochs = 500
workers = 0
batch_size = 32
image_size = 64
num_channel_img = 3  # <---change
z_dim = 100
gen_dim = 64
dis_dim = 64
ngpu = 1
lr = 1e-4  # <-----changed
lr_g = 1e-4
lambda_gp = 10
embed_size = 100  # <---follow the paper
feature_size = 200

# Image save count
img_save_num = 10

ie_epoch_steps = 2
d_epoch_steps = 5
g_epoch_steps = 1

device = torch.device(get_freer_gpu()) if torch.cuda.is_available() else torch.device("cpu")


def add_proxy():
    os.environ['http_proxy'] = 'http://192.41.170.23:3128'
    os.environ['https_proxy'] = 'http://192.41.170.23:3128'


def remove_proxy():
    os.environ.pop('http_proxy')
    os.environ.pop('https_proxy')
