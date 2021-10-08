import os
import torch

# Config some proxy
# os.environ['http_proxy'] = 'http://192.41.170.23:3128'
# os.environ['https_proxy'] = 'http://192.41.170.23:3128'

num_epochs_alex = 5
num_epochs = 5000
workers = 0
batch_size = 32
image_size = 64
num_channel = 3  # <---change
z_dim = 100
gen_dim = 64
dis_dim = 64
ngpu = 1
lr = 1e-4  # <-----changed
lr_g = 1e-4
lambda_gp = 10
embed_size = 100  # <---follow the paper
feature_size = 200

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
