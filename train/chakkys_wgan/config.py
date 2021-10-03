import torch

num_epochs_alex = 5
num_epochs = 500
workers = 0
batch_size = 32
image_size = 64
num_channel = 3  # <---change
z_dim = 100
gen_dim = 64
dis_dim = 64
ngpu = 1
lr = 1e-4  # <-----changed
lambda_gp = 10
embed_size = 100  # <---follow the paper
feature_size = 200

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
