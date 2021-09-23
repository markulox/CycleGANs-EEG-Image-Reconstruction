import sys
import os
import torch
import time
import datetime
from torch.utils.data import DataLoader
from config import *
from dataset.very_nice_dataset import VeryNiceDataset

from model.semi_supervised.model import Generator
from model.simple_gans import SimpleDiscrim

from torchvision.utils import make_grid, save_image

dataset = VeryNiceDataset(dev=DEV)
train_loader = DataLoader(dataset, shuffle=True, batch_size=BS)

NUM_CLASS = 6

D = SimpleDiscrim().to(DEV)
G = Generator(num_classes=NUM_CLASS, latent_size=200).to(DEV)

D_optim = torch.optim.Adam(D.parameters(), lr=LR_D)
G_optim = torch.optim.Adam(G.parameters(), lr=LR_G)

loss_func = torch.nn.BCELoss()

__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "saved_models/%s/" % dataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "images/%s/" % dataset.get_name())


# Some helper functions
def sample_images(epch):
    """Saves a generated sample from the test set"""
    real_eeg, _, real_stim = next(iter(train_loader))

    curr_bs = real_eeg.shape[0]
    z1 = torch.rand(curr_bs, G.EXPECTED_NOISE).to(DEV)
    z2 = torch.rand(curr_bs, 200).to(DEV)
    z3 = torch.rand(curr_bs, NUM_CLASS).to(DEV)

    G.eval()

    fake_stim = G(z1, z2, z3)
    fake_stim = make_grid(fake_stim, nrow=5, normalize=True)
    # Arange images along y-axis
    real_stim = make_grid(real_stim, nrow=5, normalize=True)
    image_grid = torch.cat((real_stim, fake_stim), 1)
    save_image(image_grid, IMAGE_PATH + "%s_reduce.png" % epch, normalize=False)


prev_time = time.time()
for epch in range(EPCH_END):
    for idx, (y_p, l_real_p, x_p) in enumerate(train_loader):
        # x_p = x_p / 255.0
        curr_bs = x_p.shape[0]
        z1 = torch.rand(curr_bs, G.EXPECTED_NOISE).to(DEV)
        z2 = torch.rand(curr_bs, 200).to(DEV)
        z3 = torch.rand(curr_bs, NUM_CLASS).to(DEV)

        G.eval()
        gen_img = G(z1, z2, z3)
        gen_img_detach = gen_img.detach()
        pred_d_real = D(x_p)

        pred_d_fake = D(gen_img)

        # Label stuff
        true = torch.ones_like(pred_d_real).type(torch.FloatTensor).to(DEV)
        false = torch.zeros_like(pred_d_fake).type(torch.FloatTensor).to(DEV)

        # Train D
        D.train()
        D_optim.zero_grad()
        loss_d = loss_func(pred_d_real, true) + loss_func(pred_d_fake, false)
        loss_d.backward()
        D_optim.step()

        # Train G
        D.eval()
        G.train()
        G_optim.zero_grad()
        loss_g = loss_func(D(G(z1, z2, z3)), true)
        loss_g.backward()
        G_optim.step()

        sys.stdout.write(
            "\r%d D[%.4f] G[%.4f]"
            % (
                epch,
                loss_d.item(),
                loss_g.item()
            )
        )

        # Logging the progress
        batches_done = epch * len(train_loader) + idx
        batches_left = EPCH_END * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if SAMPLE_INTERVAL != -1 and epch % SAMPLE_INTERVAL == 0 and idx + 1 == len(train_loader):
            sample_images(epch)
