import datetime
import itertools
import sys
import time

from train.cycle_gans.config.gans import *

import torch

from model.cycle_gans.beau_inspired_model import D_EEG as D_EEG
from model.cycle_gans.beau_inspired_reduce import D_IMG as D_IMG
from model.cycle_gans.beau_inspired_reduce import IMG2EEG as G_EEG
from model.cycle_gans.beau_inspired_reduce import EEG2IMG as G_IMG
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataset.cylinder_rgb import Cylinder_RBG_Dataset
from utils import ReplayBuffer, LambdaLR

# Init dataset
dataset = Cylinder_RBG_Dataset(dev=DEV)
dat_loader = DataLoader(dataset, shuffle=True, batch_size=BS)

valset = Cylinder_RBG_Dataset(dev=DEV, validation=True)
val_loader = DataLoader(valset, shuffle=True, batch_size=5)

IN_CHNNL: int = dataset.get_data_shape()[0]
IN_LEN: int = dataset.get_data_shape()[1]

# Init model
d_eeg = D_EEG(in_channel=IN_CHNNL, in_len=IN_LEN).to(DEV)  # In this work, we aren't going to chunk EEG.
d_img = D_IMG().to(DEV)  # No additional params required
g_eeg = G_EEG(out_channel=IN_CHNNL).to(DEV)
g_img = G_IMG(in_channel=IN_CHNNL).to(DEV)

# Optim
g_optim = torch.optim.Adam(itertools.chain(g_eeg.parameters(), g_img.parameters())
                           , lr=LR_DEFAULT
                           , betas=(B1, B2))
d_img_optim = torch.optim.Adam(d_img.parameters(), lr=LR_DEFAULT, betas=(B1, B2))
d_eeg_optim = torch.optim.Adam(d_eeg.parameters(), lr=LR_DEFAULT, betas=(B1, B2))

# LR update schedulers
lr_sched_G = torch.optim.lr_scheduler.LambdaLR(
    g_optim, lr_lambda=LambdaLR(EPCH, EPCH_START, DECAY_EPCH).step
)
lr_sched_D_img = torch.optim.lr_scheduler.LambdaLR(
    d_img_optim, lr_lambda=LambdaLR(EPCH, EPCH_START, DECAY_EPCH).step
)
lr_sched_D_eeg = torch.optim.lr_scheduler.LambdaLR(
    d_eeg_optim, lr_lambda=LambdaLR(EPCH, EPCH_START, DECAY_EPCH).step
)

# Replay Buffer
fake_eeg_buffer = ReplayBuffer()
fake_stim_buffer = ReplayBuffer()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()  # This is impossible due to domain A and domain B's shape is not the same.


# Some visualization function
def sample_images(epch):
    """Saves a generated sample from the test set"""
    real_eeg, real_label, real_stim = next(iter(dat_loader))
    g_eeg.eval()
    g_img.eval()
    fake_stim = g_img(real_eeg)
    fake_eeg = g_eeg(real_stim)
    # Arange images along x-axis
    # real_eeg = make_grid(real_eeg, nrow=5, normalize=True)
    real_stim = make_grid(real_stim, nrow=5, normalize=True)
    # fake_eeg = make_grid(fake_A, nrow=5, normalize=True)
    fake_stim = make_grid(fake_stim, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_stim, fake_stim), 1)
    save_image(image_grid, "images/%s/%s_reduce.png" % (dataset.get_name(), epch), normalize=False)


def load_model(start_epch):
    if start_epch != 0:
        # Load pretrained models
        print("<I> : Loading model at epoch check point = %d" % start_epch)
        g_img.load_state_dict(torch.load("saved_models/%s/g_img_%d_reduce.pth" % (dataset.get_name(), start_epch)))
        g_eeg.load_state_dict(torch.load("saved_models/%s/g_eeg_%d_reduce.pth" % (dataset.get_name(), start_epch)))
        d_img.load_state_dict(torch.load("saved_models/%s/d_img_%d_reduce.pth" % (dataset.get_name(), start_epch)))
        d_eeg.load_state_dict(torch.load("saved_models/%s/d_eeg_%d_reduce.pth" % (dataset.get_name(), start_epch)))


####################
# Train the model  #
####################
load_model(EPCH_START)
prev_time = time.time()
for epch in range(EPCH_START, EPCH+1):
    for i, (real_eeg, real_label, real_stim) in enumerate(dat_loader):
        valid = torch.ones(real_eeg.shape[0], 1, requires_grad=False, dtype=torch.float32, device=DEV)
        fake = torch.zeros(real_eeg.shape[0], 1, requires_grad=False, dtype=torch.float32, device=DEV)

        ###############################
        # Generator training section  #
        ###############################
        g_eeg.train()
        g_img.train()

        g_optim.zero_grad()

        # GAN Loss
        # A = EEG, B = Image
        # Tried to make generator able to fool discriminator
        fake_stim = g_img(real_eeg)
        fake_eeg = g_eeg(real_stim)
        loss_g_stim = criterion_GAN(d_img(fake_stim), valid)
        loss_g_eeg = criterion_GAN(d_eeg(fake_eeg), valid)

        loss_GAN = (loss_g_stim + loss_g_eeg) / 2

        # Cycle Loss
        recov_eeg = g_eeg(fake_stim)
        recov_stim = g_img(fake_eeg)
        loss_cycle_eeg = criterion_cycle(recov_eeg, real_eeg)
        loss_cycle_stim = criterion_cycle(recov_stim, real_stim)

        loss_cycle = (loss_cycle_stim + loss_cycle_eeg) / 2
        loss_G = loss_GAN + LAMBDA_CYC * loss_cycle
        loss_G.backward()
        g_optim.step()

        ###################################
        # Discriminator training section  #
        ###################################
        # DA (EEG)
        d_eeg_optim.zero_grad()

        loss_real = criterion_GAN(d_eeg(real_eeg), valid)
        fake_eeg_ = fake_eeg_buffer.push_and_pop(fake_eeg)

        loss_fake = criterion_GAN(d_eeg(fake_eeg_.detach()), fake)
        loss_d_eeg = (loss_real + loss_fake) / 2

        loss_d_eeg.backward()
        d_eeg_optim.step()

        # DB (STIM or IMG)
        d_img_optim.zero_grad()
        loss_real = criterion_GAN(d_img(real_stim), valid)

        fake_stim_ = fake_stim_buffer.push_and_pop(fake_stim)
        loss_fake = criterion_GAN(d_img(fake_stim_.detach()), fake)
        loss_d_img = (loss_real + loss_fake) / 2

        loss_d_img.backward()
        d_img_optim.step()

        loss_D = (loss_d_img + loss_d_eeg) / 2

        #####################
        # Progress Logging  #
        #####################

        # Determine approximate time left
        batches_done = epch * len(dat_loader) + i
        batches_left = EPCH * len(dat_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: N/A] ETA: %s"
            % (
                epch,
                EPCH,
                i,
                len(dat_loader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                time_left,
            )
        )

    if epch % SAMPLE_INTERVAL == 0:
        sample_images(epch)

    lr_sched_G.step()
    lr_sched_D_eeg.step()
    lr_sched_D_img.step()

    if CHCK_PNT_INTERVAL != -1 and epch % CHCK_PNT_INTERVAL == 0:
        # Save model checkpoints
        torch.save(g_img.state_dict(), "saved_models/%s/g_img_%d.pth" % (dataset.get_name(), epch))
        torch.save(g_eeg.state_dict(), "saved_models/%s/g_eeg_%d.pth" % (dataset.get_name(), epch))
        torch.save(d_img.state_dict(), "saved_models/%s/d_img_%d.pth" % (dataset.get_name(), epch))
        torch.save(d_eeg.state_dict(), "saved_models/%s/d_eeg_%d.pth" % (dataset.get_name(), epch))
