import datetime
import sys
import time

from config.fe import *
import torch
from model.cycle_gans.fe import EEGChannelNetEncoder as EEG_Encode
from model.module.inception_v4 import InceptionV4 as IMG_Encode
from dataset.cylinder_rgb import Cylinder_RBG_Dataset as Dataset

from torch.utils.data import DataLoader
from utils import LambdaLR

dataset = Dataset(dev=DEV)
dat_loader = DataLoader(dataset, shuffle=True, batch_size=BS)

valset = Dataset(dev=DEV, validation=True)
val_loader = DataLoader(valset, shuffle=True, batch_size=4)

IN_CHNNL: int = dataset.get_data_shape()[0]
IN_LEN: int = dataset.get_data_shape()[1]

# Model declaration
eeg_encoder = EEG_Encode(in_channel=IN_CHNNL, output_size=LATENT_SIZE).to(DEV)
img_encoder = IMG_Encode(num_classes=LATENT_SIZE).to(DEV)

eeg_optim = torch.optim.Adam(eeg_encoder.parameters(), lr=LR_DEFAULT, betas=(B1, B2))
img_optim = torch.optim.Adam(img_encoder.parameters(), lr=LR_DEFAULT, betas=(B1, B2))

# LR update schedulers
lr_sched_eeg_optim = torch.optim.lr_scheduler.LambdaLR(
    eeg_optim, lr_lambda=LambdaLR(EPCH, EPCH_START, DECAY_EPCH).step
)
lr_sched_img_optim = torch.optim.lr_scheduler.LambdaLR(
    img_optim, lr_lambda=LambdaLR(EPCH, EPCH_START, DECAY_EPCH).step
)

criterion = torch.nn.MSELoss()


# I use this instead of channelNetLoss because we have too few of y.


def load_model(start_epch):
    if start_epch != 0:
        # Load pretrained models
        print("<I> : Loading model at epoch check point = %d" % start_epch)
        eeg_encoder.load_state_dict(torch.load("saved_models/%s/eeg_encoder_%d.pth" % (dataset.get_name(), start_epch)))
        img_encoder.load_state_dict(torch.load("saved_models/%s/img_encoder_%d.pth" % (dataset.get_name(), start_epch)))


####################
# Train the model  #
####################
load_model(EPCH_START)
prev_time = time.time()
for epch in range(EPCH_START, EPCH + 1):
    for i, (eeg, label, stim) in enumerate(dat_loader):
        eeg_encoder.train()
        img_encoder.train()

        eeg_optim.zero_grad()
        img_optim.zero_grad()

        # Data transforming
        eeg = eeg.unsqueeze(1)

        eeg_latent = eeg_encoder(eeg)
        img_latent = img_encoder(stim)

        loss_latent = criterion(eeg_latent, img_latent)
        loss_latent.backward()
        eeg_optim.step()
        img_optim.step()

        # Determine approximate time left
        batches_done = epch * len(dat_loader) + i
        batches_left = EPCH * len(dat_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [latent_MSE %f] ETA: %s"
            % (
                epch,
                EPCH,
                i,
                len(dat_loader),
                loss_latent.item(),
                time_left,
            )
        )

    lr_sched_eeg_optim.step()
    lr_sched_img_optim.step()

    if CHCK_PNT_INTERVAL != -1 and epch % CHCK_PNT_INTERVAL == 0:
        # Save model checkpoints
        torch.save(eeg_encoder.state_dict(), "saved_models/%s/eeg_encoder_%d.pth" % (dataset.get_name(), epch))
        torch.save(img_encoder.state_dict(), "saved_models/%s/img_encoder_%d.pth" % (dataset.get_name(), epch))
