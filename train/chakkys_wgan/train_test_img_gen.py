import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader
from model import Generator, Discriminator2
from model_lib.semi_supervised.model import SemanticImageExtractorV2
from dataset.EEGImageNet_Spam_et_al import UnpairedStimuliDataset
from config import *
from libs.model_utils import weights_init, gradient_penalty
from libs.utilities import save_img, acc_calc
from libs.eeg_features import tsne2d, isomap2d, scater_plot

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % UnpairedStimuliDataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "export_content/images/%s/" % UnpairedStimuliDataset.get_name())

# Set random seeds
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Check device type
print("<I> Device =", device)

# Dataset declaration
num_classes = UnpairedStimuliDataset.NUM_CLASSES
up_ds = UnpairedStimuliDataset(dev=device)
up_ld = DataLoader(up_ds, batch_size=batch_size, shuffle=True)
# - Check the correctness of dataset
up_ds.set_train()
tr_len = len(up_ld)
up_ds.set_test()
ts_len = len(up_ld)
print("<I> Training len ", tr_len, ", Testing len ", ts_len)

# Model declaration
add_proxy()
netIE = SemanticImageExtractorV2(num_classes, feature_size, pretrain=True).to(device)
remove_proxy()
netD = Discriminator2(ngpu, num_classes).to(device)
netG = Generator(ngpu, num_classes).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netG = nn.DataParallel(netG, list(range(ngpu)))

# - Init model with some weight or resume from previous training
if load_at_epoch != 0:
    if LOAD_GEN:
        netG.load_state_dict(torch.load(MODEL_PATH + "%d_G.pth" % load_at_epoch))
    if LOAD_DIS:
        netD.load_state_dict(torch.load(MODEL_PATH + "%d_D.pth" % load_at_epoch))
else:
    netD.apply(weights_init)
    netG.apply(weights_init)
# - Load model

# Optimizer declaration
optim_IE = optim.SGD(netIE.parameters(), lr=lr, momentum=0.9)
optim_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))
optim_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))

# Criterion declaration
criterion = nn.CrossEntropyLoss()

# Training loop
print("<I> Training info")
print("\tEpoch = ", num_epochs)

iters = 0
G_losses = []
D_losses = []
I_losses = []
for epoch in range(num_epochs):
    for i, (img, label) in enumerate(up_ld):
        label_digit = torch.argmax(label, dim=1).to(device)
        curr_bs = img.shape[0]

        # Train netIE
        for _ in range(ie_epoch_steps):
            latent, pred_lab = netIE(img)
            ie_loss = criterion(pred_lab, label_digit)

            optim_IE.zero_grad()
            ie_loss.backward()
            optim_IE.step()

        ie_acc = acc_calc(pred_lab, label)

        # Train netD
        for _ in range(d_epoch_steps):
            latent, pred_lab = netIE(img)
            pred_lab_digit = torch.argmax(pred_lab, dim=1)

            zz = torch.randn(curr_bs, z_dim, 1, 1, device=device)
            d_real_pred = netD(img, latent, pred_lab_digit).view(-1)  # Todo: try pred_lab_digit -> label_digit
            fake_img = netG(zz, pred_lab_digit, latent)
            d_fake_pred = netD(fake_img, latent, pred_lab_digit).view(-1)

            gp = gradient_penalty(netD, latent, pred_lab_digit, img, fake_img, device)
            d_loss = (
                    -(torch.mean(d_real_pred) - torch.mean(d_fake_pred)) + lambda_gp * gp
            )

            optim_D.zero_grad()
            d_loss.backward(retain_graph=True)
            optim_D.step()

        for _ in range(g_epoch_steps):
            latent, pred_lab = netIE(img)
            pred_lab_digit = torch.argmax(pred_lab, dim=1)

            zz = torch.randn(curr_bs, z_dim, 1, 1, device=device)
            fake_img = netG(zz, pred_lab_digit, latent)
            d_pred = netD(fake_img, latent, pred_lab_digit).view(-1)

            g_loss = -torch.mean(d_pred)
            optim_G.zero_grad()
            g_loss.backward(retain_graph=True)
            optim_G.step()

        I_losses.append(ie_loss.item())
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        sys.stdout.write("\r" + str(epoch) + "/" + str(num_epochs) + "; batch: " + str(
            i) + "/" + str(tr_len) + "; G Loss: %.04f; D Loss: %.04f; IE Loss: %.04f; IE Acc: %.02f%%" % (
                             g_loss.item(), d_loss.item(), ie_loss.item(), ie_acc))

        # Export image stuff
        if epoch % (num_epochs // 100) == 1:
            netG.eval()
            netIE.eval()
            up_ds.set_test()
            val_img, val_label = next(iter(up_ld))
            with torch.no_grad():
                val_label_digit = torch.argmax(val_label, dim=1)
                val_bs = val_img.shape[0]

                val_latent, val_pred_lab = netIE(val_img)
                val_pred_lab_digit = torch.argmax(val_pred_lab, dim=1)

                val_zz = torch.randn(val_bs, z_dim, 1, 1, device=device)
                val_fake_img = netG(val_zz, val_pred_lab_digit, val_latent).detach().cpu()
                val_img_detach = val_img.detach().cpu()

                save_img(val_img_detach, val_fake_img, export_path=IMAGE_PATH, item_num=img_save_num, epoch=epoch)
