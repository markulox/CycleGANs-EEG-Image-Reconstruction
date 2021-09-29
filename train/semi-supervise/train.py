import torch
import os
import sys
import time
import datetime
import math

from dataset.mind_big_data import MindBigData
from dataset.six_objects_1000stimuli import SixObject1KStimuli
from dataset.cylinder_rgb import Cylinder_RBG_Dataset
from dataset.EEGImageNet_Spam_et_al import EEGImageNetDataset_Spam
from model.semi_supervised.loss_func import *  # The model already import in loss_func.py file
from torch.utils.data import DataLoader
from model.extras.EEGNet import EEGNet_Extractor
from dataset.very_nice_dataset import VeryNiceDataset

from config import *

from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import numpy as np
from utils import WeightClipper, weights_init

# Set anomaly detection while doing back-prop
# torch.autograd.set_detect_anomaly(True)

# Dataset initialization
# dataset_paired = MindBigData(dev=DEV)
dataset_paired = VeryNiceDataset(dev=DEV)
dataset_unpaired = SixObject1KStimuli(dev=DEV)

# dataset_unpaired = _____ รอไปก่อนน้ะ


# dataset = Cylinder_RBG_Dataset(dev=DEV)
# dataset2 = EEGImageNetDataset_Spam(dev=DEV, sample_max_idx=380)
# val_dataset = Cylinder_RBG_Dataset(dev=DEV)
val_dataset = MindBigData(dev=DEV)

dat_loader_paired = DataLoader(dataset=dataset_paired, batch_size=BS, shuffle=True)
dat_loader_unpaired = DataLoader(dataset=dataset_unpaired, batch_size=BS_UNPAIRED, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BS, shuffle=True)

# Model initialization
eeg_sample = next(iter(dat_loader_paired))[0]
sample_len = eeg_sample.shape[2]
channel_num = eeg_sample.shape[1]

NUM_LIM_CLASS = 6
EXPORT_DISABLE = False

sx = SemanticImageExtractor(output_class_num=NUM_LIM_CLASS,
                            feature_size=feature_size).to(DEV)
# Argument expected_shape : send some sample data to let model determine its structure
# sy = SemanticEEGExtractor(expected_shape=input_sample,
#                           output_class_num=NUM_LIM_CLASS,
#                           feature_size=feature_size).to(DEV)
sy = EEGNet_Extractor(in_channel=channel_num,
                      samples=sample_len,
                      kern_len=sample_len // 2,
                      F1=10,
                      F2=10,
                      D=2,
                      nb_classes=NUM_LIM_CLASS,
                      latent_size=feature_size).to(DEV)

d1 = D1().to(DEV)
d2 = D2().to(DEV)
# d3 = SimpleDiscriminator().to(DEV)
# G = Generator(num_classes=NUM_LIM_CLASS).to(DEV)
input_size = NUM_LIM_CLASS + feature_size + ChakkyGenerator.EXPECTED_NOISE
G = ChakkyGenerator(input_size=input_size).to(DEV)

# Init model weight
G.apply(weights_init)
d1.apply(weights_init)
# d1.apply(weights_init)
d2.apply(weights_init)

# Model util
weight_cliper = WeightClipper(min=WEIGHT_MIN, max=WEIGHT_MAX)

# Optimizer initialization
sx_op = torch.optim.Adam(sx.parameters(), lr=mu1, betas=(0.5, 0.999))
sy_op = torch.optim.Adam(sy.parameters(), lr=mu1, betas=(0.5, 0.999))

d1_op = torch.optim.Adam(d1.parameters(), lr=mu2, betas=(0.0, 0.999))
# d1_op = torch.optim.SGD(d1.parameters(), lr=mu2, momentum=0.5)
# d1_op = torch.optim.RMSprop(d1.parameters(), lr=mu2)
d2_op = torch.optim.Adam(d2.parameters(), lr=mu2, betas=(0.0, 0.999))
# d2_op = torch.optim.SGD(d2.parameters(), lr=mu2, momentum=0.5)
# d2_op = torch.optim.RMSprop(d2.parameters(), lr=mu2)
# d3_op = torch.optim.Adam(d3.parameters(), lr=mu2, betas=(0.0, 0.999))

G_op = torch.optim.Adam(G.parameters(), lr=mu2, betas=(0.5, 0.999))

# Set some path for export stuff
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "saved_models/%s/" % dataset_paired.get_name())
IMAGE_PATH = os.path.join(__dirname__, "images/%s/" % dataset_paired.get_name())


def load_model(start_epch):
    if start_epch != 0:
        # Load pretrained models
        print("<I> : Loading model at epoch check point = %d" % start_epch)
        if LOAD_FE:
            sx.load_state_dict(torch.load(MODEL_PATH + "%d_sx.pth" % start_epch))
            sy.load_state_dict(torch.load(MODEL_PATH + "%d_sy_EEGNet.pth" % start_epch))
        if LOAD_GEN:
            G.load_state_dict(torch.load(MODEL_PATH + "%d_G.pth" % start_epch))
        if LOAD_DIS:
            d1.load_state_dict(torch.load(MODEL_PATH + "%d_d1.pth" % start_epch))
            # d2.load_state_dict(torch.load(MODEL_PATH + "%d_d2.pth" % start_epch))

        # Some visualization function


def sample_images(epch):
    """Saves a generated sample from the test set"""
    real_eeg, real_label, real_stim = next(iter(dat_loader_paired))
    real_stim, real_label = next(iter(dat_loader_unpaired))
    real_stim = real_stim[0:10, :, :, :]
    real_label = real_label[0:10, :]
    sy.eval()
    G.eval()
    # eeg_features, p_label = sy(real_eeg)
    curr_BS = real_stim.shape[0]

    # eeg_features = eeg_features.squeeze(1).detach()
    # p_label = p_label.squeeze(1).detach()

    eeg_features = torch.randn(size=(curr_BS, feature_size)).to(DEV)
    p_label = torch.randn(size=(curr_BS, NUM_LIM_CLASS)).to(DEV)

    fake_stim = G(z=torch.rand(curr_BS, G.EXPECTED_NOISE).to(DEV), semantic=eeg_features, label=p_label)
    fake_stim = make_grid(fake_stim, nrow=5, normalize=True)
    # Arange images along y-axis
    real_stim = make_grid(real_stim, nrow=5, normalize=True)
    image_grid = torch.cat((real_stim, fake_stim), 1)
    save_image(image_grid, IMAGE_PATH + "%05d_reduce.png" % epch, normalize=False)


def check_nan(chck, **log):
    if math.isnan(chck) or math.isinf(chck):
        print("<!> : NAN/INF detected --------")
        for each_elm in log:
            print(each_elm, log[each_elm], sep='=')
        print("---------------------------")
        exit()


def acc_calc(pred_l, real_l):
    """
    input expected in one hot encoded
    """
    p_l = torch.argmax(pred_l, axis=1)
    r_l = torch.argmax(real_l, axis=1)
    return (torch.sum(p_l == r_l).item() / float(p_l.shape[0])) * 100.0


# X stands for image
# Y stands for EEG

j1_hist = []
j2_hist = []
j3_hist = []

prev_time = time.time()
load_model(EPCH_START)
for epch in range(EPCH_START, EPCH_END + 1):
    batch_j1 = []
    batch_j2 = []
    batch_j3 = []
    for i, ((y_p, l_real_p, x_p), (x_u, l_real_u)) in enumerate(zip(dat_loader_paired, dat_loader_unpaired)):
        # Get some stuff first
        # Since some of batch might get reduce due to end of iteration
        curr_BS_pair = x_p.shape[0]
        curr_BS_unpair = x_u.shape[0]

        # SEMANTIC NETWORK TRAINING SECTION
        # print("UPDATING SEMANTIC EXTRACTOR")
        sx_op.zero_grad()
        sy_op.zero_grad()

        fx_p, lx_p = sx(x_p)
        fy_p, ly_p = sy(y_p)

        fx_u, lx_u = sx(x_u)  # For this dataset, I think its still make sense to do this

        j1 = j1_loss(l=l_real_p, fx=fx_p, fy=fy_p)
        j2 = j2_loss(lx_p, torch.argmax(l_real_p, dim=1))
        j3 = j3_loss(ly_p, torch.argmax(l_real_p, dim=1))
        j4 = j4_loss(fy_p=fy_p, l_p=l_real_p, fx_u=fx_u, l_u=l_real_u)
        j5 = j5_loss(lx_u, torch.argmax(l_real_u, dim=1))
        j_loss = (alp0 * j1) + (alp1 * j2) + (alp2 * j3) + (alp0 * j4) + (alp3 * j5)
        # j_loss = (alp1 * j2) + (alp2 * j3) + (alp0 * j4) + (alp3 * j5)
        # j_loss = j1 + (alp1 * j2) + (alp2 * j3)
        # check_nan(j_loss.item(), j1=j1.item(), j2=j2.item(), j3=j3.item(), j4=j4.item(), j5=j5.item())
        j_loss.backward()
        # j2.backward()
        # j3.backward()

        j2_acc = acc_calc(lx_p, l_real_p)
        j3_acc = acc_calc(ly_p, l_real_p)

        # torch.nn.utils.clip_grad_norm_(sx.parameters(), MAX_GRAD_FLOAT32)
        # torch.nn.utils.clip_grad_norm_(sy.parameters(), MAX_GRAD_FLOAT32)

        # Reshape the tensor corresponding to the generator and detach everything
        fy_p = fy_p.squeeze(1).detach()
        ly_p = ly_p.squeeze(1).detach()
        fx_u = fx_u.squeeze(1).detach()
        lx_u = lx_u.squeeze(1).detach()

        # DISCRIMINATOR TRAINING SECTION #########################################
        # print("UPDATING DISCRIM")
        for d_ex_tr in range(DIS_TRAIN_ITER):
            d1_op.zero_grad()
            d2_op.zero_grad()
            # d3_op.zero_grad()

            # noise_1 = torch.normal(mean=1, std=1, size=(curr_BS_pair, G.EXPECTED_NOISE)).to(DEV)
            # x_p_gen = G.forward(z=noise_1, semantic=fy_p, label=ly_p)
            # x_p_gen_dtch = x_p_gen.detach()
            # l1 = l1_loss_v2(d1, x_p, x_p_gen, LAMBDA_GP)
            # l2 = l2_loss_v2(d2, x_p, x_p_gen, fy_p, ly_p, LAMBDA_GP)

            # fx_u = torch.randn_like(fx_u)
            # lx_u = torch.randn_like(lx_u)

            noise_2 = torch.randn(size=(curr_BS_unpair, G.EXPECTED_NOISE)).to(DEV)
            x_u_gen = G.forward(z=noise_2, semantic=fx_u, label=lx_u)
            # x_u_gen_dtch = x_u_gen.detach()
            # lwgan = wgan_loss(d3, x_u, x_u_gen, LAMBDA_GP)
            l3 = l3_loss_v2(d1, x_u, x_u_gen, LAMBDA_GP)
            l4 = l4_loss_v2(d2, x_u, x_u_gen, fx_u, lx_u, LAMBDA_GP)

            # dl_loss = (ld1 * l1) + l2 + (ld2 * l3) + l4
            dl_loss = (ld2*l3) + l4
            # dl_loss = lwgan
            # check_nan(dl_loss.item(), dl1=l1.item(), dl2=l2.item(), dl3=l3.item(), dl4=l4.item())
            dl_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(d1.parameters(), MAX_GRAD_FLOAT32)
            # torch.nn.utils.clip_grad_norm_(d2.parameters(), MAX_GRAD_FLOAT32)
            d1_op.step()
            d2_op.step()
            # d3_op.step()

        # Apply weight clipper
        # d1.apply(weight_cliper)
        # d2.apply(weight_cliper)

        # GENERATOR TRAINING SECTION #########################################
        # print("UPDATING GENERATOR")
        G_op.zero_grad()

        # noise_1 = torch.normal(mean=1, std=1, size=(curr_BS_pair, G.EXPECTED_NOISE)).to(DEV)
        # x_p_gen = G.forward(z=noise_1, semantic=fy_p, label=ly_p)
        # l1 = l1_loss_v2(d1, x_p, x_p_gen, LAMBDA_GP, train_gen=True)
        # l2 = l2_loss_v2(d2, x_p, x_p_gen, fy_p, ly_p, LAMBDA_GP, train_gen=True)

        # fx_u = torch.randn_like(fx_u)
        # lx_u = torch.randn_like(lx_u)
        noise_2 = torch.rand(size=(curr_BS_unpair, G.EXPECTED_NOISE)).to(DEV)
        x_u_gen = G.forward(z=noise_2, semantic=fx_u, label=lx_u)
        l3 = l3_loss_v2(d1, x_u, x_u_gen, LAMBDA_GP, train_gen=True)
        l4 = l4_loss_v2(d2, x_u, x_u_gen, fx_u, lx_u, LAMBDA_GP, train_gen=True)
        # lwgan = wgan_loss(d3, x_u, x_u_gen, LAMBDA_GP, train_gen=True)

        # gl_loss = (ld1 * l1) + l2 + (ld2 * l3) + l4
        gl_loss = (ld2 * l3) + l4
        # gl_loss = lwgan
        # check_nan(gl_loss.item(), gl1=l1.item(), gl2=l2.item(), gl3=l3.item(), gl4=l4.item())
        gl_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), MAX_GRAD_FLOAT32)
        G_op.step()

        batch_j1.append(j1.item())
        batch_j2.append(j2.item())
        batch_j3.append(j3.item())

        # Logging the progress
        batches_done = epch * len(dat_loader_paired) + i
        batches_left = EPCH_END * len(dat_loader_paired) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        # print(j_loss.item(), dl_loss.item())

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [J loss: %f 1[%.4f] 2[%.4f acc=%.2f] 3[%.2f, acc=%.2f] 4[%.2f] 5[%.2f]] [D loss: %.4f] [G loss: %.4f] ETA: %s"
            % (
                epch,
                EPCH_END,
                i + 1,
                len(dat_loader_paired),
                j_loss.item() / BS,
                j1.item() / BS,
                j2.item() / BS,
                j2_acc,
                j3.item() / BS,
                j3_acc,
                j4.item() / BS,
                j5.item() / BS,
                dl_loss.item(),
                gl_loss.item(),
                time_left,
            )
        )
        # print(j_loss.item(), dl_loss.item())

        if SAMPLE_INTERVAL != -1 and epch % SAMPLE_INTERVAL == 0 and i + 1 == len(dat_loader_paired) // 2:
            sys.stdout.write("\r<I> : Sampling images...")
            sample_images(epch)

        if CHCK_PNT_INTERVAL != -1 and epch % CHCK_PNT_INTERVAL == 0 and i + 1 == len(
                dat_loader_paired) // 2 and not EXPORT_DISABLE:
            # Save model checkpoints
            sys.stdout.write("\033[K \rExporting model...")
            torch.save(G.state_dict(), MODEL_PATH + "%d_G.pth" % epch)
            torch.save(d1.state_dict(), MODEL_PATH + "%d_d1.pth" % epch)
            # torch.save(d2.state_dict(), MODEL_PATH + "%d_d2.pth" % epch)
            torch.save(sx.state_dict(), MODEL_PATH + "%d_sx.pth" % epch)
            torch.save(sy.state_dict(), MODEL_PATH + "%d_sy_EEGNet.pth" % epch)

    j1_hist.append(np.mean(batch_j1) / BS)
    j2_hist.append(np.mean(batch_j2) / BS)
    j3_hist.append(np.mean(batch_j3) / BS)

plt.plot(j1_hist, label="j1")
plt.legend()
plt.show()

plt.plot(j2_hist, label="j2")
plt.plot(j3_hist, label="j3")
plt.legend()
plt.show()
