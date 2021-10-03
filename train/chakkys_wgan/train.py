import sys
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from config import *
from util import *
from dataset.six_objects_1000stimuli import SixObject1KStimuli
from dataset.very_nice_dataset import VeryNiceDataset
from model.semi_supervised.model import SemanticImageExtractorV2
from model.extras.EEGNet import EEGNet_Extractor
from model import Generator, Discriminator2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Dataset declaration
dataset = SixObject1KStimuli(dev=device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

dataset_eeg = VeryNiceDataset(dev=device)
eeg_loader = DataLoader(dataset_eeg, batch_size=batch_size, shuffle=True, num_workers=workers)

eeg_sample = next(iter(eeg_loader))[0]
eeg_label_sample = next(iter(eeg_loader))[1]
img_label_sample = next(iter(dataloader))[1]
sample_len = eeg_sample.shape[2]
channel_num = eeg_sample.shape[1]

assert eeg_label_sample.shape[1] == img_label_sample.shape[1], "2 Dataset label has not a same shape"
num_classes = img_label_sample.shape[1]

# =====[Model initialization]====
# Also apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netIE = SemanticImageExtractorV2(output_class_num=num_classes, pretrain=True).to(device).apply(weights_init)
netEE = EEGNet_Extractor(in_channel=channel_num, samples=sample_len, kern_len=sample_len // 2, F1=10, F2=10, D=2,
                         nb_classes=num_classes, latent_size=feature_size).to(device).apply(weights_init)
netG = Generator(ngpu=device, num_classes=num_classes).to(device).apply(weights_init)
netD = Discriminator2(ngpu=device, num_classes=num_classes).to(device).apply(weights_init)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Training config
optimizerIE = optim.SGD(netIE.parameters(), lr=lr, momentum=0.9)
optimizerEE = optim.Adam(netEE.parameters(), lr=lr, betas=(0.0, 0.999))

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))  # <-----changed back! to RMSprop

criterion = nn.CrossEntropyLoss()
tune_up_criterion = nn.MSELoss()

# Training conf Loop
ee_epchs = 2
ie_epchs = 2

tunes_up = 3

dis_iter = 1
gen_iter = 2
img_list = []
G_losses = []
D_losses = []
I_losses = []

# ====[Training Loop]====
# torch.autograd.set_detect_anomaly(True)

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, ((eeg, eeg_label, stim), (real, labels)) in enumerate(zip(eeg_loader, dataloader)):

        #         if eeg.shape[0] != real.shape[0]:
        #             continue

        eeg_label_single = torch.argmax(eeg_label, dim=1)
        label_single = torch.argmax(labels, dim=1)
        labels_float = labels.float()

        # Move stuff to device
        real = real.to(device)
        labels = labels.long().to(device)
        cur_batch_size = real.shape[0]
        eeg_batch_size = eeg.shape[0]

        for _ in range(ie_epchs):
            semantic_latent, alex_pred = netIE(real)  # <------we use alex_pred
            ie_loss = criterion(alex_pred, label_single)

            optimizerIE.zero_grad()
            ie_loss.backward()
            optimizerIE.step()

        for _ in range(ee_epchs):
            eeg_latent, eeg_lab_pred = netEE(eeg)

            ee_loss = criterion(eeg_lab_pred, eeg_label_single)

            optimizerEE.zero_grad()
            ee_loss.backward()
            optimizerEE.step()

        # move 2 latent close together
        for _ in range(tunes_up):
            eeg_latent, eeg_lab_pred = netEE(eeg)
            semantic_latent, alex_pred = netIE(stim)

            label_onehot = F.one_hot(labels)

            # j1_l = j1_loss(l=eeg_label, fx=semantic_latent, fy=eeg_latent)
            j1_l = tune_up_criterion(semantic_latent, eeg_latent)
            optimizerEE.zero_grad()
            optimizerIE.zero_grad()
            j1_l.backward()
            optimizerEE.step()
            optimizerIE.step()

        # Train D min - (E[dis(real)] - E[dis(fake)])
        for _ in range(dis_iter):
            semantic_latent, alex_pred = netIE(real)
            apl_dt = torch.argmax(alex_pred.long(), 1)
            noise = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)

            eeg_latent, eeg_lab_pred = netEE(eeg)
            epl_dt = torch.argmax(eeg_lab_pred.long(), 1)
            noise_eeg = torch.randn(eeg_batch_size, z_dim, 1, 1, device=device)

            # output_real = netD(real, semantic_latent, apl_dt).view(-1)   #<------we use alex_pred
            output_real = torch.cat(
                (netD(real, semantic_latent, apl_dt).view(-1), netD(stim, eeg_latent, epl_dt).view(-1)))

            fake = netG(noise, apl_dt, semantic_latent)  # <------we use alex_pred
            fake_eeg_stim = netG(noise_eeg, epl_dt, eeg_latent)
            # output_fake = netD(fake, semantic_latent, apl_dt).view(-1)  #<------we use alex_pred
            output_fake = torch.cat(
                (netD(fake, semantic_latent, apl_dt).view(-1), netD(fake_eeg_stim, eeg_latent, epl_dt).view(-1)))
            gp = gradient_penalty(netD, semantic_latent, apl_dt, real, fake, device=device)  # <------we use alex_pred
            dis_loss = (
                    -(torch.mean(output_real) - torch.mean(output_fake)) + lambda_gp * gp
            )
            optimizerD.zero_grad()
            dis_loss.backward(retain_graph=True)  # similar to detach()
            optimizerD.step()

        for _ in range(gen_iter):
            # Train G: min -E[dis(gen_fake)]
            semantic_latent, alex_pred = netIE(real)
            apl_dt = torch.argmax(alex_pred.long(), 1)
            noise = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)

            eeg_latent, eeg_lab_pred = netEE(eeg)
            epl_dt = torch.argmax(eeg_lab_pred.long(), 1)
            noise_eeg = torch.randn(eeg_batch_size, z_dim, 1, 1, device=device)

            fake = netG(noise, apl_dt, semantic_latent)
            fake_eeg_stim = netG(noise_eeg, epl_dt, eeg_latent)
            # output = netD(fake, semantic_latent, apl_dt).view(-1)
            output = torch.cat(
                (netD(fake, semantic_latent, apl_dt).view(-1), netD(fake_eeg_stim, eeg_latent, epl_dt).view(-1)))
            gen_loss = -torch.mean(output)
            optimizerG.zero_grad()
            gen_loss.backward(retain_graph=True)
            optimizerG.step()

        # Save Losses for plotting later
        I_losses.append(ie_loss.item())
        G_losses.append(gen_loss.item())
        D_losses.append(dis_loss.item())

        # Check how the generator is doing by saving G's output on noise
        if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(noise, epl_dt,
                            eeg_latent).detach().cpu()  # <---send in noise instead to make sure the label matches
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            sys.stdout.write("\rEpch:" + str(epoch) + "; Iters: " + str(
                iters) + "; Gen Loss: %.04f; Dis Loss: %.04f; IE Loss: %.04f; EE Loss: %.04f; J1 Loss: %.04f" % (
                                 gen_loss.item(), dis_loss.item(), ie_loss.item(), ee_loss.item(), j1_l.item()))
        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.plot(I_losses,label="I")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
