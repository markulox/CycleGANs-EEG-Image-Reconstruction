import sys
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from config import *
from util import *
from dataset.six_objects_1000stimuli import SixObject1KStimuliv2
from dataset.very_nice_dataset import VeryNiceDatasetv2
from model_lib.semi_supervised.model import SemanticImageExtractorV2
from model_lib.extras.EEGNet import EEGNet_Extractor
from model import Generator, Discriminator2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Dataset declaration
dataset = SixObject1KStimuliv2(dev=device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

dataset_eeg = VeryNiceDatasetv2(dev=device)
eeg_loader = DataLoader(dataset_eeg, batch_size=batch_size, shuffle=True, num_workers=workers)

eeg_sample = next(iter(eeg_loader))[0]
eeg_label_sample = next(iter(eeg_loader))[2]
img_label_sample = next(iter(dataloader))[2]
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

optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))  # <-----changed back! to RMSprop

criterion = nn.CrossEntropyLoss(reduction='mean')
tune_up_criterion = nn.MSELoss(reduction='mean')

# Training conf Loop
ie_iter = 1
ee_iter = 1
tunes_up = 1
dis_iter = 2
gen_iter = 1
img_list = []
G_losses = []
D_losses = []
I_losses = []

# ====[Training Loop]====
# torch.autograd.set_detect_anomaly(True)

print("Starting Training Loop...")
iters = 0
for epoch in range(num_epochs):
    for i, ((y_p, y_p_w, l_p, l_p_w, x_p, x_p_w), (x_u, x_u_w, l_u, l_u_w)) in enumerate(zip(eeg_loader, dataloader)):

        #         if eeg.shape[0] != real.shape[0]:
        #             continue

        l_p = torch.argmax(l_p, dim=1)
        l_u = torch.argmax(l_u, dim=1)
        l_u_float = l_u.float()

        # Move stuff to device
        x_u = x_u.to(device)
        l_u = l_u.long().to(device)
        cur_batch_size = x_u.shape[0]
        eeg_batch_size = y_p.shape[0]

        for _ in range(ie_iter):
            fx, lx_p = netIE(x_p)
            _, lx_u = netIE(x_u)

            # ie_loss_p = criterion(lx_p, l_p)
            ie_loss_u = criterion(lx_u, l_u)
            # ie_loss = ie_loss_p + ie_loss_u

            ie_loss = ie_loss_u

            optimizerIE.zero_grad()
            ie_loss.backward()
            optimizerIE.step()

        for _ in range(ee_iter):
            fy, ly_p = netEE(y_p)
            ee_loss = criterion(ly_p, l_p)

            optimizerEE.zero_grad()
            ee_loss.backward()
            optimizerEE.step()

        # Train the FE
        for _ in range(tunes_up):
            fy, ly_p = netEE(y_p)
            fx, lx_p = netIE(x_p)
            _, lx_u = netIE(x_u)
            # j1_l = j1_loss(l=eeg_label, fx=semantic_latent, fy=eeg_latent)

            # ie_loss_p = criterion(lx_p, l_p)
            # ie_loss_u = criterion(lx_u, l_u)
            # ee_loss = criterion(ly_p, l_p)
            j1_l = tune_up_criterion(fx, fy)  # + ee_loss + ie_loss_u + ie_loss_p

            optimizerEE.zero_grad()
            optimizerIE.zero_grad()
            j1_l.backward()
            optimizerEE.step()
            optimizerIE.step()

        # Train D min - (E[dis(real)] - E[dis(fake)])
        for _ in range(dis_iter):
            fx_u, lx_u = netIE(x_u)
            lx_u_num = torch.argmax(lx_u.long(), 1)

            fx_u_w, lx_u_w = netIE(x_u_w)
            lx_u_num_w = torch.argmax(lx_u_w.long(), 1)
            noise = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)

            fy_p, ly_p = netEE(y_p)
            ly_p_num = torch.argmax(ly_p.long(), 1)
            noise_eeg = torch.randn(eeg_batch_size, z_dim, 1, 1, device=device)

            # output_real = netD(real, semantic_latent, apl_dt).view(-1)   #<------we use alex_pred
            # Real stuff
            # min_D = netD(x_u, fx_u, lx_u_num).view(-1)
            min_D = torch.cat((
                netD(x_u, fx_u, lx_u_num).view(-1),
                netD(x_u_w, fx_u_w, lx_u_num_w).view(-1),
            ))

            x_u_gen = netG(noise, lx_u_num, fx_u)  # <------we use alex_pred
            x_u_gen_w = netG(noise, lx_u_num_w, fx_u_w)
            x_p_gen = netG(noise_eeg, ly_p_num, fy_p)

            fx_u_w, lx_u_w = netIE(x_u_w)
            lx_u_w_num = torch.argmax(lx_u_w.long(), 1)

            fy_p_w, ly_p_w = netEE(y_p_w)
            ly_p_w_num = torch.argmax(ly_p_w.long(), 1)
            # output_fake = netD(fake, semantic_latent, apl_dt).view(-1)  #<------we use alex_pred
            # Fake stuff
            # max_D = netD(x_u_gen, fx_u, lx_u_num).view(-1)
            max_D = torch.cat((
                netD(x_u_gen, fx_u, lx_u_num).view(-1),
                netD(x_u_gen, fx_u_w, lx_u_w_num).view(-1),  # This
                # netD(x_p_gen, fy_p, ly_p_num).view(-1),  # This
                # netD(x_p_gen, fy_p_w, ly_p_w_num).view(-1)  # This
                netD(x_u_gen_w, fx_u, lx_u_num).view(-1),
                netD(x_u_gen_w, fx_u_w, lx_u_w_num).view(-1)
            ))

            gp = gradient_penalty(netD, fx_u, lx_u_num, x_u, x_u_gen, device=device)  # <------we use alex_pred
            dis_loss = (
                    -(torch.mean(min_D) - torch.mean(max_D)) + lambda_gp * gp
            )
            optimizerD.zero_grad()
            dis_loss.backward(retain_graph=True)  # similar to detach()
            optimizerD.step()

        for _ in range(gen_iter):
            # Train G: min -E[dis(gen_fake)]
            fx_u, lx_u = netIE(x_u)
            lx_u_num = torch.argmax(lx_u.long(), 1)
            noise = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)

            fy_p, ly_p = netEE(y_p)
            ly_p_num = torch.argmax(ly_p.long(), 1)
            noise_eeg = torch.randn(eeg_batch_size, z_dim, 1, 1, device=device)

            x_u_gen = netG(noise, lx_u_num, fx_u)
            # x_p_gen = netG(noise_eeg, ly_p_num, fy_p)
            output = netD(x_u_gen, fx_u, lx_u_num).view(-1)
            # output = torch.cat((
            #     netD(x_u_gen, fx_u, lx_u_num).view(-1),
            #     netD(x_p_gen, fy_p, ly_p_num).view(-1)
            # ))
            gen_loss = -torch.mean(output)

            optimizerG.zero_grad()
            gen_loss.backward(retain_graph=True)
            optimizerG.step()

        # Save Losses for plotting later
        I_losses.append(ie_loss.item())
        G_losses.append(gen_loss.item())
        D_losses.append(dis_loss.item())

        # Check how the generator is doing by saving G's output on noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                x_u_gen = netG(noise_eeg, lx_u_num,
                               fx_u).detach().cpu()  # <- send in noise instead to make sure the label matches

            img_list.append(vutils.make_grid(x_u_gen, padding=2, normalize=True))

            # Plot the real images
            # plt.figure(figsize=(15, 15))
            # plt.subplot(1, 2, 1)
            # plt.axis("off")
            # plt.title("Real Images")
            # plt.imshow(np.transpose(vutils.make_grid(x_p.to(device)[:64], padding=5, normalize=True).cpu(),
            #                         (1, 2, 0)))

            # Plot the fake images from the last epoch
            # plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.show()

        sys.stdout.write("\rEpch:" + str(epoch) + "; Iters: " + str(
            iters) + "; Gen Loss: %.04f; Dis Loss: %.04f; IE Loss: %.04f; EE Loss: %.04f; J1 Loss: %.04f" % (
                             gen_loss.item(), dis_loss.item(), ie_loss.item(), ee_loss.item(), j1_l.item()))
        iters += 1

print("\nPlotting...")
# os.environ.pop('http_proxy')
# os.environ.pop('https_proxy')

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.plot(I_losses, label="I")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

# %%capture
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
