import matplotlib.pyplot as plt
import torch
from model.module.wavelet_func import *
import torch.nn.functional as F

KERN_SIZE = 51
KERN_RANGE = 9.5
IN_CHANNEL = 1
HIDDEN_UNIT = 1

BS = 4
SIGNAL_LEN = 160

trn = torch.zeros(HIDDEN_UNIT, IN_CHANNEL, 1)
dil = torch.ones(HIDDEN_UNIT, IN_CHANNEL, 1)

k = k1d_build(KERN_SIZE, KERN_RANGE, IN_CHANNEL, HIDDEN_UNIT)
krn = mexican_hat(k)

x = torch.linspace(-30, 30, 323)
#x = (torch.cos(0.4 * x) + 0.2 * torch.sin(0.2 * x))
# x = x + mexican_hat(x)
#x = x + (torch.cos(0.125 * x) * torch.sin(3 * x))
x = x.reshape(1, 1, -1)
plt.plot(krn.detach().numpy().reshape(krn.shape[2]))
plt.show()

a = F.conv1d(x, krn)

plt.plot(x.detach().numpy().reshape(x.shape[2]), label="x")
plt.plot(a.detach().numpy().reshape(a.shape[2]), label="a")
plt.legend()
plt.show()

print(a.shape)
