import numpy as np
import numpy.random as np_rand
import torch
import matplotlib.pyplot as plt

from model.module.wavelet_func import *

x = torch.linspace(-5, 5, 100)
x1 = torch.sin(x-1.2)
a1 = mexican_hat(x1)

x2 = torch.cos(1.4*x) + 0.3*torch.sin(0.2*x)
a2 = mexican_hat(x2)

a_final = a1*a2

plt.plot(x)
plt.plot(a_final)
plt.show()
