import matplotlib.pyplot as plt
import numpy as np
import torch

from model.module.wavelet_func import mexican_hat

x_np = np.linspace(-10, 10, num=800)
noise_np = np.random.rand(200)

x = torch.linspace(-10, 10, 800)
noise = torch.rand(200)


def mexican_hat_np(z):
    return (1 - np.power(z, 2)) * np.power(np.e, -0.5 * np.power(z, 2))


def sine(z):
    freq = [10, 5, 2.5, 1.25]
    s = None
    for f in freq:
        if s is None:
            s = torch.sin(f * z)
        else:
            s += torch.sin(f * z)
    return s


def sine_np(z):
    freq = [10, 5, 2.5, 1.25]
    s = None
    for f in freq:
        if s is None:
            s = np.sin(f * z)
        else:
            s += np.sin(f * z)
    return s


basic_wave = sine(x)
basic_wave_np = sine_np(x_np)
print(noise.shape)
res = mexican_hat(basic_wave)
res_np = mexican_hat_np(basic_wave_np)
print(res)

plt.plot(basic_wave.detach().numpy())
plt.plot(basic_wave_np)
plt.plot(res.detach().numpy())
plt.plot(res_np)
plt.show()
