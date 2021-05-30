import torch


def kernel_build(k_s, start, end, in_ch, trn, dil, bs=1):
    k = torch.linspace(start, end, k_s).expand(bs, )
    return k


def mexican_hat(z):  # Tested and yeah... I think its correct
    return (1 - torch.pow(z, 2)) * torch.exp(torch.Tensor([-0.5]) * torch.pow(z, 2))
