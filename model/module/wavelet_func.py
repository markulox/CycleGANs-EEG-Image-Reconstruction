import torch


def kernel_build(k_s, start, end):
    return torch.linspace(start, end, k_s)


def mexican_hat(z, trn=1, dil=1):  # Tested and yeah... I think its correct
    #z = (z - trn) / dil
    return (1 - torch.pow(z, 2)) * torch.exp(torch.Tensor([-0.5]) * torch.pow(z, 2))
