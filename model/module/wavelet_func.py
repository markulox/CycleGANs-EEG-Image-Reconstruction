import torch


def k1d_build(k_s, k_r, in_ch, out_ch):
    """
    This function output the simple linspace kernel within specific range
    before passing to the mother wavelet function. The shape format of kernel is
    [h_u, in_ch, k_s]
    :param k_s: kernel size, How many steps being created via linspace
    :param k_r: (Kernel Range) Range of the wavelet kernel
    :param in_ch: Input channel
    :param out_ch: Number of output channel
    :return: The translated and dilated linspace tensor
    """
    start, end = -1 * (k_r / 2), (k_r / 2)
    k = torch.linspace(start, end, k_s).reshape(1, 1, -1)
    k = k.expand(out_ch, in_ch, -1)
    return k


def mexican_hat(z):  # Tested and yeah... I think its correct
    """
    :param z: a linspace param
    :return:
    """
    return (1 - torch.pow(z, 2)) * torch.exp(torch.Tensor([-0.5]) * torch.pow(z, 2))


def partial_sine(z):
    return
