import torch
from model.module.wavelet_func import k1d_build
from torch.nn import Parameter
import torch.nn.functional as F


class MultiDimWN(torch.nn.Module):
    """
    This class followed the implementation on Antonios K. et al (Wavelet neural networks: A practical guide)
    """

    def __init__(self, in_channel, hidden_unit, mother_wavelet):
        """
        :param in_channel: Number of input channel
        :param hidden_unit: Number of hidden units
        """
        super(MultiDimWN, self).__init__()

        # model config var
        self.in_ch = in_channel
        self.h_u_num = hidden_unit
        self.m_wvl = mother_wavelet

        # model parameter
        self.bias = Parameter(torch.Tensor([0]), requires_grad=True)

        self.w_wvl = Parameter(torch.ones(1, 1, self.h_u_num, 1), requires_grad=True)  # Wavelet weight
        self.w_lin = Parameter(torch.zeros(1, self.in_ch, 1), requires_grad=True)  # Linear weight

        self.w_trn = Parameter(torch.zeros(1, self.in_ch, self.h_u_num, 1), requires_grad=True)  # Translation param
        self.w_dil = Parameter(torch.ones(1, self.in_ch, self.h_u_num, 1),
                               requires_grad=True)  # Dilation param (This should be one to prevent inf)

    def forward(self, x):
        """
        The calculation are follow the mathematics expression below
        y^ = bias + sum(w_wvl * prod( phi( (x-w_trn)/w_dil ) ) ) + sum(w_lin*x)
        :param x: Expected shape [BS, IN_CHANNEL, LENGTH]
        :return: output in shape [BS, 1, LENGTH]
        """
        x_bs = x.shape[0]
        x_len = x.shape[2]

        x_xp = x.unsqueeze(2).expand(x_bs, self.in_ch, self.h_u_num, x_len)
        x_wvl = (x_xp - self.w_trn) / self.w_dil
        hu = self.w_wvl * torch.prod(self.m_wvl(x_wvl), dim=1, keepdim=True)
        sum_hu = torch.sum(hu, dim=2, keepdim=True)

        lin = torch.sum(self.w_lin * x, dim=1, keepdim=True).unsqueeze(2)  # Expand HU dim

        x = (self.bias + sum_hu + lin).squeeze(2)  # Now destroy HU dim

        return x


def _param_init(out_ch: int, in_ch: int, l: int, scale, offset=0.5) -> torch.Tensor:
    return (torch.rand(out_ch, in_ch, l) * scale) + torch.Tensor([offset])


class FlexMultiDimWCN(torch.nn.Module):
    """
    aka Flexible Multiple Dimension Wavelet "Convolution" Network (FMDWCN)
    For this layer, we do convolution operation instead of
    """

    def __init__(self, in_ch, out_ch, k_s, k_r, mother_wavelet, pd=None):
        """

        :param in_ch: Number of input channel
        :param out_ch: Number of output channel
        :param k_s: size of the kernel (Number of array)
        :param k_r: range of the wavelet kernel
        :param mother_wavelet: Mother wavelet function
        :param pd: Padding size. Leave this as None to make a same len output
        """
        super(FlexMultiDimWCN, self).__init__()

        self.pd = pd  # The calculation will do on forward method

        assert k_s % 2 != 0, "Kernel size should be odd number"

        self.k_s = k_s
        self.k_r = k_r
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mt_wvl = mother_wavelet

        self.bias = Parameter(torch.Tensor([0]), requires_grad=True)
        self.w_wvl = Parameter(_param_init(out_ch, in_ch, 1, 1.5), requires_grad=True)
        self.w_lin = Parameter(_param_init(1, self.in_ch, 1, 0.5, 0.8), requires_grad=True)

        self.w_trn = Parameter(_param_init(out_ch, in_ch, 1, scale=0, offset=0), requires_grad=True)
        self.w_dil = Parameter(_param_init(out_ch, in_ch, 1, scale=0, offset=1), requires_grad=True)

        self.k_linspace = k1d_build(self.k_s, self.k_r, self.in_ch, self.out_ch)
        # Translation and dilation will be compute in forward method

    def forward(self, x):
        """
        :param x: Expected shape [BS, CH, LEN]
        :return:
        """
        k = self.mt_wvl(self.k_linspace)
        k = (k - self.w_trn) / self.w_dil
        k = k * self.w_wvl
        if self.pd is None:
            self.pd = int(self.k_s / 2)
        wvl = F.conv1d(x, k, padding=self.pd)
        lin = torch.sum(self.w_lin * x, dim=1, keepdim=True)
        return self.bias + wvl + lin


class FlexMultiDimWN(torch.nn.Module):
    """
    This class expected to process the signal data type. Not an image one.
    Compare the the MultiDimWN class, this one is make it more flexible to construct the model
    (number of output channel now customizable, eliminated hidden units)
    """

    def __init__(self, in_channel, hidden_unit, mother_wavelet):
        """
        :param in_channel: Number of input channel
        :param hidden_unit: Number of hidden units
        """
        super(FlexMultiDimWN, self).__init__()

        # model config var
        self.in_ch = in_channel
        self.h_u_num = hidden_unit
        self.m_wvl = mother_wavelet

        # model parameter
        self.bias = Parameter(torch.Tensor(torch.zeros(1, self.h_u_num, 1)), requires_grad=True)

        self.w_wvl = Parameter(torch.ones(1, self.in_ch, self.h_u_num, 1), requires_grad=True)  # Wavelet weight
        self.w_lin = Parameter(torch.ones(1, self.in_ch, self.h_u_num, 1), requires_grad=True)  # Linear weight

        self.w_trn = Parameter(torch.zeros(1, self.in_ch, self.h_u_num, 1), requires_grad=True)  # Translation param
        self.w_dil = Parameter(torch.ones(1, self.in_ch, self.h_u_num, 1),
                               requires_grad=True)  # Dilation param (This should be one to prevent inf)

    def forward(self, x):
        """
        The calculation are follow the mathematics expression below
        y^ = bias + sum(w_wvl * prod( phi( (x-w_trn)/w_dil ) ) ) + sum(w_lin*x)
        :param x: Expected shape [BS, IN_CHANNEL, LENGTH]
        :return: output in shape [BS, HIDDEN_UNIT_NUM, LENGTH]
        """
        x_bs = x.shape[0]
        x_len = x.shape[2]

        x_xp = x.unsqueeze(2).expand(x_bs, self.in_ch, self.h_u_num, x_len)
        x_wvl = (x_xp - self.w_trn) / self.w_dil
        sum_hu = torch.sum(self.w_wvl * self.m_wvl(x_wvl), dim=1)

        lin = torch.sum(self.w_lin * x_xp, dim=1)  # Expand HU dim

        x = (self.bias + sum_hu + lin)  # Now destroy HU dim

        return x
