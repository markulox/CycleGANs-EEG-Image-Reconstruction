import torch
from torch.nn import Parameter


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


class FlexMultiDimWN(torch.nn.Module):
    """
    This class expected to process the signal data type. Not an image one.
    Compare the the MultiDimWN class, this one is make it more flexible to construct the model
    (number of output channel now customizable, eliminated hidden units)
    """

    def __init__(self, in_channel=1, out_channel=1, kernel_size=10):
        """
        :param in_channel: number of input channel
        :param out_channel: number of output channel (Also the number of hidden units)
        :param kernel_size: size of wavelet kernel which used in convolution
        """
        super(FlexMultiDimWN, self).__init__()
        w0 = torch.nn.parameter.Parameter()

    def forward(self, x):
        """
        :param x: Expected shape [BS, IN_CHANNEL, LENGTH]
        :return: output in shape [BS, OUT_CHANNEL, LENGTH]
        """
        return x
