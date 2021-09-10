from model.semi_supervised.model import *

LOSS_MIN_CLAMP = 1e-38


def __pairwise_matmul(x, y) -> torch.Tensor:
    """
    Perform a multiplication in every object in tensor x with every object in tensor y
    :param x: A tensor
    :param y: A tensor
    :return: Element wise multiplication
    """
    ret_val = None
    for each_ix in x:
        for each_iy in y:
            res = each_ix.matmul(each_iy.transpose(0, 1)).unsqueeze(2)
            ret_val = res if ret_val is None else torch.cat((ret_val, res), 0)
    return ret_val


def j1_loss(l, fx, fy) -> torch.Tensor:
    """
    Calculate the L1 loss which I'm not sure what loss is this...
    Shape_ref = (Batch_size (depends), 1, Len (depends))

    all input expect its shape to be same as 'Shape_ref'

    :param l: A real label from dataset (Batch_size, 1, Class num)
    :param fx: Semantic features of img modality (Batch_size, 1, feature_size)
    :param fy: Semantic features of EEG modality (Batch_size, 1, feature_size)
    :return: The J1 loss value
    """

    s_ij = __pairwise_matmul(l, l)  # Do matmul first then mask it
    s_ij[s_ij > 0] = 1  # others will be 0

    lo_ij = __pairwise_matmul(fx, fy).clamp(max=88)
    loss = (s_ij * lo_ij) - torch.log((1 + torch.exp(lo_ij)).clamp(min=LOSS_MIN_CLAMP))
    loss = torch.sum(- loss)
    return loss


def j2_loss(l_real: torch.Tensor, lx: torch.Tensor) -> torch.Tensor:
    """
    Calculate J2 loss
    :param l_real: A label from dataset
    :param lx: A label from image modality
              The range of value expect to be (0,1) in other words, cannot be 0 or 1
    :return: J2 loss value
    """
    lx = lx.transpose(1, 2).clamp(min=LOSS_MIN_CLAMP, max=0.999999)
    loss = torch.matmul(l_real, torch.log(lx)) + torch.matmul((1 - l_real), torch.log((1 - lx)))
    loss = torch.sum(-loss)
    return loss


def j3_loss(l_real: torch.Tensor, ly: torch.Tensor):
    """
    Calculate J3 loss
    :param l_real: A label from dataset
    :param ly: A label from EEG modality
              The range of value expect to be (0,1) in other words, cannot be 0 or 1

    :return J3 loss value
    """
    return j2_loss(l_real, ly)


def j4_loss(fy_p, l_p, fx_u, l_u, ) -> torch.Tensor:
    """
    Calculate J4 loss
    :param l_p: paired label (might be a real label of fy_p)
    :param l_u: unpaired label (might be a real label of fx_u)
    :param fx_u: Semantic features from img modality (expected to be unpaired since we can find it on internet)
    :param fy_p: Semantic features from EEG modality (expected to be paired data)
    :return: J4 loss value
    """
    s_ij_bar = __pairwise_matmul(l_p, l_u)
    s_ij_bar[s_ij_bar > 0] = 1  # other will be 0

    lo_ij_bar = __pairwise_matmul(fy_p, fx_u).clamp(max=88)
    loss = (s_ij_bar * lo_ij_bar) - torch.log((1 + torch.exp(lo_ij_bar)).clamp(min=LOSS_MIN_CLAMP))
    loss = torch.sum(-loss)
    return loss


def j5_loss(l_real_u: torch.Tensor, lx_u: torch.Tensor):
    """
    Calculate J5 loss
    :param l_real_u: is unpaired image data
    :param lx_u: a label from unpaired image modal
    :return: J5 loss value
    """
    return j2_loss(l_real_u, lx_u)


# # s_ij expected to be shape (10,1,1)
# i = torch.rand((10, 1, 10)) - 0.5
# i_real = F.one_hot(i.argmax(dim=0), 10).transpose(0, 1).type(torch.DoubleTensor)
# i_predict = F.one_hot(i.argmax(dim=0), 10).transpose(0, 1).type(torch.DoubleTensor)
#
# i_predict[i_predict == 0] = 0.000001
# i_predict[i_predict == 1] = 0.999999
# fx = torch.rand((10, 1, 200))
# fy = torch.rand((10, 1, 200))
#
# loss_j1 = j1_loss(i, fx, fy)
# print(loss_j1)
# print(j2_3_loss(i_real, i_predict))


def l1_loss(d1: D1, x_p, x_p_gen: torch.Tensor, train_gen=False) -> torch.Tensor:
    """
    This is function calculate the loss value
    :param d1: The discriminator (D1) model object.
    :param x_p: The image that expected to be from real dataset.
    :param x_p_gen: Expected to be generated by Generator with EEG semantic and label from EEG Extractor.
    :param train_gen: Set this to be True when you want to train the generator
    :return: l1 loss if x and x_gen from paired data.
    """
    if train_gen:
        d1.eval()
    else:
        d1.train()

    real_pair = d1.forward(x_p, x_p).clamp(min=LOSS_MIN_CLAMP, max=0.999999)
    fake_pair = d1.forward(x_p, x_p_gen).clamp(min=LOSS_MIN_CLAMP, max=0.999999)

    loss = torch.log(real_pair) + torch.log(1 - fake_pair)
    loss = torch.sum(loss)
    return loss


def l3_loss(d1: D1, x_u, x_u_gen, train_gen=False) -> torch.Tensor:
    """
    Calculate l3 loss
    :param d1:
    :param x_u: The image that expected to be from real (unpaired) dataset.
    :param x_u_gen: Expected to be generated by Generator with IMG semantic (unpaired)
                   and label from IMG Extractor (unpaired).
    :param train_gen: Set this to be True when you want to train the generator
    :return: l3 loss
    """
    return l1_loss(d1, x_u, x_u_gen, train_gen)


def l2_loss(d2: D2, x_p, x_p_gen, train_gen=False) -> torch.Tensor:
    """
    This function calculate the l2 loss value
    :param d2: The discriminator (D2) model object
    :param x_p: The real image
    :param x_p_gen: The generated image
    :param fy_p: semantic features from EEG modality
    :param ly_p: label from EEG modality
    :param train_gen: Set this to be True when you want to train the generator
    :return: l2 loss
    """
    if train_gen:
        d2.eval()
    else:
        d2.train()
    d2_x_p = d2.forward(x_p).clamp(min=LOSS_MIN_CLAMP, max=0.999999)
    d2_x_p_gen = d2.forward(x_p_gen).clamp(min=LOSS_MIN_CLAMP, max=0.999999)
    loss = torch.log(d2_x_p) + torch.log(1 - d2_x_p_gen)
    loss = torch.sum(loss)
    return loss


def l4_loss(d2: D2, x_u, x_u_gen, train_gen=False) -> torch.Tensor:
    """
    Calculate l4 loss
    :param d2: The discriminator (D2) model object
    :param x_u: The real image (unpaired)
    :param x_u_gen: The generated image from Generator with IMG semantic (unpaired)
                   and label from IMG Extractor (unpaired).
    :param fx_u: semantic features from IMG modality
    :param lx_u: label from IMG modality
    :param train_gen: Set this to be True when you want to train the generator
    :return: l4 loss
    """
    return l2_loss(d2, x_u, x_u_gen, train_gen)