import os
import numpy as np
import torch

from torchvision.utils import make_grid, save_image
from itertools import combinations


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > gpu_free')
    memory_available = [int(x.split()[2]) for x in open('gpu_free', 'r').readlines()]
    gpu = f'cuda:{np.argmax(memory_available)}'
    # if os.path.exists("gpu_free"):
    #     os.remove("gpu_free")
    # else:
    #       print("The file does not exist")
    # print(get_freer_gpu())
    return gpu


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def clean_string(string):
    for ch in ['(', ')', '[', ']', "'", ' ']:
        string = string.replace(ch, "")
    return string


def string_to_list(string):
    string = clean_string(string)
    words = string.split(",")
    print(words)
    return words


def squeeze_tensor_to_list(_tmp):
    from functools import reduce
    import operator

    xx = [i.cpu().detach().numpy().ravel().tolist() for i in _tmp]
    xx = reduce(operator.concat, xx)
    return xx


import csv, os


def save_result_csv(_header_name, _row_data, _path):
    filename = _path
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(f"{filename}", mode) as myfile:
        fileEmpty = os.stat(filename).st_size == 0
        writer = csv.DictWriter(myfile, delimiter='|', lineterminator='\n', fieldnames=_header_name)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        row_dic = dict(zip(_header_name, _row_data))
        writer.writerow(row_dic)
        print(f"....save file to {filename} success")
        myfile.close()


# ====================================================================
def get_unique_combination(_elements):
    # start_time   = time.time()
    _result = []
    for _loop in range(1, len(_elements)):
        _comb = combinations(_elements, _loop)
        for _ele in list(_comb):
            # print(f'_{_ele}')
            _result.append(list(_ele))
    _result.append(_elements)

    # if _is_debug:
    #     print(f'time_trace of {inspect.stack()[0][3]} = {time.time()-start_time}')

    return _result


def save_img(real_img, fake_img, export_path, item_num=10, epoch=0):
    if real_img.shape[0] < item_num:
        item_num = real_img.shape[0]

    fake_stim = make_grid(fake_img[0:item_num, :, :, :], nrow=5, normalize=True)
    # Arange images along y-axis
    real_stim = make_grid(real_img[0:item_num, :, :, :], nrow=5, normalize=True)
    image_grid = torch.cat((real_stim, fake_stim), 1)
    save_image(image_grid, export_path + "%05d_reduce.png" % epoch, normalize=False)


def acc_calc(pred_l, real_l):
    """
    input expected in one hot encoded
    """
    p_l = torch.argmax(pred_l, axis=1)
    r_l = torch.argmax(real_l, axis=1)
    return (torch.sum(p_l == r_l).item() / float(p_l.shape[0])) * 100.0
