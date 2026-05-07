import os
import random
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu=0):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu))
    else:
        device = torch.device('cpu')
    return device

