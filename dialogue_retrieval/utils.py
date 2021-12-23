# MIT License
# code by Soohwan Kim @sooftware

import os
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
