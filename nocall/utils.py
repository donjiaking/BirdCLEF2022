import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

# import cv2
# from PIL import Image

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

import warnings 
warnings.filterwarnings('ignore')

from config import CFG


"""def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)  # prediction accuracy


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def channel_norm(x):
    """per-channel normalization"""
    mean = x.mean((1, 2), keepdim=True)
    std = x.std((1, 2), keepdim=True)
    x = (x - mean) / (std + 1e-7) 
    return x


def get_mel_transform():
    mel_spectrogram = T.MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.n_fft,
            win_length=CFG.win_length,
            hop_length=CFG.hop_length,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            pad=0,
            n_mels=CFG.n_mels,
            power=CFG.power,
            normalized=False,
        )
    return mel_spectrogram


@contextmanager
def timer(name):
    t0 = time.time()
    # LOGGER.info(f'[{name}] start')
    print("[{%s}] start" % name)
    yield
    # LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')
    print('[{%s}] done in {%.0f} s.' % (name, time.time() - t0))


"""def init_logger(log_file='./train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()"""


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_result(result_df):    
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    score = get_score(labels, preds)
    # LOGGER.info(f'Score: {score:<.5f}')
    print('Score: {%.5f}' % score)


def get_confusion_mat(result_df):
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    matrix = get_confusion_matrix(labels, preds)
    print('TN', matrix[0,0])
    print('FP', matrix[0,1])
    print('FN', matrix[1,0])
    print('TP', matrix[1,1])

