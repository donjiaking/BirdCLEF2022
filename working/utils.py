import os
import logging
import random
from time import strftime
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import CFG


def load_model(model, model_name):
    model.load_state_dict(torch.load(CFG.model_out_path+f"{model_name}.pt"))


def save_model(model, model_name):
    if(not os.path.exists(CFG.model_out_path)):
        os.makedirs(CFG.model_out_path)
    torch.save(model.state_dict(), CFG.model_out_path+f"{model_name}.pt")


def fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def plot_history(history, model_name):
    plt.plot(history[:,0], history[:,1], label='train_loss')
    plt.plot(history[:,0], history[:,2], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.savefig(f"plots/loss_{model_name}.png")
    plt.clf()

    plt.plot(history[:,0], history[:,3])
    plt.xlabel('epoch')
    plt.ylabel('macro f1 score')
    plt.title(f"{model_name} Macro F1 Score on Validation Set")
    plt.savefig(f"plots/f1_{model_name}.png")


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


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def channel_norm(x):
    """per-channel normalization"""
    mean = x.mean((1, 2), keepdim=True)
    std = x.std((1, 2), keepdim=True)
    x = (x - mean) / (std + 1e-7) 
    return x 


def get_logger(log_name):
    if(not os.path.exists("logs")):
        os.mkdir("logs")
    log_path = "logs/"+log_name
    with open(log_path, "w") as file:
        file.write(f"[Log Created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    file_handler = logging.FileHandler(log_path, mode='a', encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)
    
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger


def write_tensorboard(log_name, train_loss, val_loss, val_f1, epoch=0):
    tb_path = "tb_logs/" + log_name
    if(not os.path.exists(tb_path)):
        os.makedirs(tb_path)
    else:
        shutil.rmtree(tb_path)  # clear prev records
        os.makedirs(tb_path)

    writer = SummaryWriter(tb_path)

    if train_loss:
        writer.add_scalar("train_loss",train_loss, epoch)
    if val_loss:
        writer.add_scalar("val_loss", val_loss, epoch)
    if val_f1:
        writer.add_scalar("val_f1", val_f1, epoch)

    writer.close()
