import os
import logging
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 

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
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=CFG.n_mels,
        mel_scale="htk",
    )
    return mel_spectrogram


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def get_logger(log_name):
    if(not os.path.exists("logs")):
        os.mkdir("loygs")
    log_path = "logs/"+log_name
    with open(log_path, "w") as file:
        file.write('[Log Created!]\n')

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
