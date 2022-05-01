import math
import pandas as pd
import numpy as np
import colorednoise as cn
import random
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchaudio

from config import CFG
import utils

UNIFORM = 0
GAUSSIAN = 1
PINK_NOISE = 2


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)

def noise_injection(y: np.ndarray):
    noise_level = np.random.uniform(CFG.noise_level[0], CFG.noise_level[1])
    noise = np.random.randn(len(y))
    augmented = (y + noise * noise_level).to(y.dtype)
    return augmented

def gaussian_noise(y: np.ndarray):
    noise = np.random.normal(CFG.mean, CFG.sigma, y.shape)
    augmented = (y + noise).to(y.dtype)
    return augmented

def pink_noise(y: np.ndarray):
    noise = cn.powerlaw_psd_gaussian(1, len(y))
    augmented = (y + noise).to(y.dtype)
    return augmented

def pitch_shift_spectrogram(y: np.ndarray):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = y.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    print("max_shifts:", max_shifts)
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    augmented = np.roll(y, nb_shifts, axis=0).to(y.dtype)

    return augmented


class MyDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        self.mode = mode
        self.df = df
        self.mel_transform = utils.get_mel_transform()

        train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')
        self.all_bird = train_meta["primary_label"].unique()

        self.duration = CFG.segment_train if (self.mode=='train') else CFG.segment_test

    def __getitem__(self, index):
        row = self.df.iloc[index]

        label = [row['primary_label']] + eval(row['secondary_labels'])
        label_all = np.zeros(CFG.n_classes)
        for label_temp in label:  
            label_all += (label_temp == self.all_bird)
        label_all = np.clip(label_all, 0, 1)

        waveform, _ = torchaudio.load(filepath=CFG.input_path+row['filename'])
        len_wav = waveform.shape[1]
        waveform = waveform[0, :].reshape(1, len_wav)  # stereo->mono mono->mono
        chunks = math.ceil(len_wav / self.duration)
        end_times = [(i+1)*self.duration for i in range(chunks)]

        if(self.mode == 'train'):
            end_time = random.choice(end_times)

            if(end_time > len_wav):
                pad_len = end_time - len_wav
                waveform_chunk = F.pad(waveform, (0, pad_len))[0,end_time-self.duration:end_time]
            else:
                waveform_chunk = waveform[0,end_time-self.duration:end_time]
            
            waveform_chunk = self.wave_transforms(waveform_chunk)
        
            log_melspec = torch.log10(self.mel_transform(waveform_chunk)+1e-10)
            log_melspec = (log_melspec - torch.mean(log_melspec)) / torch.std(log_melspec)

            return log_melspec, label_all  # f*t, 152

        elif(self.mode == 'val'):
            log_melspec_list = []
            label_all_list = []

            for end_time in end_times:
                if(end_time > len_wav):
                    pad_len = end_time - len_wav
                    waveform_chunk = F.pad(waveform, (0, pad_len))[0,end_time-self.duration:end_time]
                else:
                    waveform_chunk = waveform[0,end_time-self.duration:end_time]
            
                log_melspec = torch.log10(self.mel_transform(waveform_chunk)+1e-10)
                log_melspec = (log_melspec - torch.mean(log_melspec)) / torch.std(log_melspec)
                log_melspec_list.append(log_melspec)
                label_all_list.append(torch.from_numpy(label_all))

            return torch.stack(log_melspec_list), torch.stack(label_all_list)  # part*f*t, part*152


    @staticmethod
    def wave_transforms(y: np.ndarray, **params):
        # random noise: uniform noise, gaussian noise, pink noise
        transforms = [UNIFORM, GAUSSIAN, PINK_NOISE]
        data = y
        if transforms and (random.random() < CFG.noise_p):
            t = np.random.choice(transforms, p=[0.5,0.3,0.2])
            if t == UNIFORM:
                data = noise_injection(data)
            elif t == GAUSSIAN:
                data = gaussian_noise(data)
            elif t == PINK_NOISE:
                data = pink_noise(data)

        # random volume
        if random.random() < CFG.volume_p:
            db = np.random.uniform(-CFG.db_limit, CFG.db_limit)
            if db >= 0:
                data *= _db2float(db)
            else:
                data *= _db2float(-db)

        # pitch shifting
        # if random.random() < CFG.pitch_p:
        #     data = pitch_shift_spectrogram(data)

        # normalize
        if random.random() < CFG.normalize_p:
            max_vol = np.abs(data).max()
            data = data * (1 / max_vol)

        np.clip(data, 0, 1)
        return data

    
    def __len__(self):
        return self.df.shape[0]
        # return 500
