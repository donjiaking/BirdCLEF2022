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
import torchaudio.transforms as T
from audiomentations import Compose, GainTransition,  AddGaussianSNR, Normalize, AddBackgroundNoise

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
    """
    Shift a spectrogram along the frequency axis in the spectral-domain at random
    """
    nb_cols = y.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    # print("max_shifts:", max_shifts)
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    augmented = np.roll(y, nb_shifts, axis=0)
    augmented = torch.tensor(augmented).to(y.dtype)

    return augmented

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
        if random.random() < CFG.pitch_p:
            data = pitch_shift_spectrogram(data)

        # normalize
        if random.random() < CFG.normalize_p:
            max_vol = np.abs(data).max()
            data = data * (1 / max_vol)

        np.clip(data, 0, 1)
        return data

def add_background_noise(y, sr):
    transforms = Compose(
    [
        # AddGaussianSNR(p=CFG.gaussianSNR_p),
        # GainTransition(p=CFG.gainTransition_p,min_gain_in_db=-2,max_gain_in_db= 2, min_duration = 0.2, max_duration = 4.5),
        # PitchShift(min_semitones=-4, max_semitones=4, p=CFG.pitch_shift_p),
        AddBackgroundNoise(
            sounds_path=CFG.BACKGROUND_PATH1, min_snr_in_db=3, max_snr_in_db=30, p=0.5
        ),
        AddBackgroundNoise(
            sounds_path=CFG.BACKGROUND_PATH2, min_snr_in_db=3, max_snr_in_db=30, p=0.25
        ),
        AddBackgroundNoise(
            sounds_path=CFG.BACKGROUND_PATH3,
            min_snr_in_db=3,
            max_snr_in_db=30,
            p=0.25,
        ),
    ]
    )
    return transforms(y, sr)


class MyDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        self.mode = mode
        self.df = df.copy()
        if(self.mode == 'train'):
            self.df = self.df[self.df['rating'] >= CFG.min_rating]
            self.df["weight"] = self.df["rating"] / self.df["rating"].max()

        train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')
        self.all_bird = train_meta["primary_label"].unique()

        self.duration = CFG.segment_train if (self.mode=='train') else CFG.segment_test

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # treat primary and secondary label differently
        label_all = np.zeros(CFG.n_classes)
        for bird in eval(row['secondary_labels']):
            label_all[np.argwhere(self.all_bird == bird)] = 1  # 0.6 0.3
        for bird in [row['primary_label']]:
            label_all[np.argwhere(self.all_bird == bird)] = 1

        waveform, _ = torchaudio.load(filepath=CFG.input_path+row['filename'])
        len_wav = waveform.shape[1]
        waveform = waveform[0, :].reshape(1, len_wav)  # stereo->mono mono->mono
        chunks = math.ceil(len_wav / self.duration)
        end_times = [(i+1)*self.duration for i in range(chunks)]

        if(self.mode == 'train'):
            weight = row['weight']
            end_time = random.choice(end_times)

            if(end_time > len_wav):
                pad_len = end_time - len_wav
                waveform_chunk = F.pad(waveform, (0, pad_len))[0,end_time-self.duration:end_time]
            else:
                waveform_chunk = waveform[0,end_time-self.duration:end_time]
            
            waveform_chunk = add_background_noise(np.array(waveform_chunk), CFG.sample_rate)
            waveform_chunk = wave_transforms(torch.tensor(waveform_chunk))
            return waveform_chunk, label_all, weight  # duration, 152, 1

        elif(self.mode == 'val'):
            wave_chunk_list = []
            label_all_list = []

            for end_time in end_times:
                if(end_time > len_wav):
                    pad_len = end_time - len_wav
                    waveform_chunk = F.pad(waveform, (0, pad_len))[0,end_time-self.duration:end_time]
                else:
                    waveform_chunk = waveform[0,end_time-self.duration:end_time]
            
                wave_chunk_list.append(waveform_chunk)
                label_all_list.append(torch.from_numpy(label_all))

            return torch.stack(wave_chunk_list), torch.stack(label_all_list)  # part*duration, part*152
            
    
    def __len__(self):
        return self.df.shape[0]
        # return 100
