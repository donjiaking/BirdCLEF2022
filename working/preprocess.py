import os
import random
import colorednoise as cn
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

from config import CFG
import utils

UNIFORM = 0
GAUSSIAN = 1
PINK_NOISE = 2

if not os.path.exists(CFG.out_train_path):
    os.mkdir(CFG.out_train_path)

if not os.path.exists(CFG.out_val_path):
    os.mkdir(CFG.out_val_path)

train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')

with open(CFG.root_path + 'scored_birds.json') as sbfile:
    scored_birds = json.load(sbfile)
    
all_bird = train_meta["primary_label"].unique()
print(len(all_bird))

mel_transform = utils.get_mel_transform()


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
    if transforms and (random.random() < CFG.volume_p):
        db = np.random.uniform(-CFG.db_limit, CFG.db_limit)
        if db >= 0:
            data *= _db2float(db)
        else:
            data *= _db2float(-db)

    # normalize
    if transforms and (random.random() < CFG.normalize_p):
        max_vol = np.abs(data).max()
        data = data * (1 / max_vol)

    np.clip(data, 0, 1)
    return data


def preprocess_train(filepath, outpath, segment_train, label_list, data_index=0, label_file=[]):
    label_file_all = np.zeros(all_bird.shape)
    for label_file_temp in label_file:  # label_file is primary label + secondary label
        label_file_all += (label_file_temp == all_bird)
    label_file_all = np.clip(label_file_all, 0, 1)
    
    waveform, _ = torchaudio.load(filepath=filepath)
    len_wav = waveform.shape[1]
    waveform = waveform[0, :].reshape(1, len_wav)  # stereo->mono mono->mono

    if len_wav < segment_train:
        for _ in range(round(segment_train/len_wav)):
            waveform = torch.cat((waveform, waveform[:, 0:len_wav]), 1)
        len_wav = segment_train
        waveform = waveform[:, 0:len_wav]

    for index in range(int(len_wav/segment_train)):
        wave_seg = waveform[0, index * segment_train:(index + 1) * segment_train]
        wave_seg = wave_transforms(wave_seg)

        log_melspec = torch.log10(mel_transform(wave_seg) + 1e-10)
        log_melspec = (log_melspec - torch.mean(log_melspec)) / torch.std(log_melspec)

        torch.save(log_melspec, outpath + str(data_index) + '.pt')
        label_list.append(label_file_all)
        data_index += 1
            
    return data_index


if __name__ == "__main__":
    train_index, val_index = train_test_split(range(0, train_meta.shape[0]), train_size=0.8, random_state=42)
    # print(len(train_meta['primary_label'][train_index]), len(train_meta['primary_label']))

    # generate train images
    data_index = 0
    label_list = []
    for pri_label, secon_label, f_name in zip((train_meta['primary_label'][train_index]), train_meta['secondary_labels'][train_index], train_meta['filename'][train_index]):
        data_index = preprocess_train(CFG.input_path + f_name, CFG.out_train_path, CFG.segment_train, label_list, data_index, [pri_label] + eval(secon_label))
    torch.save(np.stack(label_list), CFG.out_train_path + 'label_list.pt')
    print("out train finished")

    # generate validation images
    data_index = 0
    label_list = []
    for pri_label, secon_label, f_name in zip((train_meta['primary_label'][val_index]), train_meta['secondary_labels'][val_index], train_meta['filename'][val_index]):
        data_index = preprocess_train(CFG.input_path + f_name, CFG.out_val_path, CFG.segment_train, label_list, data_index, [pri_label] + eval(secon_label))
    torch.save(np.stack(label_list), CFG.out_val_path + 'label_list.pt')
