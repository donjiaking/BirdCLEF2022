import os
from sklearn.model_selection import train_test_split
import json
from lightgbm import train
import pandas as pd
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

from config import CFG
import utils

if(not os.path.exists(CFG.out_train_path)):
    os.mkdir(CFG.out_train_path)

if(not os.path.exists(CFG.out_val_path)):
    os.mkdir(CFG.out_val_path)

train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')

with open(CFG.root_path + 'scored_birds.json') as sbfile:
    scored_birds = json.load(sbfile)
    
all_bird = train_meta["primary_label"].unique()

mel_transform = utils.get_mel_transform()


def preprocess_train(filepath, outpath, segment_train, label_list, data_index=0, label_file=[]):
    label_file_all = np.zeros(all_bird.shape)
    for label_file_temp in label_file:  # label_file is primary label + secondary label
        label_file_all += (label_file_temp == all_bird)
    label_file_all = np.clip(label_file_all, 0, 1)
    
    waveform, _ = torchaudio.load(filepath=filepath)
    len_wav = waveform.shape[1]
    waveform = waveform[0,:].reshape(1, len_wav) # stereo->mono mono->mono
    if len_wav < segment_train:
        for _ in range(round(segment_train/len_wav)):
            waveform = torch.cat((waveform,waveform[:,0:len_wav]),1)
        len_wav = segment_train
        waveform = waveform[:,0:len_wav]

    for index in range(int(len_wav/segment_train)):
        log_melspec = torch.log10(mel_transform(waveform[0, index*segment_train:index*segment_train+segment_train]).reshape(1, 128, 157)+1e-10)
        log_melspec = (log_melspec - torch.mean(log_melspec)) / torch.std(log_melspec)

        torch.save(log_melspec, outpath + str(data_index) + '.pt')
        label_list.append(label_file_all)
        data_index += 1
            
    return data_index


if __name__ == "__main__":
    train_index, val_index = train_test_split(range(0,train_meta.shape[0]), train_size=0.8, random_state=42)
    # print(len(train_meta['primary_label'][train_index]), len(train_meta['primary_label']))

    # generate train images
    data_index = 0
    label_list = []
    for pri_label, secon_label, f_name in zip((train_meta['primary_label'][train_index]), train_meta['secondary_labels'][train_index], train_meta['filename'][train_index]):
        data_index = preprocess_train(CFG.input_path+f_name, CFG.out_train_path, CFG.segment_train, label_list, data_index, [pri_label]+eval(secon_label))
    torch.save(np.stack(label_list), CFG.out_train_path+'label_list.pt')

    # generate validation images
    data_index = 0
    label_list = []
    for pri_label, secon_label, f_name in zip((train_meta['primary_label'][val_index]), train_meta['secondary_labels'][val_index], train_meta['filename'][val_index]):
        data_index = preprocess_train(CFG.input_path+f_name, CFG.out_val_path, CFG.segment_train, label_list, data_index, [pri_label]+eval(secon_label))
    torch.save(np.stack(label_list), CFG.out_val_path+'label_list.pt')