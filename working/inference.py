import math
import os
import shutil
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json

from config import CFG
import models
import utils


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

test_meta = pd.read_csv(CFG.root_path + 'test.csv')
train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')

all_bird = train_meta["primary_label"].unique()
with open(CFG.root_path + 'scored_birds.json') as sbfile:
    scored_birds = json.load(sbfile)
file_list = [f.split('.')[0] for f in sorted(os.listdir(CFG.test_audio_path))]
mel_transform = utils.get_mel_transform()

print('Number of test soundscapes:', len(file_list))


def get_mel_list(filepath, segment_test):
    mel_list_test = []

    waveform, _ = torchaudio.load(filepath=filepath)
    len_wav = waveform.shape[1]
    waveform = waveform[0,:].reshape(1, len_wav) # stereo->mono mono->mono

    chunks = math.ceil(len_wav / segment_test)
    end_times = [(i+1)*5 for i in range(chunks)]

    for end_time in end_times:
        if(end_time*CFG.sample_rate >= len_wav):
            pad_len = end_time*CFG.sample_rate - len_wav
            waveform_chunk = F.pad(waveform, (0, pad_len))[0,end_time*CFG.sample_rate-segment_test:end_time*CFG.sample_rate]
        else:
            waveform_chunk = waveform[0,end_time*CFG.sample_rate-segment_test:end_time*CFG.sample_rate]
    
        log_melspec = torch.log10(mel_transform(waveform_chunk).reshape(1, 128, 157)+1e-10)
        log_melspec = (log_melspec - torch.mean(log_melspec)) / torch.std(log_melspec)
        mel_list_test.append(log_melspec)

    return mel_list_test, end_times


def test(model):
    pred = {'row_id': [], 'target': []}

    model.eval()
    with torch.no_grad():
        for file_id in file_list:
            path = CFG.test_audio_path + file_id + '.ogg'

            mel_list_test, end_times = get_mel_list(path, CFG.segment_test)
            mel_list_test = torch.stack(mel_list_test).to(device)

            outputs = model(mel_list_test)
            outputs_test = torch.sigmoid(outputs)

            for idx, end_time in enumerate(end_times):
                for bird in scored_birds:
                    score = outputs_test[idx][np.where(all_bird==bird)]
                    row_id = file_id + '_' + bird + '_' + str(end_time)
                    pred['row_id'].append(row_id)
                    pred['target'].append(True if score > CFG.binary_th else False)

    results = pd.DataFrame(pred, columns = ['row_id', 'target'])
    return results


def submit(results):
    results.to_csv("submission.csv", index=False)    

    OUTPUT_DATA_DELETE = False

    if OUTPUT_DATA_DELETE == True:
        shutil.rmtree(CFG.out_train_path)
        shutil.rmtree(CFG.out_val_path)
        shutil.rmtree(CFG.model_out_path)


if __name__ == "__main__":
    model = models.ResNet50Bird(152).to(device)
    utils.load_model(model, model_name='resnet50_best_f1')

    results = test(model)
    # print(results) 
    submit(results)

