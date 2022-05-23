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
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

nocall_detector = models.NocallNet(CFG.nocall_backbone).to(device)
nocall_detector.load_state_dict(torch.load(CFG.nocall_detector_path))  # don't know whether strict=False will cause issues
nocall_detector.eval()

test_meta = pd.read_csv(CFG.root_path + 'test.csv')
train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')

all_bird = train_meta["primary_label"].unique()
with open(CFG.root_path + 'scored_birds.json') as sbfile:
    scored_birds = json.load(sbfile)
file_list = [f.split('.')[0] for f in sorted(os.listdir(CFG.test_audio_path))]

print('Number of test soundscapes:', len(file_list))


def get_wave_list(filepath, segment_test):
    wave_chunk_list = []

    waveform, _ = torchaudio.load(filepath=filepath)
    len_wav = waveform.shape[1]
    waveform = waveform[0, :].reshape(1, len_wav)  # stereo->mono, mono->mono

    chunks = math.ceil(len_wav / segment_test)
    end_times = [(i + 1) * 5 for i in range(chunks)]

    for end_time in end_times:
        if end_time * CFG.sample_rate >= len_wav:
            pad_len = end_time * CFG.sample_rate - len_wav
            waveform_chunk = F.pad(waveform, (0, pad_len))[0, end_time * CFG.sample_rate - segment_test:end_time * CFG.sample_rate]
        else:
            waveform_chunk = waveform[0, end_time * CFG.sample_rate - segment_test:end_time * CFG.sample_rate]
    
        wave_chunk_list.append(waveform_chunk)

    return wave_chunk_list, end_times


def test(model):
    pred = {'row_id': [], 'target': [], 'call_prob': []}

    model.eval()
    with torch.no_grad():
        for file_id in file_list:
            path = CFG.test_audio_path + file_id + '.ogg'

            wave_chunk_list, end_times = get_wave_list(path, CFG.segment_test)
            wave_chunk_list = torch.stack(wave_chunk_list).to(device)  # convert to a batch

            outputs = model(wave_chunk_list)
            outputs_test = torch.sigmoid(outputs)

            with torch.no_grad():
                nocall_res = nocall_detector(wave_chunk_list)
            
            nocall_res = nocall_res.softmax(1).to('cpu').numpy()

            for idx, end_time in enumerate(end_times):
                call_prob = nocall_res[idx][1]  # 1(has a call)'s probility

                for bird in scored_birds:
                    score = outputs_test[idx][np.where(all_bird==bird)]
                    row_id = file_id + '_' + bird + '_' + str(end_time)
                    pred['row_id'].append(row_id)
                    pred['target'].append(True if score > CFG.binary_th and call_prob > CFG.nocall_th else False)
                    pred['call_prob'].append(call_prob)

    results = pd.DataFrame(pred, columns = ['row_id', 'target', 'call_prob'])
    return results


def submit(results):
    results.to_csv("submission.csv", index=False)    

    OUTPUT_DATA_DELETE = False

    if OUTPUT_DATA_DELETE == True:
        shutil.rmtree(CFG.out_train_path)
        shutil.rmtree(CFG.out_val_path)
        shutil.rmtree(CFG.model_out_path)


if __name__ == "__main__":
    model = models.Net(CFG.backbone).to(device)
    load_model(model, model_name='tf_efficientnetv2_s_in21k_best_f1')  # tf_efficientnetv2_s_in21k_best_f1

    results = test(model)
    # print(results) 
    submit(results)

