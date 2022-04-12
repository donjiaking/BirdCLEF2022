import os
import shutil
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score 

from config import CFG
import models
import utils


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

test_meta = pd.read_csv(CFG.root_path + 'test.csv')
train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')

all_bird = train_meta["primary_label"].unique()
file_list = [f.split('.')[0] for f in sorted(os.listdir(CFG.test_audio_path))]
mel_transform = utils.get_mel_transform()

print('Number of test soundscapes:', len(file_list))


def get_mel_list(filepath, end_time, segment_test):
    mel_list = []
    waveform, _ = torchaudio.load(filepath=filepath)
    waveform = waveform[0,end_time*CFG.sample_rate-segment_test:end_time*CFG.sample_rate].reshape(1, segment_test) # stereo->mono mono->mono

    log_melspec = torch.log10(mel_transform(waveform).reshape(1, 128, 157)+1e-10)
    log_melspec = (log_melspec - torch.mean(log_melspec)) / torch.std(log_melspec)
        
    mel_list.append(log_melspec)
    return mel_list


def test(model):
    pred = {'row_id': [], 'target': []}
    binary_th = 0.3

    model.eval()

    for row_id, file_id, bird, end_time in zip(test_meta['row_id'], test_meta['file_id'], test_meta['bird'], test_meta['end_time']):
        path = CFG.test_audio_path + file_id + '.ogg'

        mel_list_test = get_mel_list(path, end_time, CFG.segment_test)
        mel_list_test = torch.stack(mel_list_test).to(device)

        outputs = model(mel_list_test)
        
        outputs_test = torch.sigmoid(outputs)
        score = outputs_test[0][np.where(all_bird==bird)]
                
        pred['row_id'].append(row_id)
        pred['target'].append(True if score > binary_th else False)

    results = pd.DataFrame(pred, columns = ['row_id', 'target'])
    # print(results) 
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

    utils.load_model(model, model_name='resnet50')

    results = test(model)

    submit(results)

