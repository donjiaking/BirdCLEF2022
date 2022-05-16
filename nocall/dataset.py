from utils import *
import torchaudio
from config import CFG


class MyDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        self.mode = mode
        self.df = df.copy()
        self.duration = CFG.segment

    def __getitem__(self, index):
        row = self.df.iloc[index]

        label = row['hasbird']  # 0 or 1

        waveform, _ = torchaudio.load(filepath=CFG.wav_path+str(row['itemid'])+'.wav')
        len_wav = waveform.shape[1]
        waveform = waveform[0, :].reshape(1, len_wav)  # stereo->mono, mono->mono
        waveform_chunk = waveform[0, 0:self.duration]  # since every file is 10s long, only fetch 0-5s period
                  
        waveform_chunk = np.array(waveform_chunk)
        # waveform_chunk = self.wav_trans(waveform_chunk, CFG.sample_rate)
        waveform_chunk = torch.tensor(waveform_chunk)
        # print("In dataset, waveform_chunk.shape =", waveform_chunk.shape)  # torch.Size([160000])

        return waveform_chunk, label  # duration, 1
    
    def __len__(self):
        return self.df.shape[0]

