from distutils.command.config import config
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from config import CFG

class MyDataset(Dataset):
    def __init__(self, mode='train', transforms=None):
        self.mode = mode
        self.path = CFG.out_train_path if self.mode == 'train' else CFG.out_val_path

        # load label_list
        self.label_list = torch.from_numpy(torch.load(self.path + "label_list.pt"))

    def __getitem__(self, index):
        x = torch.load(self.path + str(index) + '.pt')
        y = self.label_list[index]

        return x, y
    
    def __len__(self):
        return len(self.label_list)
