import torch
import torch.nn as nn
import timm
from utils import *
from config import CFG

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    # Generalized mean: https://arxiv.org/abs/1711.02512
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return self.__class__.__name__ + "(p=" + "{:.4f}".format(self.p.data.tolist()[0]) + ", eps=" + str(self.eps) + ")"


class CustomResNext(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=1)
        n_features = self.model.feature_info[-1]["num_chs"]  # 2048? ; .fc.in_features
        # print('n_features in resnext:', n_features)
        self.global_pool = GeM()
        self.linear = nn.Linear(n_features, CFG.target_size)
        self.wav2img = nn.Sequential(get_mel_transform(), T.AmplitudeToDB(top_db=None))

    def forward(self, x):
        # x.shape = bs * time(160000)
        x = self.wav2img(x)  # convert wave to image (2-dim)
        x = channel_norm(x)  # normalize
        x = x.permute(0, 2, 1)  # bs * im1 * im2
        x = x[:, None, :, :]  # bs * 1* im1 * im2
        x = self.model(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.linear(x)
        return x


def get_scheduler(optimizer):
    '''cosine annealing scheduler'''
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    return scheduler

