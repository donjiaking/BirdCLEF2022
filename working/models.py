import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import torchaudio
import timm
import random
from torch.cuda.amp import GradScaler, autocast

from utils import get_mel_transform as mel_transform


class ResNet50Bird(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_outputs)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def num_features(self):
        return self.backbone.fc.out_features


class ResNeXtBird(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.resnext50_32x4d()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_outputs)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def num_features(self):
        return self.backbone.fc.out_features


class EfficientNetBird(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.efficientnet_b2()
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, n_outputs)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def num_features(self):
        return self.backbone.fc.out_features


class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC, n_outputs):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(n_outputs * 3, n_outputs)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out

    def num_features(self):
        return self.backbone.fc.out_features


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


class Mixup(nn.Module):
    def __init__(self, mix_beta):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):
        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight


def mixup(input_x):
    origin = []
    for part in input_x:
        origin.append(part)
    id_list = [i for i in range(len(origin))]
    random.shuffle(id_list)
    res = []
    for i in id_list:
        res.append(origin[i])
    print("In mixup, #parts =", len(res), ", with each", res[0].shape)
    res = torch.stack(res, dim=0)
    return res, id_list


def restore(input_x, lis):
    origin = []
    for part in input_x:
        origin.append(part)
    res = []
    for idx in range(len(lis)):
        res.append(input_x[lis.index(idx)])
    return torch.stack(res, dim=0)


class Net(nn.Module):
    def __init__(self, backbone, training=True, validation=False, testing=False):
        super(Net, self).__init__()
        self.n_classes = 152
        self.training = training
        self.validation = validation
        self.testing = testing
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            in_chans=1
        )

        if "efficientnet" in backbone:
            self.backbone_out = self.backbone.num_features
        else:
            self.backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.linear = nn.Linear(self.backbone_out, self.n_classes)
        self.factor = 6  # int(30.0 / 5.0)

    def forward(self, x):
        print('In the beginning, x.shape =', x.shape)
        bs, freq, time = x.shape
        mix_list = []
        if not self.testing:  # for both training and validation
            x = x.reshape(bs * self.factor, freq, time // self.factor)  #
            print('After reshape, x.shape =', x.shape)
            x, mix_list = mixup(x)
            print('After mixup, x.shape =', x.shape)
            x = x[:, None, :, :]
            print('After dim-increase, x.shape =', x.shape)

        x = self.backbone(x)
        print('After backbone, x.shape =', x.shape)
        
        if not self.testing:
            x = restore(x, mix_list)
            print('After restore, x.shape =', x.shape)
            b, c, t, f = x.shape
            x = x.reshape(b // self.factor, c, self.factor * t, f)
            print('After reshape, x.shape =', x.shape)

            x = F.avg_pool2d(x, kernel_size=(self.factor * t, f))
            print('After pool, x.shape =', x.shape)
            x = x[:, :, 0, 0]
            x = self.linear(x)
            print('After linear, x.shape =', x.shape)

        return x
