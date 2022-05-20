from pyexpat import model
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import torchaudio.transforms as T
import timm
import random

from config import CFG
import utils


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

    def forward(self, X, Y, weights=None):
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

        if weights is None:
            return X, Y
        else:
            weights = coeffs.view(-1) * weights + (1 - coeffs.view(-1)) * weights[perm]
            return X, Y, weights


class Net(nn.Module):
    def __init__(self, backbone_name):
        super(Net, self).__init__()
        self.mixup = Mixup(mix_beta=CFG.mix_beta)
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=CFG.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=1,
        )

        if "efficientnet" in self.backbone_name:
            self.backbone_out = self.backbone.num_features
        else:
            self.backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()
        self.linear = nn.Linear(self.backbone_out, CFG.n_classes)

        self.factor = int(CFG.segment_train / CFG.segment_test)  # int(30.0 / 5.0)

        self.wav2img = nn.Sequential(utils.get_mel_transform(), T.AmplitudeToDB(top_db=None))

    def forward(self, x, y=None, weights=None):
        b, t = x.shape
        if self.training:
            x = x.reshape(b * self.factor, t // self.factor)

        x = self.wav2img(x) 
        x = utils.channel_norm(x)

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]  # 6bs*1*t*f

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)  # 6bs*t*1*f
            x = x.reshape(b // self.factor, t * self.factor, c, f)  # bs*6t*1*f
            x, y, weights = self.mixup(x, y, weights)

            x = x.reshape(b, t, c, f)  # 6bs*t*1*f
            x = x.permute(0, 2, 1, 3)  # 6bs*1*t*f

        x = self.backbone(x)
        
        if self.training: 
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)  # 6bs*t*1*f
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)  # 6bs*1*t*f

        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.linear(x)
        
        if self.training:
            return x, y, weights
        else:
            return x


class NocallNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=1)
        
        if "efficientnet" in model_name:
            n_features = self.model.num_features
        else:
            n_features = self.model.feature_info[-1]["num_chs"]
        
        self.global_pool = GeM()
        self.linear = nn.Linear(n_features, 2)

        self.wav2img = nn.Sequential(utils.get_mel_transform(), T.AmplitudeToDB(top_db=None))

    def forward(self, x):
        # x.shape = bs * time(160000)
        x = self.wav2img(x)  # convert wave to image (2-dim)
        x = utils.channel_norm(x)  # normalize
        x = x.permute(0, 2, 1)  # bs * im1 * im2
        x = x[:, None, :, :]  # bs * 1* im1 * im2
        x = self.model(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.linear(x)
        return x


# class ResNet50Bird(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         self.backbone = models.resnet50()
#         self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_outputs)
#         self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     def forward(self, x):
#         x = self.backbone(x)
#         return x

#     def num_features(self):
#         return self.backbone.fc.out_features


# class ResNeXtBird(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         self.backbone = models.resnext50_32x4d()
#         self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_outputs)
#         self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     def forward(self, x):
#         x = self.backbone(x)
#         return x

#     def num_features(self):
#         return self.backbone.fc.out_features


# class EfficientNetBird(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         self.backbone = models.efficientnet_b2()
#         self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, n_outputs)
#         self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

#     def forward(self, x):
#         x = self.backbone(x)
#         return x

#     def num_features(self):
#         return self.backbone.fc.out_features


# class EnsembleModel(nn.Module):
#     def __init__(self, modelA, modelB, modelC, n_outputs):
#         super().__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.modelC = modelC
#         self.classifier = nn.Linear(n_outputs * 3, n_outputs)
        
#     def forward(self, x):
#         x1 = self.modelA(x)
#         x2 = self.modelB(x)
#         x3 = self.modelC(x)
#         x = torch.cat((x1, x2, x3), dim=1)
#         out = self.classifier(x)
#         return out

#     def num_features(self):
#         return self.backbone.fc.out_features
