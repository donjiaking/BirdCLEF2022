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
    res = torch.cat(res)
    return res, id_list


def restore(input_x, lis):
    origin = []
    for part in input_x:
        origin.append(part)
    res = []
    for idx in range(len(lis)):
        res.append(input_x[lis.index(idx)])
    return torch.cat(input_x)


class Net(nn.Module):
    def __init__(self, backbone, training=1, validation=0, testing=0):
        super(Net, self).__init__()
        # self.cfg = cfg
        self.n_classes = 152
        self.training = training
        self.validation = validation
        self.testing = testing
        # self.mel_spec = mel_transform
        # self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=cfg.top_db)
        # self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

        if backbone == 'resnet50':
            self.backbone = ResNet50Bird(self.n_classes)
        elif backbone == 'resnext':
            self.backbone = ResNeXtBird(self.n_classes)
        elif backbone == 'efficientnet':
            self.backbone = EfficientNetBird(self.n_classes)
        else:
            self.backbone = EnsembleModel(ResNet50Bird, ResNeXtBird, EfficientNetBird, self.n_classes)

        """self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
        )
        if "efficientnet" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]"""

        self.global_pool = GeM()
        self.linear = nn.Linear(self.backbone.num_features(), self.n_classes)

        """if cfg.pretrained_weights is not None:
            sd = torch.load(cfg.pretrained_weights, map_location="cpu")["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", cfg.pretrained_weights)"""

        # self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        # self.mixup = Mixup(mix_beta=cfg.mix_beta)
        self.factor = 6  # int(cfg.wav_crop_len / 5.0)

    def forward(self, x):
        """if not self.training:
            # x = batch["input"]
            bs, parts, time = x.shape
            x = x.reshape(parts, time)
            # y = batch["target"]
            # y = y[0]
        else:
            # x = batch["input"]
            # y = batch["target"]
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)"""
        
        bs, freq, time = x.shape
        if not self.testing:
            x = x.reshape(bs * self.factor, freq, time // self.factor)
            x, mixlist = mixup(x)

        print(x.shape)

        """with autocast(enabled=False):
            x = self.wav2img(x)  # (bs, mel, time)
            if self.cfg.mel_norm:
                x = (x + 80) / 80

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        weight = batch["weight"]

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)"""

        x = self.backbone(x)

        """if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)"""
        
        if not self.testing:
            x = restore(x, mixlist)
            b, f, t = x.shape
            b //= self.factor
            t *= self.factor
            x = x.reshape(b, f, t)

            x = F.avg_pool2d(x, kernel_size=(f, t))
            x = x[:, 0, 0]
            x = self.linear(x)

        """loss = self.loss_fn(logits, y)
        loss = (loss.mean(dim=1) * weight) / weight.sum()
        loss = loss.sum()"""

        return x  # {"loss": loss, "logits": logits.sigmoid(), "logits_raw": logits, "target": y}
