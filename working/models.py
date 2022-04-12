import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

class ResNet50Bird(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_outputs)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNeXtBird(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.resnext50_32x4d()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_outputs)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientNetBird(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.efficientnet_b2()
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, n_outputs)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EnsembleModel(nn.Module):   
    def __init__(self, modelA, modelB, modelC, n_outputs):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(n_outputs*3, n_outputs)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out