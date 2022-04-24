import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 

from config import CFG
import utils
from dataset import MyDataset
import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

utils.fix_seed()


def evaluate(model, criterion, val_loader):
    val_loss = 0
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for i, (inputs_val, labels_val) in enumerate(val_loader):
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)

            outputs_val = model(inputs_val)
            loss_val = criterion(outputs_val, labels_val)
            val_loss += loss_val.item()

            y_true.append(labels_val)
            y_pred.append(outputs_val)
    
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_pred = torch.sigmoid(y_pred)
    y_pred[y_pred >= CFG.binary_th] = 1
    y_pred[y_pred < CFG.binary_th] = 0
    
    val_f1 = utils.get_f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

    return val_loss, val_f1


def train(model, model_name):
    logger = utils.get_logger(f"log_{model_name}.txt")

    num_epochs = CFG.num_epochs
    lr = CFG.lr
    batch_size = CFG.batch_size

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    history = np.zeros((0, 4))

    train_dataset = MyDataset(mode='train')
    val_dataset = MyDataset(mode='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_iters = len(train_loader)
    val_iters = len(val_loader)

    best_loss = 1000
    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss, val_loss = 0, 0

        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            print("inputs.shape =", inputs.shape)  # batch_size * 128 * 938
            print("labels.shape =", labels.shape)  # batch_size * 152

            inputs = inputs.to(device)
            labels = labels.to(device)

            # mixup - augmentation for melspectrum
            

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % CFG.print_feq == 0:
                logger.info(f'Epoch[{epoch + 1}/{num_epochs}] Iter[{i + 1}/{train_iters}] : train_loss {train_loss/(i + 1):.5f}')
        
        scheduler.step()
        val_loss, val_f1 = evaluate(model, criterion, val_loader)
        
        train_loss = train_loss / train_iters
        val_loss = val_loss / val_iters
        logger.info(f'== Epoch [{(epoch + 1)}/{num_epochs}]: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}, val_f1 {val_f1:.5f} ==')
        item = np.array([epoch + 1, train_loss, val_loss, val_f1])
        history = np.vstack((history, item))

        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model(model, model_name+f"_best_loss")
            logger.info(f"== Saving Best Loss Model: epoch {epoch + 1} val_loss {val_loss:.5f} val_f1 {val_f1:.5f} ==")
        if val_f1 > best_f1:
            best_f1 = val_f1
            utils.save_model(model, model_name+f"_best_f1")
            logger.info(f"== Saving Best F1 Model: epoch {epoch + 1} val_loss {val_loss:.5f} val_f1 {val_f1:.5f} ==")

    utils.save_model(model, model_name+"_last")
    logger.info(f"== Saving Last Model: epoch {epoch + 1} val_loss {val_loss:.5f} val_f1 {val_f1:.5f} ==")
    utils.plot_history(history, model_name)


if __name__ == "__main__":
    # modelA = models.ResNet50Bird(152).to(device)
    # train(modelA, 'resnet50')

    cfg = CFG()
    model = models.Net('resnet50').to(device)
    train(model, 'resnet50')

    # modelB = models.ResNeXtBird(152).to(device)
    # train(modelB, 'resnext50')

    # modelC = models.EfficientNetBird(152).to(device)
    # train(modelC, 'efficientnet')

    # model_ensemble = models.EnsembleModel(modelA, modelB, modelC).to(device)
    
    # for param in model_ensemble.parameters():
    #     param.requires_grad = False

    # for param in model_ensemble.classifier.parameters():
    #     param.requires_grad = True  

    # train(model_ensemble, 'ensemble')
