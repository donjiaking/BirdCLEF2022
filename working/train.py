from cProfile import label
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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

def val_collate_fn(batch):
    """define how to form a batch given a set of samples"""
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.cat(imgs, 0), torch.cat(targets, 0)

def evaluate(model, criterion, val_loader):
    val_loss = 0
    y_true = []
    y_pred = []

    model.eval()
    model.to('cpu')  # change to cpu since bs here is not deterministic
    with torch.no_grad():
        for i, (inputs_val, labels_val) in enumerate(val_loader):
            inputs_val = inputs_val.to('cpu')
            labels_val = labels_val.to('cpu')

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


def train(model, model_name, train_loader, val_loader):
    logger = utils.get_logger(f"log_{model_name}.txt")

    num_epochs = CFG.num_epochs
    lr = CFG.lr

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-7)
    history = np.zeros((0, 4))

    train_iters = len(train_loader)
    val_iters = len(val_loader)

    best_loss = 1000
    best_f1 = 0
    print("Training started")
    for epoch in range(num_epochs):
        train_loss, val_loss = 0, 0

        model.train()
        model.to(device)
        for i, (inputs, labels, weights) in enumerate(train_loader):
            inputs = inputs[:, :, :-2]  # batch_size*128*936, batch_size*152
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            outputs, labels_new, weights_new = model(inputs, labels, weights)
            criterion.weight = weights_new
            loss = criterion(outputs, labels_new)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % CFG.print_feq == 0:
                logger.info(f'Epoch[{epoch + 1}/{num_epochs}] Iter[{i + 1}/{train_iters}] : train_loss {train_loss/(i + 1):.5f}')
        
        scheduler.step()
        val_loss, val_f1 = evaluate(model, criterion, val_loader)
        
        train_loss = train_loss / train_iters
        val_loss = val_loss / val_iters
        logger.info(f'== Epoch [{(epoch + 1)}/{num_epochs}]: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}, val_f1 {val_f1:.5f} ')
        item = np.array([epoch + 1, train_loss, val_loss, val_f1])
        history = np.vstack((history, item))

        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model(model, model_name+f"_best_loss")
            logger.info(f"== Saving Best Loss Model ")
        if val_f1 > best_f1:
            best_f1 = val_f1
            utils.save_model(model, model_name+f"_best_f1")
            logger.info(f"== Saving Best F1 Model ")

    utils.save_model(model, model_name+"_last")
    logger.info(f"== Saving Last Model ")
    utils.plot_history(history, model_name)


if __name__ == "__main__":
    train_meta = pd.read_csv(CFG.root_path + 'train_metadata.csv')
    train_index, val_index = train_test_split(range(0, train_meta.shape[0]), train_size=0.8, random_state=42)

    train_dataset = MyDataset(train_meta.iloc[train_index], mode='train')
    val_dataset = MyDataset(train_meta.iloc[val_index], mode='val')

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, collate_fn=val_collate_fn)

    model = models.Net(CFG.backbone).to(device)
    train(model, CFG.backbone, train_loader, val_loader)

    # modelA = models.ResNet50Bird(152).to(device)
    # train(modelA, 'resnet50')

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
