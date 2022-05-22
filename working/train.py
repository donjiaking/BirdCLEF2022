import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
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
# device = 'cpu'
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
    val_iters = len(val_loader)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for i, (inputs_val, labels_val) in enumerate(val_loader):
            if(inputs_val.shape[0] > 64):  # cut if too large
                inputs_val = inputs_val[:64]
                labels_val = labels_val[:64]

            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(inputs_val)
            loss_val = criterion(outputs_val, labels_val)
            loss_val = loss_val.mean(dim=1).mean()

            val_loss += loss_val.item()

            y_true.append(labels_val)
            y_pred.append(outputs_val)

    y_true = torch.cat(y_true)
    # y_true[y_true == 0.3] = 1
    y_pred = torch.cat(y_pred)
    y_pred = torch.sigmoid(y_pred)
    y_pred[y_pred >= CFG.binary_th] = 1
    y_pred[y_pred < CFG.binary_th] = 0
    
    val_f1 = utils.get_f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

    return val_loss / val_iters, val_f1


def train(model, model_name, train_loader, val_loader):
    logger = utils.get_logger(f"log_{model_name}.txt")

    num_epochs = CFG.num_epochs
    train_iters = len(train_loader)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.t_max*train_iters, eta_min=1e-6)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps = CFG.warmup_epochs * len(train_loader),
    #     num_training_steps = num_epochs * len(train_loader),
    # )
    history = np.zeros((0, 4))


    best_loss = 1000
    best_f1 = 0
    print("Training started")
    for epoch in range(num_epochs):
        train_loss, val_loss = 0, 0

        model.train()
        model.to(device)
        for i, (inputs, labels, weights) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            outputs, labels_new, weights_new = model(inputs, labels, weights)
            loss = criterion(outputs, labels_new)
            loss = (loss.mean(dim=1) * weights_new) / weights_new.sum()
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (i + 1) % CFG.print_feq == 0:
                logger.info(f'Epoch[{epoch + 1}/{num_epochs}] Iter[{i + 1}/{train_iters}] : train_loss {train_loss/(i + 1):.5f}')

            del loss, outputs

        val_loss, val_f1 = evaluate(model, criterion, val_loader)
        
        train_loss = train_loss / train_iters
        logger.info(f'== Epoch [{(epoch + 1)}/{num_epochs}]: train_loss {train_loss:.5f}, val_loss {val_loss:.5f}, val_f1 {val_f1:.5f} ')
        utils.write_tensorboard(f"log_{model_name}_batch{CFG.batch_size}_lr{CFG.lr}",train_loss,val_loss,val_f1,epoch+1)
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
    train_index, val_index = train_test_split(range(0, train_meta.shape[0]), train_size=0.8, test_size=0.2, random_state=42)

    train_dataset = MyDataset(train_meta.iloc[train_index], mode='train')
    val_dataset = MyDataset(train_meta.iloc[val_index], mode='val')

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.val_batch_size, num_workers=4, shuffle=False, collate_fn=val_collate_fn)

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
