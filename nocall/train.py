from utils import *
from config import CFG
from dataset import *
from model import *


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    '''perform training on one epoch of data.'''    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()    
        
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # load data
        images = images.to(device)  # torch.Size([16, 160000])
        labels = labels.to(device)  # torch.Size([16])
        batch_size = labels.size(0)
        
        # forward pass
        y_preds = model(images.to(torch.float32))  # torch.Size([16, 2])

        # calculate loss
        loss = criterion(y_preds, labels)        
        
        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(
                   epoch + 1, step + 1, len(train_loader), 
                   batch_time=batch_time, data_time=data_time, 
                   remain=timeSince(start, float(step + 1) / len(train_loader)),
                   loss=losses,
                   grad_norm=grad_norm,
                   ))

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    '''perform validation'''    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(images.to(torch.float32))
        
        print("y_preds.shape =", y_preds.shape)
        print(y_preds[0] + y_preds[1] == 1)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)        
        
        # record accuracy
        y_pred = y_preds.softmax(1).to('cpu').numpy()
        print("y_pred.shape =", y_pred.shape)
        preds.append(y_pred)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step + 1, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, 
                   remain=timeSince(start, float(step + 1) / len(valid_loader)),
                   loss=losses
                   ))
    
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference(model, states, test_loader, device):
    '''inference'''
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


def train_loop(train_meta, train_idx, valid_idx):
    # LOGGER.info(f"---------- training ----------")
    print("========== Training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = train_meta.iloc[train_idx]
    valid_folds = train_meta.iloc[valid_idx]

    train_dataset = MyDataset(train_folds, mode='train', transforms=None)  # get_transforms(data='train')
    valid_dataset = MyDataset(valid_folds, mode='valid', transforms=None)  # get_transforms(data='valid')

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomResNext(CFG.model_name, pretrained=True)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = 0.
    best_loss = np.inf
    
    scores = []
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        
        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_col].values
        
        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds.argmax(1))  # argmax(dim=1) returns the column (0 / 1) having the max value corresponding to each row

        elapsed = time.time() - start_time

        # LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print('Epoch {%d} - avg_train_loss: {%.4f}  avg_val_loss: {%.4f}  time: {%.0f}s' % (epoch + 1, avg_loss, avg_val_loss, elapsed))
        # LOGGER.info(f'Epoch {epoch + 1} - Accuracy(validation): {score}')
        print('Epoch {%d} - Accuracy(validation): {%.5f}' % (epoch + 1, score))
        
        scores.append(score)        
        
        # save the model weights with the best score 
        if score > best_score:
            best_score = score
            # LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            print('Epoch {%d} - Save Best Score: {%.4f} Model' % (epoch + 1, best_score))
            torch.save({'model': model.state_dict(), 'preds': preds},
                        './{%s}_best.pt' % CFG.model_name)
    
    check_point = torch.load('./{%s}_best.pt' % CFG.model_name)
    valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds, scores

