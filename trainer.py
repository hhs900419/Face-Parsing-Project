import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from metrics import *
from utils import *
from criterion import *

class Trainer:
    def __init__(self, model, trainloader, validloader, epochs, criterion, optimizer, device, savepath, savename, scheduler=None):
        
        self.model = model
        self.train_loader = trainloader
        self.valid_loader = validloader

        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.savepath = savepath
        self.savename = savename

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

    def train_fn(self):
        self.model.train()
        tr_losses = []
        # To record all kinds of evaluation metrics
        metrics = SegMetric(n_classes=19)

        for img, mask in tqdm(self.train_loader):
            img = img.to(self.device)
            mask = mask.to(self.device)
            
            h, w = mask.size()[1], mask.size()[2]

            # 1. clear gradient
            self.optimizer.zero_grad()
            # 2. forward
            outputs = self.model(img)
            # 3. compute the loss
            loss1 = self.criterion(outputs, mask)
            loss2 = cross_entropy2d(outputs, mask.long(), reduction='mean')
            # loss = loss1 + loss2
            loss = loss2
            # 4. back propagation
            loss.backward()
            # 5. parameter update
            self.optimizer.step()
            tr_losses.append(loss.cpu().detach().numpy())
            
            # compute metric score
            outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
            pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
            gt = mask.cpu().numpy()
            metrics.update(gt, pred)

            avg_tr_loss = sum(tr_losses) / len(tr_losses)
            metric_scores = metrics.get_scores()[0]
        return avg_tr_loss, metric_scores
        
    def valid_fn(self):
        self.model.eval()
        val_losses = []
        
        # To record all kinds of evaluation metrics
        metrics = SegMetric(n_classes=19)

        with torch.no_grad():
            for img, mask in tqdm(self.valid_loader):
                img = img.to(self.device)
                mask = mask.to(self.device)
                
                h, w = mask.size()[1], mask.size()[2]
                
                outputs = self.model(img)
                loss1 = self.criterion(outputs, mask)
                loss2 = cross_entropy2d(outputs, mask.long(), reduction='mean')
                # loss = loss1 + loss2
                loss = loss2
                val_losses.append(loss.cpu().detach().numpy())
                
                # Match the dimension to compute different metrics
                outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
                gt = mask.cpu().numpy()
                metrics.update(gt, pred)

            avg_val_loss = sum(val_losses) / len(val_losses)
            metric_scores = metrics.get_scores()[0]
            return avg_val_loss, metric_scores
        
    def run(self):
        history = {
        'train_loss' : [],
        'train_miou' : [],
        'train_f1': [],
        'valid_loss' : [],
        'valid_miou' : [],
        'valid_f1': [],
        'lr' : []
        }

        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1:03d} / {self.epochs:03d}:')
            train_loss, tr_metric_score = self.train_fn()
            valid_loss, val_metric_score = self.valid_fn()
            
            tr_miou = tr_metric_score["Mean IoU : \t"]
            train_f1 = tr_metric_score["Overall F1: \t"]
            val_miou = val_metric_score["Mean IoU : \t"]
            valid_f1 = tr_metric_score["Overall F1: \t"]
            
            # learning rate scheduling
            if self.scheduler:
                self.scheduler.step(valid_f1)
                # self.scheduler.step(val_miou)
            
            # Logs
            print('lr:',self.get_lr(self.optimizer))
            print(f'train loss: {train_loss}  valid loss: {valid_loss}')
            print("----------------- Total Train Performance --------------------")
            for k, v in tr_metric_score.items():
                print(k, v)
            print("---------------------------------------------------")

            print("----------------- Total Val Performance --------------------")
            for k, v in val_metric_score.items():
                print(k, v)
            print("---------------------------------------------------")

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_miou'].append(tr_miou)
            history['valid_miou'].append(val_miou)
            history['train_f1'].append(train_f1)
            history['valid_f1'].append(valid_f1)

            # save model if best valid
            # if torch.tensor(history['valid_loss']).argmin() == epoch:
            if torch.tensor(history['valid_f1']).argmax() == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.savepath, self.savename))
                print('Model Saved!')
        self.plot_save_history(history)


    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def plot_save_history(self, metrics):
        # Plot the loss curve against epoch
        fig, ax = plt.subplots(3, 1, figsize=(15, 15), dpi=100)
        ax[0].set_title('Loss (CE)')
        ax[0].plot(range(self.epochs), metrics['train_loss'], label='Train')
        ax[0].plot(range(self.epochs), metrics['valid_loss'], label='Valid')
        ax[0].legend()
        ax[1].set_title('F1')
        ax[1].plot(range(self.epochs), metrics['train_f1'], label='Train')
        ax[1].plot(range(self.epochs), metrics['valid_f1'], label='Valid')
        ax[1].legend()
        ax[2].set_title('learning rate')
        ax[2].plot(range(self.epochs), metrics['lr'], label='Train')
        ax[2].legend()
        plt.show()
        fig.savefig(os.path.join(self.savepath , f'metrics_{get_current_timestamp()}.jpg'))
        plt.close()

    
