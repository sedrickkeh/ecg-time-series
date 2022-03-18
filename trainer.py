import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

class Trainer:
    def __init__(self, model, device, loss_func, optimizer):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer

    def train(self, train_loader):
        running_loss, preds_arr, gtruth_arr = 0, [], []
        for X, y in tqdm(train_loader):
            X, y = X.to(self.device), y.to(self.device)
            self.model.train()
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_func(preds, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()*y.shape[0]
            preds_arr.extend(np.argmax(preds.detach().cpu().numpy(), axis=1))
            gtruth_arr.extend(y.detach().cpu().numpy())

        accuracy = accuracy_score(gtruth_arr, preds_arr)
        return running_loss/len(train_loader), accuracy


    def validate(self, valid_loader):
        running_loss, preds_arr, gtruth_arr = 0, [], []
        for X, y in tqdm(valid_loader):
            X, y = X.to(self.device), y.to(self.device)
            self.model.eval()
            preds = self.model(X)
            loss = self.loss_func(preds, y)

            running_loss += loss.item()*y.shape[0]
            preds_arr.extend(np.argmax(preds.detach().cpu().numpy(), axis=1))
            gtruth_arr.extend(y.detach().cpu().numpy())

        accuracy = accuracy_score(gtruth_arr, preds_arr)
        print(classification_report(gtruth_arr, preds_arr, digits=5))
        return running_loss/len(valid_loader), accuracy