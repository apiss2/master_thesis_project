import os
import cv2
import torch
import numpy as np

from .train_util import Epoch

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, \
                        optimizer, device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, batch):
        x, y = batch['x'], batch['y'].long()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss, _ = self.update_loss(y, pred, self.loss)
        loss.backward()
        self.optimizer.step()
        self.update_metrics(y, pred, self.metrics)

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='valid', device=device
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, batch):
        with torch.no_grad():
            x, y = batch['x'], batch['y'].long()
            pred = self.model(x)
            self.update_loss(y, pred, self.loss)
            self.update_metrics(y, pred, self.metrics)
