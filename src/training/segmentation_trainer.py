import os
import cv2
import torch
import numpy as np

from .train_util import Epoch, Tester

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

class TestEpoch(Tester):
    def __init__(self, model, loss, metrics:list, device:str='cpu', \
                    label_type:str='binary_label', color_palette:list=None,
                    save_path:str=None, mean:list=None, std:list=None):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device,
            label_type=label_type, color_palette=color_palette,
            save_path=save_path, mean=mean, std=std
        )

    def batch_update(self, batch):
        self.iter_num += 1
        self.all_logs[self.iter_num] = dict()

        with torch.no_grad():
            x, y = batch['x'], batch['y']
            pred = self.model(x)
            _, loss_value = self.update_loss(y, pred, self.loss)
            metrics_values = self.update_metrics(y, pred, self.metrics)
            self.all_logs[self.iter_num][self.loss.__name__] = loss_value
            self.all_logs[self.iter_num].update(metrics_values)

        if self.save_image:
            # predict image
            name = 'predict_{}_{:03}.png'.format(modality, self.iter_num)
            self.imwrite(pred[0], name)

            # image
            name = 'image_{}_{:03}.png'.format(modality, self.iter_num)
            self.imwrite(x[0], name, is_image=True)

            # label
            name = 'label_{}_{:03}.png'.format(modality, self.iter_num)
            self.imwrite(y[0], name)