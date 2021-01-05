import os
import cv2
import torch
import numpy as np

from .train_util import Epoch
from ..transformation.UnNormalize import UnNormalize
from ..utils.utils import onehot2color

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

class TestEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, device:str='cpu', \
                    label_type:str='binary_label', color_palette:list=None,
                    save_path:str=None, mean:list=None, std:list=None):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='test', device=device
        )
        self.label_type = label_type
        self.save_image = False if save_path is None else True
        self.save_path = save_path
        self.color_palette = color_palette
        self.all_logs = dict()
        self.unorm = None if None in [mean, std] else UnNormalize(mean=mean, std=std)

    def on_epoch_start(self):
        self.iter_num = 0
        self.model.eval()

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
            pred_np = pred[0].cpu().detach().numpy().transpose([1,2,0])
            image = self.convert2image(pred_np)
            image = (image*255).astype('uint8')
            path = os.path.join(self.save_path, 'predict_{:03}.png'.format(self.iter_num))
            cv2.imwrite(path, image)

            # image
            image = self.unorm(x[0]).cpu().detach().numpy().transpose([1,2,0])
            image = (image*255).astype('uint8')
            path = os.path.join(self.save_path, 'image_{:03}.png'.format(self.iter_num))
            cv2.imwrite(path, image)

            # label
            image = y[0].cpu().detach().numpy().transpose([1,2,0])
            image = self.convert2image(image)
            image = (image*255).astype('uint8')
            path = os.path.join(self.save_path, 'label_{:03}.png'.format(self.iter_num))
            cv2.imwrite(path, image)

    def convert2image(self, output):
        output = np.where(output>0.5, 1, 0)
        if self.label_type=='binary_label':
            output = output[...,0]
        else:
            output = onehot2color(output, self.color_palette)
        return output