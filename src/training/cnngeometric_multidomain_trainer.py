import sys

import torch
from tqdm import tqdm as tqdm

from ..utils.meter import AverageValueMeter
from .train_util import Epoch


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics: list, optimizer, \
                    geometric_transform=None, freeze_encoder:bool=False,
                    metrics_seg: list = None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.freeze_encoder = freeze_encoder

        self.geometric_transform = geometric_transform
        self.metrics_seg = [metric.to(self.device) for metric in metrics_seg] if metrics_seg is not None else None

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.metrics_seg is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_seg})

    def on_epoch_start(self):
        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        if self.freeze_encoder:
            self.model.encoder.eval()
            self.set_requires_grad(self.model.encoder, requires_grad=False)

    def batch_update(self, batch):
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        x = torch.cat([x_A, x_B])
        y = torch.cat([y_A, y_B])
        theta = torch.cat([theta_A, theta_B])

        ### update discriminator ###
        tgt_x = self.geometric_transform(x, theta)
        self.optimizer.zero_grad()
        ### update generator ###
        # predict
        pred_theta = self.model.forward(x, tgt_x)
        loss, _ = self.update_loss(theta, pred_theta, self.loss)
        # backward
        loss.backward()
        self.optimizer.step()
        self.update_metrics(theta, pred_theta, self.metrics)

        if self.metrics_seg is not None:
            src_label = y.float()
            # Create pseudo-labels by randomly deforming the image
            tgt_label = self.geometric_transform(src_label, theta)
            # Trim the center of the original image
            pred_label = self.geometric_transform(src_label, pred_theta)
            # metrics
            self.update_metrics(tgt_label, pred_label, self.metrics_seg)


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics: list, geometric_transform,\
                freeze_encoder:bool=False, metrics_seg: list = None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='valid', device=device,
        )
        self.geometric_transform = geometric_transform
        self.metrics_seg = [metric.to(self.device) for metric in metrics_seg] if metrics_seg is not None else None

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.metrics_seg is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_seg})

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, batch):
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']

        for i, (x, y, theta) in enumerate(zip([x_A, x_B], [y_A, y_B], [theta_A, theta_B])):
            y_D = (torch.ones(x.size()[0])*i).to(self.device)
            tgt_x = self.geometric_transform(x, theta)
            with torch.no_grad():
                ### update generator ###
                # predict
                pred_theta = self.model.forward(x, tgt_x)
                # logging
                self.update_loss(theta, pred_theta, self.loss)
                self.update_metrics(theta, pred_theta, self.metrics)

            if self.metrics_seg is not None:
                src_label = y.float()
                # Create pseudo-labels by randomly deforming the image
                tgt_label = self.geometric_transform(src_label, theta)
                # Trim the center of the original image
                pred_label = self.geometric_transform(src_label, pred_theta)
                # metrics
                self.update_metrics(tgt_label, pred_label, self.metrics_seg)

