import sys

import torch
from tqdm import tqdm as tqdm

from ..utils.meter import AverageValueMeter
from .train_util import GANEpoch


class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics: list, optimizer, \
                    model_D, loss_D, metrics_D: list, optimizer_D,
                    modelupdate_freq: int = 10, discupdate_freq: int = 5,
                    geometric_transform=None,
                    metrics_seg: list = None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.modelupdate_freq = modelupdate_freq
        self.discupdate_freq = discupdate_freq

        self.geometric_transform = geometric_transform
        self.metrics_seg = [metric.to(self.device) for metric in metrics_seg] if metrics_seg is not None else None

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        self.loss_meters.update({self.loss_D.__name__: AverageValueMeter()})
        self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_D})
        if self.metrics_seg is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_seg})

    def on_epoch_start(self):
        self.iter = 0
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']

        ### update discriminator ###
        if self.iter % self.modelupdate_freq == 0:
            self.model.encoder.train()
            self.set_requires_grad(self.model, requires_grad=True)
        if self.iter%self.discupdate_freq == 0:
            self.model_D.train()
            self.set_requires_grad(self.model_D, requires_grad=True)
        for i, x in enumerate([x_A, x_B]):
            #y_D = (torch.rand(x.size()[0]) * 0.2 + 0.8 * i).to(self.device)
            y_D = (torch.ones(x.size()[0])*i).to(self.device)
            half = (torch.ones(x.size()[0])/2).to(self.device)
            if self.iter % self.modelupdate_freq == 0:
                self.optimizer.zero_grad()
            if self.iter % self.discupdate_freq == 0:
                self.optimizer_D.zero_grad()
            # predict
            features = self.model.encoder.forward(x)
            pred_D = self.model_D.forward(features).squeeze()
            loss, _ = self.update_loss(y_D, pred_D, self.loss_D)
            loss += torch.mean(half - torch.abs(pred_D-half))
            # backward
            loss.backward()
            if self.iter % self.modelupdate_freq == 0:
                self.optimizer.step()
            if self.iter % self.discupdate_freq == 0:
                self.optimizer_D.step()
            self.update_metrics(y_D, pred_D, self.metrics_D)

        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)
        for i, (x, y, theta) in enumerate(zip([x_A, x_B], [y_A, y_B], [theta_A, theta_B])):
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


class ValidEpoch(GANEpoch):
    def __init__(self, model, loss, metrics: list, \
                model_D, loss_D, metrics_D: list,
                geometric_transform, metrics_seg: list = None,
                device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='valid', device=device,
        )
        self.geometric_transform = geometric_transform
        self.metrics_seg = [metric.to(self.device) for metric in metrics_seg] if metrics_seg is not None else None

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        self.loss_meters.update({self.loss_D.__name__: AverageValueMeter()})
        self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_D})
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
                ### update discriminator ###
                # predict
                features = self.model.encoder.forward(x)
                pred_D = self.model_D.forward(features).squeeze()
                # logging
                self.update_loss(y_D, pred_D, self.loss_D)
                self.update_metrics(y_D, pred_D, self.metrics_D)

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

