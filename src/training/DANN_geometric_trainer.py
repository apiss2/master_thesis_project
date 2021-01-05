import sys
import torch
from tqdm import tqdm as tqdm

from .train_util import Epoch
from ..utils.meter import AverageValueMeter

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, optimizer, \
                    model_D, loss_D, metrics_D:list, optimizer_D,
        geometric_transform, segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, batch):
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']
        theta = batch['theta']

        ### update discriminator ###
        for i, x in enumerate([x_A, x_B]):
            y_D = (torch.ones(x.size()[0])*i).to(self.device)
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()
            # predict
            pred_theta = self.model.forward(image, tgt_image)
            loss, _ = self.update_loss(theta, pred_theta, self.loss)
            # backward
            loss.backward()
            self.optimizer.step()
            self.update_metrics(theta, pred_theta, self.metrics)

            if self.segmentation_metrics is not None:
                src_label = batch['label']
                # Create pseudo-labels by randomly deforming the image
                tgt_label = self.geometric_transform(src_label, theta)
                # Trim the center of the original image
                pred_label = self.geometric_transform(src_label, pred_theta)
                # metrics
                self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, \
                    model_D, loss_D, metrics_D:list,
                geometric_transform, segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='train', device=device,
        )
        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, batch):
        image, theta = batch['image'], batch['theta']
        tgt_image = self.geometric_transform(image, theta)
        with torch.no_grad():
            # predict
            pred_theta = self.model.forward(image, tgt_image)
            # logging
            loss, _ = self.update_loss(theta, pred_theta, self.loss)
            self.update_metrics(theta, pred_theta, self.metrics)

            if self.segmentation_metrics is not None:
                src_label = batch['label']
                # Create pseudo-labels by randomly deforming the image
                tgt_label = self.geometric_transform(src_label, theta)
                # Trim the center of the original image
                pred_label = self.geometric_transform(src_label, pred_theta)
                # metrics
                self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)

