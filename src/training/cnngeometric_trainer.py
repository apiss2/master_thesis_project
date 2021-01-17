import torch
from tqdm import tqdm as tqdm

from .train_util import Epoch
from ..utils.meter import AverageValueMeter

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, optimizer, \
        geometric_transform, segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.segmentation_metrics is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.segmentation_metrics})

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, batch):
        image, theta = batch['image'], batch['theta']
        tgt_image = self.geometric_transform(image, theta)

        self.optimizer.zero_grad()
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
            tgt_label = self.geometric_transform(src_label, theta).long()
            # Trim the center of the original image
            pred_label = self.geometric_transform(src_label, pred_theta)
            # metrics
            self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics:list, \
                geometric_transform, segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='train', device=device,
        )
        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

    def on_epoch_start(self):
        self.model.eval()

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.segmentation_metrics is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.segmentation_metrics})

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


