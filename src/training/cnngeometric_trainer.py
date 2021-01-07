import torch
from tqdm import tqdm as tqdm

from .train_util import Epoch, Tester
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

class TestEpoch(Tester):
    def __init__(self, model, loss, metrics:list, geometric_transform,\
                 save_path:str=None, mean:list=None, std:list=None,
                 segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device,
            save_path=save_path, mean=mean, std=std
        )
        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

    def on_epoch_start(self):
        self.iter_num = 0
        self.model.eval()

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.segmentation_metrics is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.segmentation_metrics})

    def batch_update(self, batch):
        self.iter_num += 1
        self.all_logs[self.iter_num] = dict()

        image, theta = batch['image'], batch['theta']
        tgt_image = self.geometric_transform(image, theta)

        with torch.no_grad():
            # predict
            pred_theta = self.model.forward(image, tgt_image)
            # logging
            _, loss_value = self.update_loss(theta, pred_theta, self.loss)
            metrics_values = self.update_metrics(theta, pred_theta, self.metrics)
            self.all_logs[self.iter_num][self.loss.__name__] = loss_value
            self.all_logs[self.iter_num].update(metrics_values)

        predict_image = self.geometric_transform(image, pred_theta)

        if self.segmentation_metrics is not None:
            src_label = batch['label']
            # Create pseudo-labels by randomly deforming the image
            tgt_label = self.geometric_transform(src_label, theta)
            # Trim the center of the original image
            pred_label = self.geometric_transform(src_label, pred_theta)
            # metrics
            metrics_values = self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)
            self.all_logs[self.iter_num].update(metrics_values)

        if self.save_image:
            # src_image
            name = 'src_image_{:03}.png'.format(self.iter_num)
            self.imwrite(image[0], name, is_image=True)

            # tgt_image
            name = 'tgt_image_{:03}.png'.format(self.iter_num)
            self.imwrite(tgt_image[0], name, is_image=True)

            # predict_image
            name = 'predict_image_{:03}.png'.format(self.iter_num)
            self.imwrite(predict_image[0], name, is_image=True)

            if self.segmentation_metrics is not None:
                # src_label
                name = 'src_label_{:03}.png'.format(self.iter_num)
                self.imwrite(src_label[0], name)

                # tgt_label
                name = 'tgt_label_{:03}.png'.format(self.iter_num)
                self.imwrite(tgt_label[0], name)

