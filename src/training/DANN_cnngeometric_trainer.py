import sys
import torch
from tqdm import tqdm as tqdm

from .train_util import GANEpoch
from ..utils.meter import AverageValueMeter

class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list, optimizer, \
                    model_D, loss_D, metrics_D:list, optimizer_D,
                    modelupdate_freq:int=10, geometric_transform=None,
                    segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.modelupdate_freq = modelupdate_freq

        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

    def on_epoch_start(self):
        self.iter = 0
        self.model.train()
        self.model_D.train()

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']

        for i, (x, y, theta) in enumerate(zip([x_A, x_B], [y_A, y_B], [theta_A, theta_B])):
            y_D = (torch.ones(x.size()[0])*i).to(self.device)
            tgt_x = self.geometric_transform(x, theta)
            ### update discriminator ###
            self.model.eval()
            if self.iter%self.modelupdate_freq==0:
                self.model.FeatureExtraction.train()
                self.optimizer.zero_grad()
            self.model_D.train()
            self.optimizer_D.zero_grad()
            # predict
            features = self.model.FeatureExtraction.forward(x)[-1]
            pred_D = self.model_D.forward(features).squeeze()
            loss, _ = self.update_loss(y_D, pred_D, self.loss_D)
            # backward
            loss.backward()
            if self.iter%self.modelupdate_freq==0:
                self.optimizer.step()
            self.optimizer_D.step()
            self.update_metrics(y_D, pred_D, self.metrics_D)

            ### update generator ###
            self.model.train()
            self.model_D.eval()
            self.optimizer.zero_grad()
            # predict
            pred_theta = self.model.forward(x, tgt_x)
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

class ValidEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list, \
                    model_D, loss_D, metrics_D:list,
                geometric_transform, segmentation_metrics:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='valid', device=device,
        )
        self.geometric_transform = geometric_transform
        self.segmentation_metrics = [metric.to(self.device) for metric in segmentation_metrics] if segmentation_metrics is not None else None

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
                features = self.model.FeatureExtraction.forward(x)[-1]
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

            if self.segmentation_metrics is not None:
                src_label = batch['label']
                # Create pseudo-labels by randomly deforming the image
                tgt_label = self.geometric_transform(src_label, theta)
                # Trim the center of the original image
                pred_label = self.geometric_transform(src_label, pred_theta)
                # metrics
                self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)

