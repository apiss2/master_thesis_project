import sys

import torch
from tqdm import tqdm as tqdm

from ..utils.meter import AverageValueMeter
from .train_util import GANEpoch, Tester
from ..transformation.UnNormalize import UnNormalize


class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list, optimizer, \
                    model_D, loss_D, metrics_D:list, optimizer_D,
                    modelupdate_freq:int=10, discupdate_freq:int=5,
                    geometric_transform=None,
                    segmentation_metrics:list=None, device='cpu'):
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
            if self.iter%self.discupdate_freq==0:
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
            if self.iter%self.discupdate_freq==0:
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
                src_label = y
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
                src_label = y
                # Create pseudo-labels by randomly deforming the image
                tgt_label = self.geometric_transform(src_label, theta)
                # Trim the center of the original image
                pred_label = self.geometric_transform(src_label, pred_theta)
                # metrics
                self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)


class TestEpoch(Tester):
    def __init__(self, model, loss, metrics: list, geometric_transform,
                 target_modality: str = 'A', multi_modality: bool = False,
                 save_path: str = None, mean_A: list = None, std_A: list = None,
                 mean_B: list = None, std_B: list = None,
                 segmentation_metrics: list = None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device, save_path=save_path
        )
        assert target_modality == 'A' or target_modality == 'B', 'target_modality must be A or B'
        self.multi_modality = multi_modality
        self.mono_or_multi = 'multi' if self.multi_modality else 'mono'
        self.target_modality = target_modality

        self.unorm_A = None if None in [mean_A, std_A] else UnNormalize(mean=mean_A, std=std_A)
        self.unorm_B = None if None in [mean_B, std_B] else UnNormalize(mean=mean_B, std=std_B)

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

        theta = batch['theta_A']
        if self.multi_modality and self.target_modality == 'A':
            x_A, x_B, src_label = batch['x_A'], batch['x_B'], batch['y_A']
            tgt_x_A = self.geometric_transform(x_A, theta)
        elif self.multi_modality and self.target_modality == 'B':
            x_A, x_B, src_label = batch['x_A'], batch['x_B'], batch['y_A']
            tgt_x_B = self.geometric_transform(x_B, theta)
        elif not self.multi_modality and self.target_modality=='A':
            x_A, src_label = batch['x_A'], batch['y_A']
            tgt_x_A = self.geometric_transform(x_A, theta)
        elif not self.multi_modality and self.target_modality=='B':
            x_B, src_label = batch['x_B'], batch['y_A']
            tgt_x_B = self.geometric_transform(x_B, theta)
        else:
            assert 'Unexpected modality settings: target={}, multi={}'.format(\
                self.target_modality, self.multi_modality)

        with torch.no_grad():
            if self.multi_modality and self.target_modality == 'A':
                image, tgt_image = x_B, tgt_x_A
                src_modality, tgt_modality = 'B', 'A'
            elif self.multi_modality and self.target_modality == 'B':
                image, tgt_image = x_A, tgt_x_B
                src_modality, tgt_modality = 'A', 'B'
            elif not self.multi_modality and self.target_modality=='A':
                image, tgt_image = x_A, tgt_x_A
                src_modality, tgt_modality = 'A', 'A'
            elif not self.multi_modality and self.target_modality=='B':
                image, tgt_image = x_B, tgt_x_B
                src_modality, tgt_modality = 'B', 'B'
            else:
                assert 'Unexpected modality settings: target={}, multi={}'.format(\
                    self.target_modality, self.multi_modality)
            # predict
            pred_theta = self.model.forward(image, tgt_image)
            # logging
            _, loss_value = self.update_loss(theta, pred_theta, self.loss)
            metrics_values = self.update_metrics(theta, pred_theta, self.metrics)
            self.all_logs[self.iter_num][self.loss.__name__] = loss_value
            self.all_logs[self.iter_num].update(metrics_values)

        predict_image = self.geometric_transform(image, pred_theta)

        if self.segmentation_metrics is not None:
            # Create pseudo-labels by randomly deforming the image
            tgt_label = self.geometric_transform(src_label, theta)
            # Trim the center of the original image
            pred_label = self.geometric_transform(src_label, pred_theta)
            # metrics
            metrics_values = self.update_metrics(tgt_label, pred_label, self.segmentation_metrics)
            self.all_logs[self.iter_num].update(metrics_values)

        if self.save_image:
            # src_image
            name = '{}_{}_src_image_{:03}.png'.format(\
                self.mono_or_multi, self.target_modality, self.iter_num)
            x = self.unorm_A(image[0]) if src_modality == 'A' else self.unorm_B(image[0])
            self.imwrite(x, name, is_image=True)

            # tgt_image
            name = '{}_{}_tgt_image_{:03}.png'.format(\
                self.mono_or_multi, self.target_modality, self.iter_num)
            x = self.unorm_A(tgt_image[0]) if tgt_modality == 'A' else self.unorm_B(tgt_image[0])
            self.imwrite(x, name, is_image=True)

            # predict_image
            name = '{}_{}_predict_image_{:03}.png'.format(\
                self.mono_or_multi, self.target_modality, self.iter_num)
            x = self.unorm_A(predict_image[0]) if src_modality == 'A' else self.unorm_B(predict_image[0])
            self.imwrite(x, name, is_image=True)

            if self.segmentation_metrics is not None:
                # src_label
                name = '{}_{}_src_label_{:03}.png'.format(\
                    self.mono_or_multi, self.target_modality, self.iter_num)
                x = self.unorm_A(src_label[0]) if src_modality == 'A' else self.unorm_B(src_label[0])
                self.imwrite(x, name)

                # tgt_label
                name = '{}_{}_tgt_label_{:03}.png'.format(\
                    self.mono_or_multi, self.target_modality, self.iter_num)
                x = self.unorm_A(tgt_label[0]) if tgt_modality == 'A' else self.unorm_B(tgt_label[0])
                self.imwrite(x, name)

                # predict_label
                name = '{}_{}_predict_label_{:03}.png'.format(\
                    self.mono_or_multi, self.target_modality, self.iter_num)
                x = self.unorm_A(pred_label[0]) if src_modality == 'A' else self.unorm_B(pred_label[0])
                self.imwrite(x, name)
