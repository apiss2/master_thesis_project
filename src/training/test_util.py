import os

import cv2
import numpy as np

import torch

from .train_util import Epoch
from ..utils.utils import onehot2color
from ..utils.meter import AverageValueMeter


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Tester(Epoch):
    def __init__(self, model, loss, metrics:list, device:str='cpu',
                    label_type:str='binary_label', color_palette:list=None,
                    save_path:str=None, mean:list=None, std:list=None):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name='test', device=device,
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

    def imwrite(self, x, name, is_image=False):
        pred_np = x.cpu().detach().numpy().transpose([1,2,0])
        image = pred_np if is_image else self.convert2image(pred_np)
        image = (image * 255).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_image else image
        path = os.path.join(self.save_path, name)
        cv2.imwrite(path, image)

    def convert2image(self, output):
        output = np.where(output>0.5, 1, 0)
        if self.label_type=='binary_label':
            output = output[...,0]
        else:
            output = onehot2color(output, self.color_palette)
        return output


class SegmentationTester(Tester):
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
            name = 'predict_{:03}.png'.format(self.iter_num)
            self.imwrite(pred[0], name)

            # image
            name = 'image_{:03}.png'.format(self.iter_num)
            self.imwrite(self.unorm(x[0]), name, is_image=True)

            # label
            name = 'label_{:03}.png'.format(self.iter_num)
            self.imwrite(y[0], name)


class RegistrationTester(Tester):
    def __init__(self, model, loss, metrics:list, geometric_transform,\
                 save_path:str=None, mean:list=None, std:list=None,
                 metrics_seg:list=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device,
            save_path=save_path, mean=mean, std=std
        )
        self.geometric_transform = geometric_transform
        self.metrics_seg = [metric.to(self.device) for metric in metrics_seg] if metrics_seg is not None else None

    def on_epoch_start(self):
        self.iter_num = 0
        self.model.eval()

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.metrics_seg is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_seg})

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

        if self.metrics_seg is not None:
            src_label = batch['label']
            # Create pseudo-labels by randomly deforming the image
            tgt_label = self.geometric_transform(src_label, theta)
            # Trim the center of the original image
            pred_label = self.geometric_transform(src_label, pred_theta)
            # metrics
            metrics_values = self.update_metrics(tgt_label, pred_label, self.metrics_seg)
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

            if self.metrics_seg is not None:
                # src_label
                name = 'src_label_{:03}.png'.format(self.iter_num)
                self.imwrite(src_label[0], name)

                # tgt_label
                name = 'tgt_label_{:03}.png'.format(self.iter_num)
                self.imwrite(tgt_label[0], name)

