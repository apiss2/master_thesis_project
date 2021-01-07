import os
import sys

import cv2
import torch
import numpy as np
from tqdm import tqdm as tqdm

from ..utils.meter import AverageValueMeter
from ..transformation.UnNormalize import UnNormalize
from ..utils.utils import onehot2color

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

class Epoch:
    def __init__(self, model, loss, metrics:list,
            stage_name:str='train', device:str='cpu'):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.device = device
        self.verbose = True

        self._to_device()

    def _to_device(self):
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        self.metrics = [metric.to(self.device) for metric in self.metrics]

    def _format_logs(self):
        str_logs = ['{} - {:02.4f}'.format(k, v) for k, v in self.logs.items()]
        s = ', '.join(str_logs)
        return s

    def _initIteration(self):
        self.on_epoch_start()
        self.logs = {}
        self.reset_meters()

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

    def update_loss(self, y, y_pred, loss_fn):
        loss = loss_fn(y_pred, y)
        loss_name = loss_fn.__name__
        loss_value = loss.cpu().detach().numpy()

        self.loss_meters[loss_name].add(loss_value)
        loss_logs = {loss_name: self.loss_meters[loss_name].mean}
        self.logs.update(loss_logs)
        return loss, float(loss_value)

    def update_metrics(self, y, y_pred, metrics):
        metrics_logs = dict()
        metrics_values = dict()
        for metric_fn in metrics:
            metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
            self.metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs[metric_fn.__name__] = self.metrics_meters[metric_fn.__name__].mean
            metrics_values[metric_fn.__name__] = float(metric_value)
        self.logs.update(metrics_logs)
        return metrics_values

    def on_epoch_start(self):
        raise NotImplementedError

    def batch_update(self, batch):
        raise NotImplementedError

    def run(self, dataloader):
        self._initIteration()
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for batch in iterator:
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)

                self.batch_update(batch)

                s = self._format_logs()
                iterator.set_postfix_str(s)

        return self.logs

class GANEpoch(Epoch):
    def __init__(self, model, loss, metrics:list,
                    model_D, loss_D, metrics_D:list,
                    device:str='cpu', stage_name='train'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name=stage_name, device=device,
        )
        self.model_D = model_D
        self.loss_D = loss_D
        self.metrics_D = metrics_D
        self._to_device_D()

    def _to_device_D(self):
        self.model_D = self.model_D.to(self.device)
        self.loss_D = self.loss_D.to(self.device)
        self.metrics_D = [metric.to(self.device) for metric in self.metrics_D]

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        self.loss_meters.update({self.loss_D.__name__: AverageValueMeter()})
        self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_D})

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

    def imwrite(self, x, name, is_image=False, unnorm=False):
        if is_image and self.unorm is not None:
            x = self.unorm(x)
        pred_np = x.cpu().detach().numpy().transpose([1,2,0])
        image = pred_np if is_image else self.convert2image(pred_np)
        image = (image * 255).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.save_path, name)
        cv2.imwrite(path, image)

    def convert2image(self, output):
        output = np.where(output>0.5, 1, 0)
        if self.label_type=='binary_label':
            output = output[...,0]
        else:
            output = onehot2color(output, self.color_palette)
        return output
