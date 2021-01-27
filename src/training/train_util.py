import sys

import torch
from tqdm import tqdm as tqdm

from ..utils.meter import AverageValueMeter


class Epoch:
    """This is the code written with reference to https://github.com/qubvel/segmentation_models.pytorch"""
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

    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

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


class WDGREpoch(Epoch):
    def __init__(self, model, loss, metrics:list,
                    decoder_1, decoder_2, loss_D, metrics_D:list,
                    device:str='cpu', stage_name='train'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            stage_name=stage_name, device=device,
        )
        self.decoder_1 = decoder_1
        self.decoder_2 = decoder_2
        self.loss_D = loss_D
        self.metrics_D = metrics_D
        self._to_device_D()

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        self.loss_meters.update({self.loss_D.__name__: AverageValueMeter()})
        self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_D})

    def _to_device_D(self):
        self.decoder_1 = self.decoder_1.to(self.device)
        self.decoder_2 = self.decoder_2.to(self.device)
        self.loss_D = self.loss_D.to(self.device)
        self.metrics_D = [metric.to(self.device) for metric in self.metrics_D]


class MultiTaskEpoch(Epoch):
    """This is the code written with reference to https://github.com/qubvel/segmentation_models.pytorch"""
    def __init__(self, model, loss, metrics: list,
            decoder_seg, loss_seg, metrics_seg:list,
            model_D, loss_D, metrics_D:list,
            device:str='cpu', stage_name='train'
            ):
        self.model = model
        self.loss = loss
        self.metrics = metrics

        self.decoder_seg = decoder_seg
        self.loss_seg = loss_seg
        self.metrics_seg = metrics_seg

        self.model_D = model_D
        self.loss_D = loss_D
        self.metrics_D = metrics_D

        self.stage_name = stage_name
        self.device = device
        self.verbose = True
        self._to_device()

    def _to_device(self):
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        self.metrics = [metric.to(self.device) for metric in self.metrics]

        self.decoder_seg = self.decoder_seg.to(self.device)
        self.loss_seg = self.loss_seg.to(self.device)
        self.metrics_seg = [metric.to(self.device) for metric in self.metrics_seg]

        self.model_D = self.model_D.to(self.device)
        self.loss_D = self.loss_D.to(self.device)
        self.metrics_D = [metric.to(self.device) for metric in self.metrics_D]

    def reset_meters(self):
        losses = [self.loss, self.loss_seg, self.loss_D]
        self.loss_meters = {loss.__name__: AverageValueMeter() for loss in losses}
        metrics = self.metrics + self.metrics_seg + self.metrics_D
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

