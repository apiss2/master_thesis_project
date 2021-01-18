import numpy as np

import torch
from torch import autograd

from .train_util import WDGREpoch
from ..transformation.affine import AffineGridGen
from ..transformation.tps import TpsGridGen
from ..utils.meter import AverageValueMeter


class TrainEpoch(WDGREpoch):
    def __init__(self, model, loss, metrics:list, optimizer,
                decoder_1, decoder_2, loss_D, metrics_D,
                optimizer_1, optimizer_2, device:str='cpu',
                modelupdate_freq: int = 1, discupdate_freq: int = 1,
                geometric:str='affine',  geometric_transform=None,
                metrics_seg:list=None
                ):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_1=decoder_1, decoder_2=decoder_2,
            loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.modelupdate_freq = modelupdate_freq
        self.discupdate_freq = discupdate_freq

        if geometric=='affine':
            self.geometric_gridgen = AffineGridGen()
        elif geometric == 'tps':
            self.geometric_gridgen = TpsGridGen(device=device)
        else:
            assert False, f'No such geometric: {geometric}'

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

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        ### update generator and classifiers ###
        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        self.decoder_1.train()
        self.set_requires_grad(self.decoder_1, requires_grad=True)
        self.decoder_2.train()
        self.set_requires_grad(self.decoder_2, requires_grad=True)
        # src
        self.optimizer.zero_grad()
        self.optimizer_1.zero_grad()
        # predict
        features_src = self.model.forward(x_A)
        features_tgt = self.model.forward(tgt_x_A)
        pred_theta = self.decoder_1.forward(features_src, features_tgt)
        # loss
        loss, _ = self.update_loss(theta_A, pred_theta, self.loss)
        self.update_metrics(theta_A, pred_theta, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_1.step()
        if self.metrics_seg is not None:
            src_label_A = y_A.float()
            # Create pseudo-labels by randomly deforming the image
            tgt_label_A = self.geometric_transform(src_label_A, theta_A)
            pred_label = self.geometric_transform(src_label_A, pred_theta)
            # metrics
            self.update_metrics(tgt_label_A, pred_label, self.metrics_seg)
        # tgt
        self.optimizer.zero_grad()
        self.optimizer_2.zero_grad()
        # predict
        features_src = self.model.forward(x_B)
        features_tgt = self.model.forward(tgt_x_B)
        pred = self.decoder_2.forward(features_src, features_tgt)
        # loss
        loss, _ = self.update_loss(theta_B, pred, self.loss)
        self.update_metrics(theta_B, pred, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_2.step()
        if self.metrics_seg is not None:
            src_label_B = y_B.float()
            # Create pseudo-labels by randomly deforming the image
            tgt_label_B = self.geometric_transform(src_label_B, theta_B)
            pred_label = self.geometric_transform(src_label_B, pred_theta)
            # metrics
            self.update_metrics(tgt_label_B, pred_label, self.metrics_seg)

        ### update classifiers ###
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.decoder_1.train()
        self.set_requires_grad(self.decoder_1, requires_grad=True)
        self.decoder_2.train()
        self.set_requires_grad(self.decoder_2, requires_grad=True)
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        # src
        src_features = self.model.forward(x_A)
        tgt_features = self.model.forward(tgt_x_A)
        pred_theta_1 = self.decoder_1.forward(src_features, tgt_features)
        pred_theta_2 = self.decoder_2.forward(src_features, tgt_features)
        loss_tmp_1, _ = self.update_loss(theta_A, pred_theta_1, self.loss)
        loss_tmp_2, _ = self.update_loss(theta_A, pred_theta_2, self.loss)
        self.update_metrics(theta_A, pred_theta_1, self.metrics)
        self.update_metrics(theta_A, pred_theta_2, self.metrics)
        # tgt
        src_features = self.model.forward(x_B)
        tgt_features = self.model.forward(tgt_x_B)
        pred_theta_1 = self.decoder_1.forward(src_features, tgt_features)
        pred_theta_2 = self.decoder_2.forward(src_features, tgt_features)
        pred_1 = self.geometric_gridgen(pred_theta_1)
        pred_2 = self.geometric_gridgen(pred_theta_2)
        loss_tmp, _ = self.update_loss(pred_1, pred_2, self.loss_D)
        loss = (loss_tmp_1 + loss_tmp_2)  - loss_tmp
        # backward
        loss.backward()
        self.optimizer_1.step()
        self.optimizer_2.step()

        ### update generator ###
        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        self.decoder_1.eval()
        self.decoder_2.eval()
        self.set_requires_grad(self.decoder_1, requires_grad=False)
        self.set_requires_grad(self.decoder_2, requires_grad=False)
        self.optimizer.zero_grad()
        # predict
        for x, tgt_x in zip([x_A, x_B], [tgt_x_A, tgt_x_B]):
            src_features = self.model.forward(x)
            tgt_features = self.model.forward(tgt_x)
            pred_1 = self.decoder_1.forward(src_features, tgt_features)
            pred_2 = self.decoder_2.forward(src_features, tgt_features)
            pred_1 = self.geometric_gridgen(pred_1)
            pred_2 = self.geometric_gridgen(pred_2)
            loss, _ = self.update_loss(pred_1, pred_2, self.loss_D)
            self.update_metrics(pred_1, pred_2, self.metrics_D)
            # backward
            loss.backward()
            self.optimizer.step()


class ValidEpoch(WDGREpoch):
    def __init__(self, model, loss, metrics:list,
                    decoder_1, decoder_2, loss_D, metrics_D:list,
                    geometric:str, geometric_transform,
                    metrics_seg:list=None, device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_1=decoder_1, decoder_2=decoder_2, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='valid', device=device,
        )
        if geometric=='affine':
            self.geometric_gridgen = AffineGridGen()
        elif geometric == 'tps':
            self.geometric_gridgen = TpsGridGen(device=device)
        else:
            assert False, f'No such geometric: {geometric}'

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
        self.decoder_1.eval()
        self.decoder_2.eval()

    def batch_update(self, batch):
        ### prepare inputs ###
        x_A, y_A, theta = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, _ = batch['x_B'], batch['y_B'], batch['theta_B']
        tgt_x_A = self.geometric_transform(x_A, theta)
        tgt_x_B = self.geometric_transform(x_B, theta)

        with torch.no_grad():
            for x, y, tgt_x in zip([x_A, x_B], [y_A, y_B], [tgt_x_A, tgt_x_B]):
                # pred
                src_features = self.model.forward(x)
                tgt_features = self.model.forward(tgt_x)
                pred_1 = self.decoder_1.forward(src_features, tgt_features)
                pred_2 = self.decoder_2.forward(src_features, tgt_features)
                # loss
                self.update_loss(theta, pred_1, self.loss)
                self.update_loss(theta, pred_2, self.loss)
                # metric
                self.update_metrics(theta, pred_1, self.metrics)
                self.update_metrics(theta, pred_2, self.metrics)
                if self.metrics_seg is not None:
                    src_label = y.float()
                    # Create pseudo-labels by randomly deforming the image
                    tgt_label = self.geometric_transform(src_label, theta)
                    pred_label_1 = self.geometric_transform(src_label, pred_1)
                    pred_label_2 = self.geometric_transform(src_label, pred_1)
                    # metrics
                    self.update_metrics(tgt_label, pred_label_1, self.metrics_seg)
                    self.update_metrics(tgt_label, pred_label_2, self.metrics_seg)
                # distance loss and metric
                pred_1 = self.geometric_gridgen(pred_1)
                pred_2 = self.geometric_gridgen(pred_2)
                self.update_loss(pred_1, pred_2, self.loss_D)
                self.update_metrics(pred_1, pred_2, self.metrics_D)
