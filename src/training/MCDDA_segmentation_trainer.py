import numpy as np

import torch
from torch import autograd

from .train_util import WDGREpoch


class TrainEpoch(WDGREpoch):
    def __init__(self, model, loss, metrics:list, optimizer,
                decoder_1, decoder_2, loss_D, metrics_D:list,
                optimizer_1, optimizer_2, device:str='cpu',
                modelupdate_freq: int = 1, discupdate_freq: int = 1,
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

    def on_epoch_start(self):
        self.iter = 0
        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        self.decoder_1.train()
        self.set_requires_grad(self.decoder_1, requires_grad=True)
        self.decoder_2.train()
        self.set_requires_grad(self.decoder_2, requires_grad=True)

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

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
        features = self.model.forward(x_A)
        pred = self.decoder_1.forward(features)
        # loss
        loss, _ = self.update_loss(y_A, pred, self.loss)
        self.update_metrics(y_A, pred, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_1.step()
        # tgt
        self.optimizer.zero_grad()
        self.optimizer_2.zero_grad()
        # predict
        features = self.model.forward(x_B)
        pred = self.decoder_2.forward(features)
        # loss
        loss, _ = self.update_loss(y_B, pred, self.loss)
        self.update_metrics(y_B, pred, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_2.step()

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
        features = self.model.forward(x_A)
        pred_1 = self.decoder_1.forward(features)
        pred_2 = self.decoder_2.forward(features)
        loss_tmp_1, _ = self.update_loss(y_A, pred_1, self.loss)
        loss_tmp_2, _ = self.update_loss(y_A, pred_2, self.loss)
        self.update_metrics(y_A, pred_1, self.metrics)
        self.update_metrics(y_A, pred_2, self.metrics)
        # tgt
        features = self.model.forward(x_B)
        pred_1 = self.decoder_1.forward(features)
        pred_2 = self.decoder_2.forward(features)
        loss_tmp, _ = self.update_loss(pred_1, pred_2, self.loss_D)
        loss = (loss_tmp_1 + loss_tmp_2)  - loss_tmp
        self.update_metrics(pred_1, pred_2, self.metrics_D)
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
        for x in [x_A, x_B]:
            features = self.model.forward(x)
            pred_1 = self.decoder_1.forward(features)
            pred_2 = self.decoder_2.forward(features)
            loss, _ = self.update_loss(pred_1, pred_2, self.loss_D)
            self.update_metrics(pred_1, pred_2, self.metrics_D)
            # backward
            loss.backward()
            self.optimizer.step()


class ValidEpoch(WDGREpoch):
    def __init__(self, model, loss, metrics:list,
                    decoder_1, decoder_2, loss_D, metrics_D:list,
                    device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_1=decoder_1, decoder_2=decoder_2, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='valid', device=device,
        )

    def on_epoch_start(self):
        self.model.eval()
        self.decoder_1.eval()
        self.decoder_2.eval()

    def batch_update(self, batch):
        ### prepare inputs ###
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

        with torch.no_grad():
            for x, y in zip([x_A, x_B], [y_A, y_B]):
                # pred
                features = self.model.forward(x)
                pred_1 = self.decoder_1.forward(features)
                pred_2 = self.decoder_2.forward(features)
                # loss
                self.update_loss(y, pred_1, self.loss)
                self.update_loss(y, pred_2, self.loss)
                self.update_loss(pred_1, pred_2, self.loss_D)
                # metric
                self.update_metrics(y, pred_1, self.metrics)
                self.update_metrics(y, pred_2, self.metrics)
                self.update_metrics(pred_1, pred_2, self.metrics_D)

