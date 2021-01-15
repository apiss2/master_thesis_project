import numpy as np

import torch
from torch import autograd

from .train_util import WDGR_Epoch, Tester, UnNormalize


class TrainEpoch(WDGR_Epoch):
    def __init__(self, model, loss, metrics:list, optimizer,
                decoder_1, decoder_2, loss_D, metrics_D:list,
                optimizer_f1, optimizer_f2, device:str='cpu',
                modelupdate_freq: int = 10,
                ):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_1=decoder_1, decoder_2=decoder_2,
            loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_f1 = optimizer_f1
        self.optimizer_f2 = optimizer_f2
        self.modelupdate_freq = modelupdate_freq

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
        # predict
        self.optimizer.zero_grad()
        self.optimizer_f1.zero_grad()
        self.optimizer_f2.zero_grad()
        loss = 0
        for x, y in zip([x_A, x_B], [y_A, y_B]):
            features = self.encoder.forward(x)
            pred_1 = self.decoder_1.forward(features)
            pred_2 = self.decoder_2.forward(features)
            # loss
            loss_tmp, _ = self.update_loss(pred_1, y, self.loss)
            loss += loss_tmp*0.25
            loss_tmp, _ = self.update_loss(pred_2, y, self.loss)
            loss += loss_tmp*0.25
            self.update_metrics(y, pred_1, self.metrics)
            self.update_metrics(y, pred_2, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_f1.step()
        self.optimizer_f2.step()

        ### update classifiers ###
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.optimizer_f1.zero_grad()
        self.optimizer_f2.zero_grad()
        loss = 0
        for x, y in zip([x_A, x_B], [y_A, y_B]):
            with torch.no_grad():
                features = self.encoder.forward(x)
            pred_1 = self.decoder_1.forward(features)
            pred_2 = self.decoder_2.forward(features)
            # loss
            loss_tmp_1, _ = self.update_loss(pred_1, y, self.loss)
            loss_tmp_2, _ = self.update_loss(pred_2, y, self.loss)
            loss_tmp, _ = self.update_loss(pred_1, pred_2, self.loss_D)
            loss -= (loss_tmp_1 + loss_tmp_2) * 0.25 + loss_tmp * 0.5
            self.update_metrics(y, pred_1, self.metrics)
            self.update_metrics(y, pred_2, self.metrics)
            self.update_metrics(pred_1, pred_2, self.metrics_D)
        # backward
        loss.backward()
        self.optimizer_f1.step()
        self.optimizer_f2.step()

        ### update generator ###
        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        self.decoder_1.eval()
        self.decoder_2.eval()
        self.set_requires_grad(self.decoder_1, requires_grad=False)
        self.set_requires_grad(self.decoder_2, requires_grad=False)
        self.optimizer.zero_grad()
        # predict
        pred_A = self.model.forward(x_A)
        pred_B = self.model.forward(x_B)
        loss, _ = self.update_loss(pred_A, pred_B, self.loss_D)
        # backward
        loss.backward()
        self.optimizer.step()
        self.update_metrics(pred_A, pred_B, self.metrics_D)


class ValidEpoch(WDGR_Epoch):
    def __init__(self, model, loss, metrics:list,
                    model_D, loss_D, metrics_D:list,
                    device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='valid', device=device,
        )

    def on_epoch_start(self):
        self.model.eval()
        self.model_D.eval()

    def batch_update(self, batch):
        ### prepare inputs ###
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

        with torch.no_grad():
            for x, y in zip([x_A, x_B], [y_A, y_B]):
                # pred
                features = self.encoder.forward(x)
                pred_1 = self.decoder_1.forward(features)
                pred_2 = self.decoder_2.forward(features)
                # loss
                self.update_loss(pred_1, y, self.loss)
                self.update_loss(pred_2, y, self.loss)
                self.update_loss(pred_1, pred_2, self.loss_D)
                # metric
                self.update_metrics(y, pred_1, self.metrics)
                self.update_metrics(y, pred_2, self.metrics)
                self.update_metrics(pred_1, pred_2, self.metrics_D)


class TestEpoch(Tester):
    def __init__(self, model, loss, metrics: list, device: str = 'cpu',
                    target_modality:str = 'A', save_path: str = None,
                    label_type: str = 'binary_label', color_palette: list = None,
                    mean_A: list = None, std_A: list = None,
                    mean_B: list = None, std_B: list = None):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device,
            label_type=label_type, color_palette=color_palette, save_path=save_path
        )
        assert target_modality == 'A' or target_modality == 'B', 'target_modality must be A or B'
        self.target_modality = target_modality

        self.unorm_A = None if None in [mean_A, std_A] else UnNormalize(mean=mean_A, std=std_A)
        self.unorm_B = None if None in [mean_B, std_B] else UnNormalize(mean=mean_B, std=std_B)

    def batch_update(self, batch):
        self.iter_num += 1
        self.all_logs[self.iter_num] = dict()

        ### prepare inputs ###
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

        for i, (x, y) in enumerate(zip([x_A, x_B], [y_A, y_B])):
            modality = 'A' if i==0 else 'B'
            if not modality in self.target_modality:
                continue

            with torch.no_grad():
                # predict
                pred = self.model.forward(x)
                # logging
                _, loss_value = self.update_loss(y, pred, self.loss)
                metrics_values = self.update_metrics(y, pred, self.metrics)
                self.all_logs[self.iter_num][self.loss.__name__] = loss_value
                self.all_logs[self.iter_num].update(metrics_values)

            if self.save_image:
                # predict image
                name = 'predict_{}_{:03}.png'.format(modality, self.iter_num)
                self.imwrite(pred[0], name)

                # image
                name = 'image_{}_{:03}.png'.format(modality, self.iter_num)
                x = self.unorm_A(x[0]) if modality == 'A' else self.unorm_B(x[0])
                self.imwrite(x, name, is_image=True)

                # label
                name = 'label_{}_{:03}.png'.format(modality, self.iter_num)
                self.imwrite(y[0], name)
