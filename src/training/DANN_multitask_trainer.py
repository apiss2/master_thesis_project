import torch
import numpy as np

from .train_util import MultiTaskEpoch


class TrainEpoch(MultiTaskEpoch):
    def __init__(self, model, loss, metrics: list, optimizer, \
                    decoder_seg, loss_seg, metrics_seg: list, optimizer_seg, \
                    model_D, loss_D, metrics_D: list, optimizer_D,
                    modelupdate_freq: int = 1, discupdate_freq: int = 1,
                    geometric_transform=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_seg=decoder_seg, loss_seg=loss_seg, metrics_seg=metrics_seg,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            device=device, stage_name='train'
        )
        self.optimizer = optimizer
        self.optimizer_seg = optimizer_seg
        self.optimizer_D = optimizer_D
        self.modelupdate_freq = modelupdate_freq
        self.discupdate_freq = discupdate_freq

        self.geometric_transform = geometric_transform

    def on_epoch_start(self):
        self.iter = 0
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.decoder_seg.eval()
        self.set_requires_grad(self.decoder_seg, requires_grad=False)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']

        ### update discriminator ###
        if self.iter % self.modelupdate_freq == 0:
            self.model.encoder.train()
            self.set_requires_grad(self.model, requires_grad=True)
        if self.iter%self.discupdate_freq == 0:
            self.model_D.train()
            self.set_requires_grad(self.model_D, requires_grad=True)
        for i, x in enumerate([x_A, x_B]):
            #y_D = (torch.rand(x.size()[0]) * 0.2 + 0.8 * i).to(self.device)
            y_D = (torch.ones(x.size()[0])*i).to(self.device)
            half = (torch.ones(x.size()[0])/2).to(self.device)
            if self.iter % self.modelupdate_freq == 0:
                self.optimizer.zero_grad()
            if self.iter % self.discupdate_freq == 0:
                self.optimizer_D.zero_grad()
            # predict
            features = self.model.encoder.forward(x)
            pred_D = self.model_D.forward(features[-1]).squeeze()
            loss, _ = self.update_loss(y_D, pred_D, self.loss_D)
            loss += torch.mean(half - torch.abs(pred_D-half))
            # backward
            loss.backward()
            if self.iter % self.modelupdate_freq == 0:
                self.optimizer.step()
            if self.iter % self.discupdate_freq == 0:
                self.optimizer_D.step()
            self.update_metrics(y_D, pred_D, self.metrics_D)

        self.set_requires_grad(self.model_D, requires_grad=False)
        self.model_D.eval()

        ### segmentation ###
        self.set_requires_grad(self.model, requires_grad=False)
        self.model.eval()
        self.set_requires_grad(self.model.encoder, requires_grad=True)
        self.model.encoder.train()
        self.set_requires_grad(self.decoder_seg, requires_grad=True)
        self.decoder_seg.train()

        for x, y in zip([x_A, x_B], [y_A, y_B]):
            self.optimizer.zero_grad()
            self.optimizer_seg.zero_grad()
            # predict
            features = self.model.encoder.forward(x)
            pred = self.decoder_seg.forward(features)
            loss, _ = self.update_loss(y, pred, self.loss_seg)
            # backward
            loss.backward()
            self.optimizer.step()
            self.optimizer_seg.step()
            self.update_metrics(y, pred, self.metrics_seg)

        ### registration ###
        self.set_requires_grad(self.model, requires_grad=True)
        self.model.train()
        self.set_requires_grad(self.decoder_seg, requires_grad=False)
        self.decoder_seg.eval()

        for i, (x, theta) in enumerate(zip([x_A, x_B], [theta_A, theta_B])):
            tgt_x = self.geometric_transform(x, theta)
            self.optimizer.zero_grad()
            ### update generator ###
            # predict
            pred_theta = self.model.forward(x, tgt_x)
            loss, _ = self.update_loss(theta, pred_theta, self.loss)
            # backward
            loss.backward()
            self.optimizer.step()
            self.update_metrics(theta, pred_theta, self.metrics)


class ValidEpoch(MultiTaskEpoch):
    def __init__(self, model, loss, metrics: list,\
                    decoder_seg, loss_seg, metrics_seg: list,\
                    model_D, loss_D, metrics_D: list,
                    geometric_transform=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_seg=decoder_seg, loss_seg=loss_seg, metrics_seg=metrics_seg,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            device=device, stage_name='train'
        )
        self.geometric_transform = geometric_transform

    def on_epoch_start(self):
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.decoder_seg.eval()
        self.set_requires_grad(self.decoder_seg, requires_grad=False)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)

    def batch_update(self, batch):
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']

        with torch.no_grad():
            for i, (x, y, theta) in enumerate(zip([x_A, x_B], [y_A, y_B], [theta_A, theta_B])):
                ### discriminator ###
                y_D = (torch.ones(x.size()[0])*i).to(self.device)
                features = self.model.encoder.forward(x)
                pred_D = self.model_D.forward(features[-1]).squeeze()
                loss, _ = self.update_loss(y_D, pred_D, self.loss_D)
                self.update_metrics(y_D, pred_D, self.metrics_D)

                ### segmentation ###
                features = self.model.encoder.forward(x)
                pred = self.decoder_seg.forward(features)
                loss, _ = self.update_loss(y, pred, self.loss_seg)
                self.update_metrics(y, pred, self.metrics_seg)

                ### registration ###
                tgt_x = self.geometric_transform(x, theta)
                pred_theta = self.model.forward(x, tgt_x)
                loss, _ = self.update_loss(theta, pred_theta, self.loss)
                self.update_metrics(theta, pred_theta, self.metrics)