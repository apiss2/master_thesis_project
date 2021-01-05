import torch
import numpy as np

from .train_util import GANEpoch, set_requires_grad

class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list, optimizer,
                    model_D, loss_D, metrics_D:list, optimizer_D,
                    method:str='DANN', device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D

    def on_epoch_start(self):
        self.model.train()
        self.model_D.train()

    def batch_update(self, batch):
        ### prepare inputs ###
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

        ### update discriminator ###
        self.model.encoder.train()
        self.model.decoder.eval()
        self.model.segmentation_head.eval()
        self.model_D.train()
        for i, x in enumerate([x_A, x_B]):
            y_D = (torch.ones(x.size()[0])*i).to(self.device)
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()
            # predict
            features = self.model.encoder.forward(x)[-1]
            pred_D = self.model_D.forward(features).squeeze()
            loss, _ = self.update_loss(y_D, pred_D, self.loss_D)
            # backward
            loss.backward()
            self.optimizer.step()
            self.optimizer_D.step()
            self.update_metrics(y_D, pred_D, self.metrics_D)

        ### update generator ###
        self.model.train()
        self.model_D.eval()
        for i, (x, y) in enumerate(zip([x_A, x_B], [y_A, y_B])):
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()
            # predict
            pred = self.model.forward(x)
            loss, _ = self.update_loss(y, pred, self.loss)
            # backward
            loss.backward()
            self.optimizer.step()
            self.update_metrics(y, pred, self.metrics)


class ValidEpoch(GANEpoch):
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
        x = torch.cat([x_A, x_B])
        y = torch.cat([y_A, y_B])
        y_D = torch.cat([  torch.ones(x_A.size()[0]),
                            torch.zeros(x_B.size()[0])  ]).to(self.device)

        with torch.no_grad():
            ### update discriminator ###
            # predict
            features = self.model.encoder.forward(x)[-1]
            pred_D = self.model_D.forward(features).squeeze()
            # logging
            self.update_loss(y_D, pred_D, self.loss_D)
            self.update_metrics(y_D, pred_D, self.metrics_D)

            ### update generator ###
            # predict
            pred = self.model.forward(x)
            # logging
            self.update_loss(y, pred, self.loss)
            self.update_metrics(y, pred, self.metrics)
