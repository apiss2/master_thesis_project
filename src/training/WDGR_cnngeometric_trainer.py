import numpy as np

import torch
from torch import autograd

from ..utils.meter import AverageValueMeter
from .train_util import GANEpoch


def compute_gradient_penalty(model_D, real, fake, device):
    '''code from https://github.com/donand/GAN_pytorch/blob/master/WGAN-GP/wgan_gp.py'''
    # Compute the sample as a linear combination
    alpha = torch.rand(real.shape[0], 1, 1, 1).to(device)
    alpha = alpha.expand_as(real)
    x_hat = alpha * real + (1 - alpha) * fake
    # Compute the output
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    out = model_D(x_hat)
    # compute the gradient relative to the new sample
    gradients = torch.autograd.grad(
        outputs=out,
        inputs=x_hat,
        grad_outputs=torch.ones(out.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    # Reshape the gradients to take the norm
    gradients = gradients.view(gradients.shape[0], -1)
    # Compute the gradient penalty
    penalty = (gradients.norm(2, dim=1) - 1) ** 2
    return penalty.mean()


class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics: list, optimizer,
                    model_D, loss_D, metrics_D: list, optimizer_D,
                    modelupdate_freq: int = 10, discupdate_freq: int = 10,
                    device: str = 'cpu', geometric_transform=None,
                    metrics_seg: list = None):
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
        self.model.train()
        self.model_D.train()

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        y_A_D = (torch.ones(x_A.size()[0])).to(self.device)
        y_B_D = (torch.zeros(x_A.size()[0])).to(self.device)
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        ### update discriminator ###
        if self.iter % self.discupdate_freq == 0:
            self.model.eval()
            self.set_requires_grad(self.model, requires_grad=False)
            self.model_D.train()
            self.set_requires_grad(self.model_D, requires_grad=True)
            self.optimizer_D.zero_grad()
            # predict
            with torch.no_grad():
                features_A = self.model.encoder.forward(x_A)
                features_B = self.model.encoder.forward(x_B)
            gp = compute_gradient_penalty(self.model_D, features_A, features_B, self.device)
            pred_A_D = self.model_D.forward(features_A).squeeze()
            pred_B_D = self.model_D.forward(features_B).squeeze()
            # NOTE
            # The Wasserstein distance is squared
            # so that it does not diverge to negative values.
            # Note that this is different from the official implementation.
            wasserstein_distance = (pred_A_D.mean() - pred_B_D.mean())**2
            # loss
            loss, _ = self.update_loss(gp, wasserstein_distance, self.loss_D)
            # backward
            loss.backward()
            self.optimizer_D.step()
            self.update_metrics(gp, wasserstein_distance, self.metrics_D)

        ### update generator ###
        if self.iter % self.modelupdate_freq == 0:
            self.set_requires_grad(self.model, requires_grad=True)
            self.model.train()
            self.set_requires_grad(self.model_D, requires_grad=False)
            self.model_D.eval()
            self.optimizer.zero_grad()
            # predict
            pred_theta_A = self.model.forward(x_A, tgt_x_A)
            loss_A, _ = self.update_loss(theta_A, pred_theta_A, self.loss)
            pred_theta_B = self.model.forward(x_B, tgt_x_B)
            loss_B, _ = self.update_loss(theta_B, pred_theta_B, self.loss)
            loss = loss_A + loss_B
            # backward
            loss.backward()
            self.optimizer.step()
            self.update_metrics(theta_A, pred_theta_A, self.metrics)
            self.update_metrics(theta_B, pred_theta_B, self.metrics)

            if self.metrics_seg is not None:
                src_label_A, src_label_B = y_A.float(), y_B.float()
                # Create pseudo-labels by randomly deforming the image
                tgt_label_A = self.geometric_transform(src_label_A, theta_A)
                tgt_label_B = self.geometric_transform(src_label_B, theta_B)
                # Trim the center of the original image
                pred_label_A = self.geometric_transform(src_label_A, pred_theta_A)
                pred_label_B = self.geometric_transform(src_label_B, pred_theta_B)
                # metrics
                self.update_metrics(tgt_label_A, pred_label_A, self.metrics_seg)
                self.update_metrics(tgt_label_B, pred_label_B, self.metrics_seg)


class ValidEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list,
                model_D, loss_D, metrics_D:list,
                geometric_transform, metrics_seg: list = None,
                device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='valid', device=device,
        )
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
        self.model_D.eval()

    def batch_update(self, batch):
        ### prepare inputs ###
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        y_A_D = (torch.ones(x_A.size()[0])).to(self.device)
        y_B_D = (torch.zeros(x_A.size()[0])).to(self.device)
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        with torch.no_grad():
            ### update discriminator ###
            # predict
            features_A = self.model.encoder.forward(x_A)
            features_B = self.model.encoder.forward(x_B)
        gp = compute_gradient_penalty(self.model_D, features_A, features_B, self.device)
        with torch.no_grad():
            pred_A_D = self.model_D.forward(features_A).squeeze()
            pred_B_D = self.model_D.forward(features_B).squeeze()
            wasserstein_distance = (pred_A_D.mean() - pred_B_D.mean())**2
            # logging
            self.update_loss(gp, wasserstein_distance, self.loss_D)
            self.update_metrics(gp, wasserstein_distance, self.metrics_D)

            ### update generator ###
            # predict
            pred_theta_A = self.model.forward(x_A, tgt_x_A)
            pred_theta_B = self.model.forward(x_B, tgt_x_B)
            # logging
            self.update_loss(theta_A, pred_theta_A, self.loss)
            self.update_loss(theta_B, pred_theta_B, self.loss)
            self.update_metrics(theta_A, pred_theta_A, self.metrics)
            self.update_metrics(theta_B, pred_theta_B, self.metrics)

            if self.metrics_seg is not None:
                src_label_A, src_label_B = y_A.float(), y_B.float()
                # Create pseudo-labels by randomly deforming the image
                tgt_label_A = self.geometric_transform(src_label_A, theta_A)
                tgt_label_B = self.geometric_transform(src_label_B, theta_B)
                # Trim the center of the original image
                pred_label_A = self.geometric_transform(src_label_A, pred_theta_A)
                pred_label_B = self.geometric_transform(src_label_B, pred_theta_B)
                # metrics
                self.update_metrics(tgt_label_A, pred_label_A, self.metrics_seg)
                self.update_metrics(tgt_label_B, pred_label_B, self.metrics_seg)
