import numpy as np

import torch
from torch import autograd

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
    def __init__(self, model, loss, metrics:list, optimizer,
                    model_D, loss_D, metrics_D:list, optimizer_D,
                    modelupdate_freq: int = 10, discupdate_freq: int = 10,
                    device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.modelupdate_freq = modelupdate_freq
        self.discupdate_freq = discupdate_freq

    def on_epoch_start(self):
        self.iter = 0
        self.model.train()
        self.model_D.train()

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

        ### update discriminator ###
        if self.iter % self.discupdate_freq == 0:
            self.model.eval()
            self.set_requires_grad(self.model, requires_grad=False)
            self.model_D.train()
            self.set_requires_grad(self.model_D, requires_grad=True)
            self.optimizer_D.zero_grad()
            # predict
            with torch.no_grad():
                features_A = self.model.encoder.forward(x_A)[-1]
                features_B = self.model.encoder.forward(x_B)[-1]
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
            self.update_metrics(wasserstein_distance, gp, self.metrics_D)

        ### update generator ###
        if self.iter % self.modelupdate_freq == 0:
            self.set_requires_grad(self.model, requires_grad=True)
            self.model.train()
            self.set_requires_grad(self.model_D, requires_grad=False)
            self.model_D.eval()
            self.optimizer.zero_grad()
            # predict
            pred_A = self.model.forward(x_A)
            loss_A, _ = self.update_loss(y_A, pred_A, self.loss)
            pred_B = self.model.forward(x_B)
            loss_B, _ = self.update_loss(y_B, pred_B, self.loss)
            loss = loss_A + loss_B
            # backward
            loss.backward()
            self.optimizer.step()
            self.update_metrics(y_A, pred_A, self.metrics)
            self.update_metrics(y_B, pred_B, self.metrics)


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

        with torch.no_grad():
            ### update discriminator ###
            # predict
            features_A = self.model.encoder.forward(x_A)[-1]
            features_B = self.model.encoder.forward(x_B)[-1]
        gp = compute_gradient_penalty(self.model_D, features_A, features_B, self.device)
        with torch.no_grad():
            pred_A_D = self.model_D.forward(features_A).squeeze()
            pred_B_D = self.model_D.forward(features_B).squeeze()
            wasserstein_distance = pred_A_D.mean() - pred_B_D.mean()
            # logging
            self.update_loss(gp, wasserstein_distance, self.loss_D)
            self.update_metrics(wasserstein_distance, gp, self.metrics_D)

            ### update generator ###
            # predict
            pred_A = self.model.forward(x_A)
            pred_B = self.model.forward(x_B)
            # logging
            self.update_loss(y_A, pred_A, self.loss)
            self.update_loss(y_B, pred_B, self.loss)
            self.update_metrics(y_A, pred_A, self.metrics)
            self.update_metrics(y_B, pred_B, self.metrics)

