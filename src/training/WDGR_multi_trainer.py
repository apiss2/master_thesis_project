import numpy as np

import torch
from torch import autograd

from ..utils.meter import AverageValueMeter
from .train_util import MultiTaskEpoch


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
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        ### update discriminator ###
        self.model.encoder.train()
        self.set_requires_grad(self.model.encoder, requires_grad=True)

        self.optimizer_D.zero_grad()

        # predict
        with torch.no_grad():
            features_A = self.model.encoder.forward(x_A)[-1]
            features_B = self.model.encoder.forward(x_B)[-1]
        gp = compute_gradient_penalty(self.model_D, features_A, features_B, self.device)
        pred_A_D = self.model_D.forward(features_A).squeeze()
        pred_B_D = self.model_D.forward(features_B).squeeze()
        wasserstein_distance = pred_A_D.mean() - pred_B_D.mean()
        # loss
        loss, _ = self.update_loss(gp, wasserstein_distance, self.loss_D)
        # backward
        loss.backward()
        self.optimizer_D.step()
        self.update_metrics(gp, wasserstein_distance, self.metrics_D)

        ### segmentation ###
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
        self.optimizer.zero_grad()
        # predict
        pred_theta_A, features_A = self.model.forward(x_A, tgt_x_A, output_src_features=True)
        loss_A, _ = self.update_loss(theta_A, pred_theta_A, self.loss)
        pred_theta_B, features_B = self.model.forward(x_B, tgt_x_B, output_src_features=True)
        loss_B, _ = self.update_loss(theta_B, pred_theta_B, self.loss)
        pred_A_D = self.model_D.forward(features_A).squeeze()
        pred_B_D = self.model_D.forward(features_B).squeeze()
        wasserstein_distance = pred_A_D.mean() - pred_B_D.mean()
        loss = loss_A + loss_B + wasserstein_distance
        # backward
        loss.backward()
        self.optimizer.step()
        self.update_metrics(theta_A, pred_theta_A, self.metrics)
        self.update_metrics(theta_B, pred_theta_B, self.metrics)



class ValidEpoch(MultiTaskEpoch):
    def __init__(self, model, loss, metrics: list, \
                    decoder_seg, loss_seg, metrics_seg: list, \
                    model_D, loss_D, metrics_D: list,
                    modelupdate_freq: int = 1, discupdate_freq: int = 1,
                    geometric_transform=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_seg=decoder_seg, loss_seg=loss_seg, metrics_seg=metrics_seg,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            device=device, stage_name='valid'
        )
        self.geometric_transform = geometric_transform

    def on_epoch_start(self):
        self.model.eval()
        self.model_D.eval()

    def batch_update(self, batch):
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        ### update discriminator ###
        # predict
        with torch.no_grad():
            features_A = self.model.encoder.forward(x_A)[-1]
            features_B = self.model.encoder.forward(x_B)[-1]
        gp = compute_gradient_penalty(self.model_D, features_A, features_B, self.device)
        with torch.no_grad():
            pred_A_D = self.model_D.forward(features_A).squeeze()
            pred_B_D = self.model_D.forward(features_B).squeeze()
            wasserstein_distance = pred_A_D.mean() - pred_B_D.mean()
            # loss
            self.update_loss(gp, wasserstein_distance, self.loss_D)
            self.update_metrics(gp, wasserstein_distance, self.metrics_D)

        ### segmentation ###
        with torch.no_grad():
            # predict
            features_A = self.model.encoder.forward(x_A)
            features_B = self.model.encoder.forward(x_B)
            pred_A = self.decoder_seg.forward(features_A)
            pred_B = self.decoder_seg.forward(features_B)
            self.update_loss(y_A, pred_A, self.loss_seg)
            self.update_loss(y_B, pred_B, self.loss_seg)
            # loss
            self.update_metrics(y_A, pred_A, self.metrics_seg)
            self.update_metrics(y_B, pred_B, self.metrics_seg)

        ### registration ###
        with torch.no_grad():
            # predict
            pred_theta_A = self.model.forward(x_A, tgt_x_A)
            self.update_loss(theta_A, pred_theta_A, self.loss)
            pred_theta_B = self.model.forward(x_B, tgt_x_B)
            self.update_loss(theta_B, pred_theta_B, self.loss)
            # loss
            self.update_metrics(theta_A, pred_theta_A, self.metrics)
            self.update_metrics(theta_B, pred_theta_B, self.metrics)