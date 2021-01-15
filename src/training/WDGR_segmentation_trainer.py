import numpy as np

import torch
from torch import autograd

from .train_util import GANEpoch, Tester, UnNormalize


def gradient_penalty_2D(model_D, real_data, fake_data, device):
    """code from https://github.com/lynnezixuan/CS231N/blob/master/3D-Generation-WGAN-GP-PYTORCH.py"""
    alpha = np.random.rand(real_data.size()[0], 1, 1, 1)
    alpha = torch.from_numpy(alpha*np.ones(real_data.size())).float()
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = model_D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


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
        y_A_D = (torch.ones(x_A.size()[0])).to(self.device)
        y_B_D = (torch.zeros(x_A.size()[0])).to(self.device)

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
            gp = gradient_penalty_2D(self.model_D, features_A, features_B, self.device)
            pred_A_D = self.model_D.forward(features_A).squeeze()
            pred_B_D = self.model_D.forward(features_B).squeeze()
            # loss
            wasserstein_distance = pred_A_D.mean() - pred_B_D.mean()
            loss, _ = self.update_loss(gp, wasserstein_distance, self.loss_D)
            # backward
            loss.backward()
            self.optimizer_D.step()
            self.update_metrics(y_A_D, pred_A_D, self.metrics_D)
            self.update_metrics(y_B_D, pred_B_D, self.metrics_D)

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
        y_A_D = (torch.ones(x_A.size()[0])).to(self.device)
        y_B_D = (torch.zeros(x_A.size()[0])).to(self.device)

        with torch.no_grad():
            ### update discriminator ###
            # predict
            features_A = self.model.encoder.forward(x_A)[-1]
            features_B = self.model.encoder.forward(x_B)[-1]
        gp = gradient_penalty_2D(self.model_D, features_A, features_B, self.device)
        with torch.no_grad():
            pred_A_D = self.model_D.forward(features_A).squeeze()
            pred_B_D = self.model_D.forward(features_B).squeeze()
            wasserstein_distance = pred_A_D.mean() - pred_B_D.mean()
            # logging
            self.update_loss(gp, wasserstein_distance, self.loss_D)
            self.update_metrics(y_A_D, pred_A_D, self.metrics_D)
            self.update_metrics(y_B_D, pred_B_D, self.metrics_D)

            ### update generator ###
            # predict
            pred_A = self.model.forward(x_A)
            pred_B = self.model.forward(x_B)
            # logging
            self.update_loss(y_A, pred_A, self.loss)
            self.update_loss(y_B, pred_B, self.loss)
            self.update_metrics(y_A, pred_A, self.metrics)
            self.update_metrics(y_B, pred_B, self.metrics)


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
