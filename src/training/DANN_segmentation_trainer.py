import torch
import numpy as np

from .train_util import GANEpoch, Tester

class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list, optimizer,
                    model_D, loss_D, metrics_D:list, optimizer_D,
                    modelupdate_freq:int=10, device:str='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
            stage_name='train', device=device,
        )
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.modelupdate_freq = modelupdate_freq

    def on_epoch_start(self):
        self.iter = 0
        self.model.train()
        self.model_D.train()

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A = batch['x_A'], batch['y_A']
        x_B, y_B = batch['x_B'], batch['y_B']

        for i, (x, y) in enumerate(zip([x_A, x_B], [y_A, y_B])):
            y_D = (torch.ones(x.size()[0])*(1-i)).to(self.device)#.unsqueeze(-1)
            ### update discriminator ###
            self.model.eval()
            if self.iter%self.modelupdate_freq==0:
                self.model.encoder.train()
                self.optimizer.zero_grad()
            self.model_D.train()
            self.optimizer_D.zero_grad()
            # predict
            features = self.model.encoder.forward(x)[-1]
            pred_D = self.model_D.forward(features).squeeze()
            loss, _ = self.update_loss(y_D, pred_D, self.loss_D)

            # backward
            loss.backward()
            if self.iter%self.modelupdate_freq==0:
                self.optimizer.step()
            self.optimizer_D.step()
            self.update_metrics(y_D, pred_D, self.metrics_D)

            ### update generator ###
            self.model.train()
            self.model_D.eval()
            self.optimizer.zero_grad()
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


class TestEpoch(Tester):
    def __init__(self, model, loss, metrics:list, device:str='cpu',
                    target_modality:str = 'A',
                    label_type:str='binary_label', color_palette:list=None,
                    save_path:str=None, mean:list=None, std:list=None):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device,
            label_type=label_type, color_palette=color_palette,
            save_path=save_path, mean=mean, std=std
        )
        assert target_modality == 'A' or target_modality == 'B', 'target_modality must be A or B'
        self.target_modality = target_modality

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
                self.imwrite(x[0], name, is_image=True)

                # label
                name = 'label_{}_{:03}.png'.format(modality, self.iter_num)
                self.imwrite(y[0], name)
