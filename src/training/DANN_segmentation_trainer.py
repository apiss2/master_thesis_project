import torch

from .train_util import GANEpoch

class TrainEpoch(GANEpoch):
    def __init__(self, model, loss, metrics:list, optimizer,
                    model_D, loss_D, metrics_D:list, optimizer_D,
                    modelupdate_freq:int=1, discupdate_freq:int=1,
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

        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)
        if self.iter % self.modelupdate_freq == 0:
            self.set_requires_grad(self.model.encoder, requires_grad=True)
            self.model.encoder.train()
        if self.iter % self.discupdate_freq == 0:
            self.set_requires_grad(self.model_D, requires_grad=True)
            self.model_D.train()
        for i, x in enumerate([x_A, x_B]):
            if self.iter % self.modelupdate_freq == 0:
                self.optimizer.zero_grad()
            if self.iter % self.discupdate_freq == 0:
                self.optimizer_D.zero_grad()
            y_D = (torch.ones(x.size()[0])*(1-i)).to(self.device)
            half = (torch.ones(x.size()[0])/2).to(self.device)
            ### update discriminator ###
            # predict
            features = self.model.encoder.forward(x)[-1]
            pred_D = self.model_D.forward(features).squeeze()
            loss, _ = self.update_loss(y_D, pred_D, self.loss_D)
            loss += torch.mean(half - torch.abs(pred_D-half))
            # backward
            loss.backward()
            if self.iter%self.modelupdate_freq==0:
                self.optimizer.step()
            if self.iter%self.discupdate_freq==0:
                self.optimizer_D.step()
            self.update_metrics(y_D, pred_D, self.metrics_D)

        self.set_requires_grad(self.model, requires_grad=True)
        self.model.train()
        self.set_requires_grad(self.model_D, requires_grad=False)
        self.model_D.eval()
        self.optimizer.zero_grad()
        for x, y in zip([x_A, x_B], [y_A, y_B]):
            ### update generator ###
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

