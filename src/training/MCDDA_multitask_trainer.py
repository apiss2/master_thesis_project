from .train_util import MultiTaskEpoch
from src.transformation.affine import AffineGridGen
from src.transformation.tps import TpsGridGen


class TrainEpoch(MultiTaskEpoch):
    def __init__(self, model, loss, metrics: list, optimizer,
                 decoder_seg, loss_seg, metrics_seg: list, optimizer_seg,
                 decoder, loss_D, metrics_D, optimizer_D, geometric,
                 modelupdate_freq: int = 1, discupdate_freq: int = 1,
                 geometric_transform=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_seg=decoder_seg, loss_seg=loss_seg,
            metrics_seg=metrics_seg,
            model_D=decoder, loss_D=loss_D, metrics_D=metrics_D,
            device=device, stage_name='train'
        )
        self.optimizer = optimizer
        self.optimizer_seg = optimizer_seg
        self.optimizer_D = optimizer_D
        self.modelupdate_freq = modelupdate_freq
        self.discupdate_freq = discupdate_freq

        if geometric == 'affine':
            self.geometric_gridgen = AffineGridGen()
        elif geometric == 'tps':
            self.geometric_gridgen = TpsGridGen(device=device)
        else:
            assert False, f'No such geometric: {geometric}'

        self.geometric_transform = geometric_transform

    def on_epoch_start(self):
        self.iter = 0
        self.model.train()
        self.set_requires_grad(self.model, requires_grad=True)
        self.model_D.train()
        self.set_requires_grad(self.model_D, requires_grad=True)
        self.decoder_seg.train()
        self.set_requires_grad(self.decoder_seg, requires_grad=True)

    def batch_update(self, batch):
        self.iter += 1
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        # Segmentation
        for x, y in zip([x_A, x_B], [y_A, y_B]):
            self.optimizer.zero_grad()
            self.optimizer_seg.zero_grad()
            # predict
            features = self.model.encoder.forward(x)
            pred = self.decoder_seg.forward(features)
            # loss
            loss, _ = self.update_loss(y, pred, self.loss_seg)
            self.update_metrics(y, pred, self.metrics_seg)
            # backward
            loss.backward()
            self.optimizer.step()
            self.optimizer_seg.step()

        # update generator and classifiers #
        # Registration #
        # src
        self.optimizer.zero_grad()
        # predict
        pred_theta = self.model.forward(x_A, tgt_x_A)
        # loss
        loss, _ = self.update_loss(theta_A, pred_theta, self.loss)
        self.update_metrics(theta_A, pred_theta, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()

        # tgt
        self.optimizer.zero_grad()
        self.optimizer_D.zero_grad()
        # predict
        features_src = self.model.encoder.forward(x_B)[-1]
        features_tgt = self.model.encoder.forward(tgt_x_B)[-1]
        pred = self.model_D.forward(features_src, features_tgt)
        # loss
        loss, _ = self.update_loss(theta_B, pred, self.loss)
        self.update_metrics(theta_B, pred, self.metrics)
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_D.step()

        # update classifiers
        self.model.encoder.eval()
        self.set_requires_grad(self.model.encoder, requires_grad=False)
        self.optimizer.zero_grad()
        self.optimizer_D.zero_grad()
        # src
        pred_theta_1, features_src, features_tgt = self.model.forward(
            x_A, tgt_x_A, output_features=True)
        pred_theta_2 = self.model_D.forward(features_src, features_tgt)
        loss_tmp_1, _ = self.update_loss(theta_A, pred_theta_1, self.loss)
        loss_tmp_2, _ = self.update_loss(theta_A, pred_theta_2, self.loss)
        # tgt
        pred_theta_1, features_src, features_tgt = self.model.forward(
            x_B, tgt_x_B, output_features=True)
        pred_theta_2 = self.model_D.forward(features_src, features_tgt)
        pred_1 = self.geometric_gridgen(pred_theta_1)
        pred_2 = self.geometric_gridgen(pred_theta_2)
        loss_tmp, _ = self.update_loss(pred_1, pred_2, self.loss_D)
        self.update_metrics(pred_1, pred_2, self.metrics_D)
        loss = (loss_tmp_1 + loss_tmp_2) - loss_tmp
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer_D.step()

        # update generator #
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.model.encoder.train()
        self.set_requires_grad(self.model.encoder, requires_grad=True)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)

        self.optimizer.zero_grad()
        # predict
        for x, tgt_x in zip([x_A, x_B], [tgt_x_A, tgt_x_B]):
            pred_1, features_src, features_tgt = self.model.forward(x, tgt_x)
            pred_2 = self.model_D.forward(features_src, features_tgt)
            pred_1 = self.geometric_gridgen(pred_1)
            pred_2 = self.geometric_gridgen(pred_2)
            loss, _ = self.update_loss(pred_1, pred_2, self.loss_D)
            self.update_metrics(pred_1, pred_2, self.metrics_D)
            # backward
            loss.backward()
            self.optimizer.step()


class ValidEpoch(MultiTaskEpoch):
    def __init__(self, model, loss, metrics: list,
                 decoder_seg, loss_seg, metrics_seg: list,
                 decoder, loss_D, metrics_D, geometric,
                 geometric_transform=None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics,
            decoder_seg=decoder_seg, loss_seg=loss_seg,
            metrics_seg=metrics_seg,
            model_D=decoder, loss_D=loss_D, metrics_D=metrics_D,
            device=device, stage_name='train'
        )
        if geometric == 'affine':
            self.geometric_gridgen = AffineGridGen()
        elif geometric == 'tps':
            self.geometric_gridgen = TpsGridGen(device=device)
        else:
            assert False, f'No such geometric: {geometric}'

        self.geometric_transform = geometric_transform

    def on_epoch_start(self):
        self.model.eval()
        self.set_requires_grad(self.model, requires_grad=False)
        self.model_D.eval()
        self.set_requires_grad(self.model_D, requires_grad=False)
        self.decoder_seg.eval()
        self.set_requires_grad(self.decoder_seg, requires_grad=False)

    def batch_update(self, batch):
        x_A, y_A, theta_A = batch['x_A'], batch['y_A'], batch['theta_A']
        x_B, y_B, theta_B = batch['x_B'], batch['y_B'], batch['theta_B']
        tgt_x_A = self.geometric_transform(x_A, theta_A)
        tgt_x_B = self.geometric_transform(x_B, theta_B)

        # Segmentation
        for x, y in zip([x_A, x_B], [y_A, y_B]):
            # predict
            features = self.model.encoder.forward(x)
            pred = self.decoder_seg.forward(features)
            # loss
            self.update_loss(y, pred, self.loss_seg)
            self.update_metrics(y, pred, self.metrics_seg)

        # update generator and classifiers #
        for x, tgt_x, theta in zip([x_A, x_B], [tgt_x_A, tgt_x_B],
                                   [theta_A, theta_B]):
            # predict
            pred_theta_1, features_src, features_tgt = \
                self.model.forward(x, tgt_x, output_features=True)
            pred_theta_2 = self.model_D.forward(features_src, features_tgt)
            # loss
            self.update_loss(theta, pred_theta_1, self.loss)
            self.update_loss(theta, pred_theta_2, self.loss)
            self.update_metrics(theta, pred_theta_1, self.metrics)
            self.update_metrics(theta, pred_theta_2, self.metrics)

            pred_1 = self.geometric_gridgen(pred_theta_1)
            pred_2 = self.geometric_gridgen(pred_theta_2)
            self.update_loss(pred_1, pred_2, self.loss_D)
            self.update_metrics(pred_1, pred_2, self.metrics_D)
