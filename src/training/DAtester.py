import torch

from .test_util import Tester, UnNormalize
from ..utils.meter import AverageValueMeter


class DAsegmentationTester(Tester):
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
        if self.target_modality == 'A':
            x, y = batch['x_A'], batch['y_A']
        elif self.target_modality == 'B':
            x, y = batch['x_B'], batch['y_B']

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
            name = 'predict_{}_{:03}.png'.format(self.target_modality, self.iter_num)
            self.imwrite(pred[0], name)

            # image
            name = 'image_{}_{:03}.png'.format(self.target_modality, self.iter_num)
            x = self.unorm_A(x[0]) if self.target_modality == 'A' else self.unorm_B(x[0])
            self.imwrite(x, name, is_image=True)

            # label
            name = 'label_{}_{:03}.png'.format(self.target_modality, self.iter_num)
            self.imwrite(y[0], name)


class DAregistrationTester(Tester):
    def __init__(self, model, loss, metrics: list, geometric_transform,
                 target_modality: str = 'A', multi_modality: bool = False,
                 label_type: str = 'binary_label', color_palette: list = None,
                 save_path: str = None, mean_A: list = None, std_A: list = None,
                 mean_B: list = None, std_B: list = None,
                 metrics_seg: list = None, device='cpu'):
        super().__init__(
            model=model, loss=loss, metrics=metrics, device=device, save_path=save_path,
            label_type=label_type, color_palette=color_palette
        )
        assert target_modality == 'A' or target_modality == 'B', 'target_modality must be A or B'
        self.multi_modality = multi_modality
        self.mono_or_multi = 'multi' if self.multi_modality else 'mono'
        self.target_modality = target_modality

        self.unorm_A = None if None in [mean_A, std_A] else UnNormalize(mean=mean_A, std=std_A)
        self.unorm_B = None if None in [mean_B, std_B] else UnNormalize(mean=mean_B, std=std_B)

        self.geometric_transform = geometric_transform
        self.metrics_seg = [metric.to(self.device) for metric in metrics_seg] if metrics_seg is not None else None

    def reset_meters(self):
        self.loss_meters = {self.loss.__name__: AverageValueMeter()}
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        if self.metrics_seg is not None:
            self.metrics_meters.update({metric.__name__: AverageValueMeter() for metric in self.metrics_seg})

    def batch_update(self, batch):
        self.iter_num += 1
        self.all_logs[self.iter_num] = dict()

        theta = batch['theta_A']
        if self.multi_modality and self.target_modality == 'A':
            x_A, x_B, src_label = batch['x_A'], batch['x_B'], batch['y_A']
            tgt_x_A = self.geometric_transform(x_A, theta)
        elif self.multi_modality and self.target_modality == 'B':
            x_A, x_B, src_label = batch['x_A'], batch['x_B'], batch['y_A']
            tgt_x_B = self.geometric_transform(x_B, theta)
        elif not self.multi_modality and self.target_modality=='A':
            x_A, src_label = batch['x_A'], batch['y_A']
            tgt_x_A = self.geometric_transform(x_A, theta)
        elif not self.multi_modality and self.target_modality=='B':
            x_B, src_label = batch['x_B'], batch['y_A']
            tgt_x_B = self.geometric_transform(x_B, theta)
        else:
            assert 'Unexpected modality settings: target={}, multi={}'.format(\
                self.target_modality, self.multi_modality)

        with torch.no_grad():
            if self.multi_modality and self.target_modality == 'A':
                image, tgt_image = x_B, tgt_x_A
                src_modality, tgt_modality = 'B', 'A'
            elif self.multi_modality and self.target_modality == 'B':
                image, tgt_image = x_A, tgt_x_B
                src_modality, tgt_modality = 'A', 'B'
            elif not self.multi_modality and self.target_modality=='A':
                image, tgt_image = x_A, tgt_x_A
                src_modality, tgt_modality = 'A', 'A'
            elif not self.multi_modality and self.target_modality=='B':
                image, tgt_image = x_B, tgt_x_B
                src_modality, tgt_modality = 'B', 'B'
            else:
                assert 'Unexpected modality settings: target={}, multi={}'.format(\
                    self.target_modality, self.multi_modality)
            # predict
            pred_theta = self.model.forward(image, tgt_image)
            # logging
            _, loss_value = self.update_loss(theta, pred_theta, self.loss)
            metrics_values = self.update_metrics(theta, pred_theta, self.metrics)
            self.all_logs[self.iter_num][self.loss.__name__] = loss_value
            self.all_logs[self.iter_num].update(metrics_values)

        predict_image = self.geometric_transform(image, pred_theta)

        if self.metrics_seg is not None:
            # Create pseudo-labels by randomly deforming the image
            tgt_label = self.geometric_transform(src_label.float(), theta)
            # Trim the center of the original image
            pred_label = self.geometric_transform(src_label.float(), pred_theta)
            # metrics
            metrics_values = self.update_metrics(tgt_label, pred_label, self.metrics_seg)
            self.all_logs[self.iter_num].update(metrics_values)

        if self.save_image:
            # src_image
            name = '{}_{}_src_image_{:03}.png'.format(\
                self.mono_or_multi, self.target_modality, self.iter_num)
            x = self.unorm_A(image[0]) if src_modality == 'A' else self.unorm_B(image[0])
            self.imwrite(x, name, is_image=True)

            # tgt_image
            name = '{}_{}_tgt_image_{:03}.png'.format(\
                self.mono_or_multi, self.target_modality, self.iter_num)
            x = self.unorm_A(tgt_image[0]) if tgt_modality == 'A' else self.unorm_B(tgt_image[0])
            self.imwrite(x, name, is_image=True)

            # predict_image
            name = '{}_{}_predict_image_{:03}.png'.format(\
                self.mono_or_multi, self.target_modality, self.iter_num)
            x = self.unorm_A(predict_image[0]) if src_modality == 'A' else self.unorm_B(predict_image[0])
            self.imwrite(x, name, is_image=True)

            if self.metrics_seg is not None:
                # src_label
                name = '{}_{}_src_label_{:03}.png'.format(\
                    self.mono_or_multi, self.target_modality, self.iter_num)
                x = src_label[0] if src_modality == 'A' else src_label[0]
                self.imwrite(x, name)

                # tgt_label
                name = '{}_{}_tgt_label_{:03}.png'.format(\
                    self.mono_or_multi, self.target_modality, self.iter_num)
                x = tgt_label[0] if tgt_modality == 'A' else tgt_label[0]
                self.imwrite(x, name)

                # predict_label
                name = '{}_{}_predict_label_{:03}.png'.format(\
                    self.mono_or_multi, self.target_modality, self.iter_num)
                x = pred_label[0] if src_modality == 'A' else pred_label[0]
                self.imwrite(x, name)
