import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Function

from .layers import ConvBNReLU, InvertedResidual, _make_divisible, GlobalAveragePooling2D

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lamba_ = 1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lamba_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Discriminator(nn.Module):
    def __init__(self, input_sample, hidden_channels):
        super(Discriminator, self).__init__()
        assert type(hidden_channels)==list, 'Type of hidden_channels must be list.'

        size = input_sample.size()
        ch_in, w, h = size[1], size[2], size[3]
        assert w==h, 'The width and height must match.'

        nn_modules = [GradientReversal()]
        for i, ch_out in enumerate(hidden_channels):
            s = 2 if i%3==1 else 1
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=s, padding=0))
            nn_modules.append(nn.LeakyReLU(0.2, inplace=True))
            nn_modules.append(nn.BatchNorm2d(ch_out))
            ch_in = ch_out
        #nn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv = nn.Sequential(*nn_modules)
        #self.GAP = GlobalAveragePooling2D()

        with torch.no_grad():
            sample = self.conv(input_sample)
        size = sample.size()
        ch, w, h = size[1], size[2], size[3]
        self.fc = nn.Sequential(
            #nn.Linear(int(ch*w*h), 1),
            nn.Linear(hidden_channels[-1], 1),
            #nn.BatchNorm1d(1),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(16, 1),
            #nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        #x = torch.flatten(x, start_dim=1)
        #x = x.view(x.shape[0], -1)
        #x = self.GAP(x)
        x = torch.mean(x, dim=[2,3])
        x = self.fc(x)
        return x


class MobilenetDiscriminator(nn.Module):
    def __init__(self,
                 input_sample, hidden_channels=[64, 64],
                 last_channel=16,
                 num_classes=1,
                 width_mult=1.0,
                 round_nearest=8,
                 norm_layer=None):
        super(MobilenetDiscriminator, self).__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        size = input_sample.size()
        input_channel = size[1]
        last_channel = last_channel

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        for i, c in enumerate(hidden_channels):
            inverted_residual_setting[i][1] = c
        inverted_residual_setting = inverted_residual_setting[:len(hidden_channels)]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # building inverted residual blocks
        features = list()
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

