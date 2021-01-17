from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib

import segmentation_models_pytorch as smp

from .cnngeometric_base import FeatureExtraction, FeatureCorrelation, FeatureRegression


class CNNGeometric(nn.Module):
    """This is the code written with reference to https://github.com/ignacio-rocco/cnngeometric_pytorch"""
    def __init__(self, encoder='resnet50',
                 in_channels=3, depth=5, weights=None,
                 output_dim=6, input_size=256,
                 fr_kernel_sizes=[3,3,3],
                 fr_channels=[128,64,32],
                 corr_type='3D',
                 freeze_encoder=True):
        super(CNNGeometric, self).__init__()
        assert len(fr_channels)==len(fr_kernel_sizes), 'The list of channels must match the list of kernel sizes in length.'

        self.encoder = FeatureExtraction(encoder, in_channels=in_channels,
                                depth=depth, weights=weights, freeze=freeze_encoder)
        self.correlater = FeatureCorrelation(shape=corr_type)

        test_batch = (torch.ones((4,3,input_size,input_size))/2).float()
        with torch.no_grad():
            output = self.__test_forward(test_batch)
        fe_output_shape = output.size()[1:]

        self.regresser = FeatureRegression(output_dim=output_dim,
                                                   input_shape=fe_output_shape,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels)

    def forward(self, src_image, tgt_image):
        # feature extraction
        feature_A = self.encoder(src_image)
        feature_B = self.encoder(tgt_image)
        # feature correlation
        correlation = self.correlater(feature_A, feature_B)
        # regression
        theta = self.regresser(correlation)

        return theta

    def __test_forward(self, image):
        # feature extraction
        feature_A = self.encoder(image)
        feature_B = self.encoder(image)
        # feature correlation
        correlation = self.correlater(feature_A,feature_B)
        return correlation


class CNNGeometricDecoder(nn.Module):
    """This is the code written with reference to https://github.com/ignacio-rocco/cnngeometric_pytorch"""
    def __init__(self, sample, output_dim=6,
                 fr_kernel_sizes=[3,3,3],
                 fr_channels=[128,64,32],
                 corr_type='3D'):
        super(CNNGeometricDecoder, self).__init__()
        assert len(fr_channels)==len(fr_kernel_sizes), 'The list of channels must match the list of kernel sizes in length.'
        self.correlater = FeatureCorrelation(shape=corr_type)

        with torch.no_grad():
            output = self.__test_forward(sample)
        fe_output_shape = output.size()[1:]

        self.regresser = FeatureRegression(output_dim=output_dim,
                                                   input_shape=fe_output_shape,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels)

    def forward(self, feature_A, feature_B):
        # feature correlation
        correlation = self.correlater(feature_A, feature_B)
        # regression
        theta = self.regresser(correlation)
        return theta

    def __test_forward(self, sample):
        # feature correlation
        correlation = self.correlater(sample, sample)
        return correlation
