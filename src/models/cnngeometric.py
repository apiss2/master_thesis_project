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
    # 入力は240*240のみを想定
    def __init__(self, encoder='resnet',
                 in_channels=3, depth=5, weights=None,
                 output_dim=6, input_size=240,
                 fr_kernel_sizes=[3,3,3],
                 fr_channels=[128,64,32],
                 use_cuda=True, corr_type='3D',
                 freeze_FeatureExtractionModel=True):
        super(CNNGeometric, self).__init__()
        assert len(fr_channels)==len(fr_kernel_sizes), 'The list of channels must match the list of kernel sizes in length.'

        self.use_cuda = use_cuda
        self.FeatureExtraction = FeatureExtraction(encoder, in_channels=in_channels,
                                use_cuda=use_cuda, freeze=freeze_FeatureExtractionModel)
        self.FeatureCorrelation = FeatureCorrelation(shape=corr_type)

        test_batch = (torch.ones((4,3,input_size,input_size))/2).float()
        if use_cuda:
            test_batch = test_batch.cuda()
        with torch.no_grad():
            output = self.__test_forward(test_batch)
        fe_output_shape = output.size()[1:]

        self.FeatureRegression = FeatureRegression(output_dim=output_dim,
                                                   use_cuda=use_cuda,
                                                   input_shape=fe_output_shape,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels)

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, src_image, tgt_image):
        # feature extraction
        feature_A = self.FeatureExtraction(src_image)[-1]
        feature_B = self.FeatureExtraction(tgt_image)[-1]
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        # regression
        theta = self.FeatureRegression(correlation)

        return theta

    def __test_forward(self, image):
        # feature extraction
        feature_A = self.FeatureExtraction(image)[-1]
        feature_B = self.FeatureExtraction(image)[-1]
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        return correlation

