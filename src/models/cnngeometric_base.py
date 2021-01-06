from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder

class FeatureExtraction(torch.nn.Module):
    def __init__(self, name, in_channels=3, depth=5,
                weights=None, freeze=True):
        super(FeatureExtraction, self).__init__()
        self.model = get_encoder(name, in_channels, depth, weights)

        # freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        return self.model(image_batch)

class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D'):
        super(FeatureCorrelation, self).__init__()
        self.shape=shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        feature_A = featureL2Norm(feature_A)
        feature_B = featureL2Norm(feature_B)
        b,c,h,w = feature_A.size()
        if self.shape=='3D':
            # Reshape features to compute the matrix product
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # Compute the matrix product
            feature_mul = torch.bmm(feature_B,feature_A)
            # Each dimension is [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        elif self.shape=='4D':
            # Reshape features to compute the matrix product
            feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
            # Compute the matrix product
            feature_mul = torch.bmm(feature_A,feature_B)
            # Each dimension is [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)

        correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
        return correlation_tensor

class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, input_shape=(225,15,15),
                    kernel_sizes=[3,3,3], channels=[128,64,32]):
        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        channels = [input_shape[0]] + channels

        nn_modules = list()
        for i in range(num_layers):
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i+1]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

        # Compute the input value of linear (fc layer)
        input_size = input_shape[-1]
        for i in kernel_sizes:
            input_size -= (i-1)
        linear_ch = channels[-1] * input_size**2
        self.linear = nn.Linear(linear_ch, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)
