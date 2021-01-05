import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from ..transformation.core import PointTnf

class TransformedGridLoss(nn.Module):
    __name__ = 'GridLoss'

    def __init__(self, geometric='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric = geometric

        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda()

    def _expand_grid(self, theta):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size, 2, self.N)
        return P

    def _transform_grid_points(self, P, theta, theta_GT):
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric=='affine':
            P_prime = self.pointTnf.affPointTnf(theta,P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)
        elif self.geometric=='hom':
            P_prime = self.pointTnf.homPointTnf(theta,P)
            P_prime_GT = self.pointTnf.homPointTnf(theta_GT,P)
        elif self.geometric=='tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)
        return P_prime, P_prime_GT

    def _mse(self, P_prime, P_prime_GT):
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)
        return loss

    def forward(self, theta, theta_GT):
        P = self._expand_grid(theta)
        P_prime, P_prime_GT = self._transform_grid_points(P, theta, theta_GT)
        return self._mse(P_prime, P_prime_GT)

class GridMetric(TransformedGridLoss):
    __name__ = 'GridMetric'

    def __init__(self, geometric='affine', use_cuda=True, grid_size=20):
        super().__init__(\
            geometric=geometric, use_cuda=use_cuda, grid_size=grid_size)

    def forward(self, theta, theta_GT):
        P = self._expand_grid(theta)
        P_prime, P_prime_GT = self._transform_grid_points(P, theta, theta_GT)
        loss = self._mse(P_prime, P_prime_GT)
        metric = 1-loss
        return (metric>0)*metric
