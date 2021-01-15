import torch
import torch.nn as nn
import torch.nn.functional as F


class MAE(nn.L1Loss):
    __name__ = 'MAE'


class MSE(nn.MSELoss):
    __name__ = 'MSE'


class WassersteinLoss(nn.Module):
    __name__ = 'WassersteinLoss'
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, wasserstein_distance, gp, gamma=10):
        return -wasserstein_distance + gamma*gp


class WassersteinMetric(nn.Module):
    __name__ = 'WassersteinDistance'
    def __init__(self):
        super(WassersteinMetric, self).__init__()

    def forward(self, wasserstein_distance, gp, gamma=10):
        loss = -wasserstein_distance + gamma*gp
        metric = 1-loss
        return (metric>0)*metric


class Diff2d(nn.Module):
    def __init__(self):
        super(Diff2d, self).__init__()

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))


class Symkl2d(nn.Module):
    def __init__(self, size_average=True):
        super(Symkl2d, self).__init__()
        self.size_average = size_average

    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1)
        self.prob2 = F.softmax(inputs2)
        self.log_prob1 = F.log_softmax(self.prob1)
        self.log_prob2 = F.log_softmax(self.prob2)

        loss = 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))
        return loss
