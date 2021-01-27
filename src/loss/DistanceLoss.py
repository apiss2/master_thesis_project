import torch
import torch.nn as nn
import torch.nn.functional as F


class MAE(nn.L1Loss):
    __name__ = 'MAE'


class MSE(nn.MSELoss):
    __name__ = 'MSE'


class WassersteinGPLoss(nn.Module):
    __name__ = 'WassersteinGPLoss'
    def __init__(self):
        super(WassersteinGPLoss, self).__init__()

    def forward(self, wasserstein_distance, gp, gamma=10):
        return -wasserstein_distance + gamma*gp


class WassersteinDistance(nn.Module):
    __name__ = 'WassersteinDistance'
    def __init__(self):
        super(WassersteinDistance, self).__init__()

    def forward(self, wasserstein_distance, gp, gamma=10):
        return wasserstein_distance


class GradientPenalty2D(nn.Module):
    __name__ = 'GradientPenalty2D'
    def __init__(self):
        super(GradientPenalty2D, self).__init__()

    def forward(self, wasserstein_distance, gp, gamma=10):
        return gp*gamma


class Diff2d(nn.Module):
    __name__ = 'Diff2D'
    def __init__(self, sum=False):
        super(Diff2d, self).__init__()
        self.sum = sum

    def forward(self, inputs1, inputs2):
        if self.sum:
            return torch.sum(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))


class Symkl2d(nn.Module):
    __name__ = 'Symkl2D'
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
