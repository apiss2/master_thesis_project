import torch.nn as nn

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