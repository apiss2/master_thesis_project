import torch.nn as nn

class MAE(nn.L1Loss):
    __name__ = 'MAE'

class MSE(nn.MSELoss):
    __name__ = 'MSE'