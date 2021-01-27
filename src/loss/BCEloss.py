import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(torch.nn.BCELoss):
    __name__ = 'BCEloss'

class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    __name__ = 'BCEWithLogitsLoss'

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, activation=False):
        super(DiceBCELoss, self).__init__()
        self.activation = activation

    def forward(self, inputs, targets, smooth=1):
        if self.activation:
            inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE