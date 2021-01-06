import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch import Tensor
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.base import Loss
from segmentation_models_pytorch.utils.functional import f_score

class CrossEntropyLoss(nn.CrossEntropyLoss):
    __name__ = 'CrossEntropyLoss'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.argmax(1).long()
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class CrossEntropyLoss2d(nn.Module):
    __name__ = 'CrossEntropyLoss2D'
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        return self.nll_loss(self.softmax(inputs), targets.argmax(dim=1))

class CE_Dice_Loss(nn.Module):
    __name__ = 'CE_Dice_Loss'
    def __init__(self, weight=None, size_average=True):
        super(CE_Dice_Loss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)
        self.softmax = nn.LogSoftmax(dim=1)
        self.diceloss = smp.utils.losses.DiceLoss(ignore_channels=[0])

    def forward(self, inputs, targets):
        CEloss = self.nll_loss(self.softmax(inputs), targets.argmax(dim=1))
        diceloss = self.diceloss(self.softmax(inputs), targets)
        return diceloss*0.1 + CEloss

class CrossEntropyWithDiceLoss(nn.CrossEntropyLoss, Loss):
    __name__ = 'CrossEntropyWithDiceLoss'

    def __init__(self, weight: Optional[torch.Tensor] = None,\
                    size_average=None, ignore_index: int = -100,\
                    reduce=None, reduction: str = 'mean',\
                    eps=1., beta=1., ignore_channels=None):
        super(nn.CrossEntropyLoss, self).__init__(weight=weight, \
            size_average=size_average, \
            reduce=reduce, reduction=reduction)
        self.ignore_index=ignore_index
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        dice = 1 - f_score(input, target,
            beta=self.beta, eps=self.eps, threshold=None,
            ignore_channels=self.ignore_channels,)
        target = target.argmax(1).long()
        crossentropy = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        return 0.5 * crossentropy + dice