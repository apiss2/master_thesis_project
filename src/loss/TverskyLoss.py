import torch
import torch.nn.functional as F

'''
This loss was introduced in
"Tversky loss function for image segmentationusing 3D fully convolutional deep networks",
retrievable here: https://arxiv.org/abs/1706.05721.

It was designed to optimise segmentation on imbalanced medical datasets by utilising constants
that can adjust how harshly different types of error are penalised in the loss function.

From the paper:
    ... in the case of α=β=0.5 the Tversky index simplifies to be the same as the Dice coefficient,
    which is also equal to the F1 score. With α=β=1, Equation 2 produces Tanimoto coefficient,
    and setting α+β=1 produces the set of Fβ scores.
    Larger βs weigh recall higher than precision (by placing more emphasis on false negatives).

To summarise, this loss function is weighted by the constants 'alpha' and 'beta'
that penalise false positives and false negatives respectively to a higher degree in the loss function as their value is increased.
The beta constant in particular has applications in situations where models can obtain misleadingly positive performance via highly conservative prediction.
You may want to experiment with different values to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
'''

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, weight=None, size_average=True, activation=False):
        super(TverskyLoss, self).__init__()
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        if self.activation:
            inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        return 1 - Tversky

# A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, smooth=1, weight=None, size_average=True, activation=False):
        super(FocalTverskyLoss, self).__init__()
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        if self.activation:
            inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        FocalTversky = (1 - Tversky)**self.gamma

        return FocalTversky