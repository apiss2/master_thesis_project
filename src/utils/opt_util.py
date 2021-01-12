import torch
import torch.optim.lr_scheduler as lrs

import segmentation_models_pytorch as smp
from warmup_scheduler import GradualWarmupScheduler

from ..loss.BCEloss import BCELoss, BCEWithLogitsLoss
from ..loss.CrossEntropy import (CE_Dice_Loss, CrossEntropyLoss,
                                 CrossEntropyLoss2d, CrossEntropyWithDiceLoss)
from ..loss.DistanceLoss import MAE, MSE
from ..loss.gridloss import GridMetric, TransformedGridLoss


def get_loss(name:str, class_weight:list=None, **kwargs):
    if class_weight is not None:
        class_weight = torch.Tensor(class_weight)

    if name.lower() == 'dice':
        loss = smp.utils.losses.DiceLoss(**kwargs)
    elif name.lower() == 'iou':
        loss = smp.utils.losses.JaccardLoss(**kwargs)
    elif name.lower() == 'bce':
        loss = BCELoss(**kwargs)
    elif name.lower() == 'mae':
        loss = MAE(**kwargs)
    elif name.lower() == 'mse':
        loss = MSE(**kwargs)
    elif name.lower() == 'bcewithlogit':
        loss = BCEWithLogitsLoss(**kwargs)
    elif name.lower() == 'ce_dice':
        loss = CE_Dice_Loss(weight=class_weight, **kwargs)
    elif name.lower() == 'cross_entropy':
        loss = CrossEntropyLoss(weight=class_weight, **kwargs)
    elif name.lower() == 'cross_entropy_2d':
        loss = CrossEntropyLoss2d(weight=class_weight, **kwargs)
    elif name.lower() == 'cross_entropy_with_dice':
        loss = CrossEntropyWithDiceLoss(weight=class_weight, **kwargs)
    elif name.lower() == 'grid':
        loss = TransformedGridLoss(**kwargs)
    else:
        assert False, 'Unexpected loss name: {}'.format(name)
    return loss


def get_metric(name:str, **kwargs):
    if name.lower() == 'iou':
        metric = smp.utils.metrics.IoU(**kwargs)
    elif name.lower() == 'accuracy':
        metric = smp.utils.metrics.Accuracy(**kwargs)
    elif name.lower() == 'recall':
        metric = smp.utils.metrics.Recall(**kwargs)
    elif name.lower() == 'precision':
        metric = smp.utils.metrics.Precision(**kwargs)
    elif name.lower() == 'fscore':
        metric = smp.utils.metrics.Fscore(**kwargs)
    elif name.lower() == 'grid':
        metric = GridMetric(**kwargs)
    else:
        assert False, 'Unexpected metric name: {}'.format(name)
    return metric


def get_optimizer(name:str, params, lr:float, **kwargs):
    if name.lower() == 'adam':
        optimizer = torch.optim.Adam(params=params, lr=lr, **kwargs)
    elif name.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=params, lr=lr, **kwargs)
    else:
        assert False, 'Unexpected optimizer name: {}'.format(name)
    return optimizer


def get_scheduler(name: str, optimizer, milestones: list = None, gamma: float = 0.2,
                    T_max: int = None, eta_min: float = 0.00001, warmupepochs: int = 10):
    if name == 'MultiStepLR':
        assert milestones is not None, '"milestones" is missing.'
        scheduler = lrs.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif name == 'CosineAnnealingLR':
        assert T_max is not None, '"T_max" is missing.'
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == 'Warmup':
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmupepochs, after_scheduler=after_scheduler)
    else:
        assert False, f'No such scheduler: {name}'
    return scheduler