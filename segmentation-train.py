# torch
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader

# model
from src.models import segmentation as seg
# dataset
from src.dataset import albmentation_augmentation as aug
from src.dataset.segmentation_dataset import SegmentationDataset
# training
from src.utils import opt_util
from src.training.segmentation_trainer import TrainEpoch, ValidEpoch
# utils
from src.utils import utils

if __name__ == '__main__':
    # init_training
    from config import segmentation_settings as settings
    train_logger, valid_logger = utils.init_logging(settings.save_dir)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    train_image_pathes = utils.get_pathes(settings.train_image_path)
    train_label_pathes = utils.get_pathes(settings.train_label_path)
    valid_image_pathes = utils.get_pathes(settings.valid_image_path)
    valid_label_pathes = utils.get_pathes(settings.valid_label_path)

    train_aug = aug.get_transforms(settings.aug_settings_path, train=True)
    valid_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    train_dataset = SegmentationDataset(train_image_pathes, train_label_pathes,
                        class_num=settings.class_num, mean=settings.mean, std=settings.std,
                        label_type=settings.label_type, color_palette=settings.color_palette,
                        augmentation=train_aug)
    valid_dataset = SegmentationDataset(valid_image_pathes, valid_label_pathes,
                        class_num=settings.class_num, mean=settings.mean, std=settings.std,
                        label_type=settings.label_type, color_palette=settings.color_palette,
                        augmentation=valid_aug)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                            shuffle=True, num_workers=settings.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=settings.batch_size,
                            shuffle=False, num_workers=settings.num_workers)

    # model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    model = seg.get_SegmentationModel(settings.model, settings.encoder, activation=settings.activation,\
        encoder_weights=settings.weights, depth=settings.depth, class_num=settings.class_num)

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, settings.class_weight)

    # metric function
    print('metrics : ', settings.metrics)
    kwargs = dict()
    if settings.label_type!='binary_label':
        kwargs.update({'ignore_channels':[0]})
    metrics = [opt_util.get_metric(name, **kwargs) for name in settings.metrics]

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = utils.init_logging(settings.optimizer, model.parameters(), settings.lr)

    # scheduler
    scheduler = lrs.MultiStepLR(optimizer, milestones=settings.decay_schedule, gamma=settings.gamma)

    # trainner
    train_epoch = TrainEpoch(
        model=model, loss=loss, metrics=metrics,
        optimizer=optimizer, device=DEVICE
    )

    valid_epoch = ValidEpoch(
        model=model, loss=loss, metrics=metrics,device=DEVICE
    )

    # training
    best_score = 0.
    for epoch in range(settings.epochs):
        print('[Epoch {:03}]'.format(epoch))

        # training
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        if utils.is_best_score(valid_logs, settings.monitor_metric, best_score):
            torch.save(model.state_dict(), settings.model_save_path)
            best_score = valid_logs[settings.monitor_metric]

        # logging
        train_msg = '[Epoch {:03}] '.format(epoch)
        valid_msg = '[Epoch {:03}] '.format(epoch)
        for metric_name in train_logs.keys():
            train_msg += ' | {}:{:.5f}'.format(metric_name, train_logs[metric_name])
            valid_msg += ' | {}:{:.5f}'.format(metric_name, valid_logs[metric_name])
        train_logger.info(train_msg)
        valid_logger.info(valid_msg)

        scheduler.step()

