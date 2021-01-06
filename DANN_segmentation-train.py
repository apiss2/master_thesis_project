import warnings
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs

# model
from src.models.DA_model import Discriminator, MobilenetDiscriminator
from src.models.segmentation import get_SegmentationModel
# dataset
from src.dataset.segmentation_dataset import DASegmentationDataset
from src.dataset.albmentation_augmentation import get_transforms, get_transformedImageSize
# training
from src.loss.get_loss import get_loss, get_metric
from src.training.DANN_segmentation_trainer import TrainEpoch, ValidEpoch
# utils
from src.utils.utils import get_pathes, get_optimizer, is_best_score, init_logging

if __name__ == '__main__':
    # init_training
    from config import DANN_segmentation_settings as settings
    train_logger, valid_logger = init_logging(settings.save_dir)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    image_A_train_pathes = get_pathes(settings.image_A_train_path)
    label_A_train_pathes = get_pathes(settings.label_A_train_path)
    image_B_train_pathes = get_pathes(settings.image_B_train_path)
    label_B_train_pathes = get_pathes(settings.label_B_train_path)

    image_A_valid_pathes = get_pathes(settings.image_A_valid_path)
    label_A_valid_pathes = get_pathes(settings.label_A_valid_path)
    image_B_valid_pathes = get_pathes(settings.image_B_valid_path)
    label_B_valid_pathes = get_pathes(settings.label_B_valid_path)

    train_aug = get_transforms(settings.aug_settings_path, train=True)
    valid_aug = get_transforms(settings.aug_settings_path, train=False)

    train_dataset = DASegmentationDataset(
                        image_A_train_pathes, label_A_train_pathes,
                        image_B_train_pathes, label_B_train_pathes,
                        mean_A=settings.mean_A, std_A=settings.std_A,
                        mean_B=settings.mean_B, std_B=settings.std_B,
                        class_num=settings.class_num, augmentation=train_aug,
                        label_type=settings.label_type, color_palette=settings.color_palette
                        )
    valid_dataset = DASegmentationDataset(
                        image_A_valid_pathes, label_A_valid_pathes,
                        image_B_valid_pathes, label_B_valid_pathes,
                        mean_A=settings.mean_A, std_A=settings.std_A,
                        mean_B=settings.mean_B, std_B=settings.std_B,
                        class_num=settings.class_num, augmentation=valid_aug,
                        label_type=settings.label_type, color_palette=settings.color_palette
                        )

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,\
                            shuffle=True, num_workers=settings.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=settings.batch_size, \
                            shuffle=False, num_workers=settings.num_workers)

    # segmentation_model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    model = get_SegmentationModel(settings.model, settings.encoder, activation=settings.activation,\
        encoder_weights=settings.weights, depth=settings.depth, class_num=settings.class_num)

    # discriminator definition
    with torch.no_grad():
        sample = torch.rand((2, 3, settings.image_size, settings.image_size))
        sample = model.encoder.forward(sample)[-1]
    model_D = Discriminator(sample, settings.discriminator_channels)

    # loss function
    print('loss_seg : ', settings.loss_seg)
    print('loss_DA : ', settings.loss_DA)
    loss = get_loss(settings.loss_seg, settings.class_weight)
    loss_D = get_loss(settings.loss_DA)

    # metric function
    print('metrics_seg : ', settings.metrics)
    print('metrics_d : ', settings.metrics_D)
    metrics = [get_metric(name, ignore_channels=[0]) for name in settings.metrics]
    metrics_D = [get_metric(name) for name in settings.metrics_D]

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = get_optimizer(settings.optimizer, model.parameters(), settings.lr)
    optimizer_D = get_optimizer(settings.optimizer, model_D.parameters(), settings.lr)

    # scheduler
    scheduler = lrs.MultiStepLR(optimizer, milestones=settings.decay_schedule, gamma=settings.gamma)
    scheduler_D = lrs.MultiStepLR(optimizer_D, milestones=settings.decay_schedule, gamma=settings.gamma)

    # trainner
    train_epoch = TrainEpoch(
        model=model, loss=loss, metrics=metrics,
        model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
        optimizer=optimizer, optimizer_D=optimizer_D,
        modelupdate_freq=settings.modelupdate_freq, device=DEVICE
    )

    valid_epoch = ValidEpoch(
        model=model, loss=loss, metrics=metrics,
        model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
        device=DEVICE
    )

    # training
    best_score = 0
    for epoch in range(settings.epochs):
        print('[Epoch {:03}]'.format(epoch))
        # training
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        if is_best_score(valid_logs, settings.monitor_metric, best_score) and epoch>=int(settings.epochs/2):
            torch.save(model.state_dict(), settings.model_save_path)
            best_score = valid_logs[settings.monitor_metric]

        # logging
        train_msg = '[Epoch {:03}] '.format(epoch)
        valid_msg = '[Epoch {:03}] '.format(epoch)
        for metric_name in list(train_logs.keys()):
            train_msg += ' | {}:{:02.4f}'.format(metric_name, train_logs[metric_name])
            valid_msg += ' | {}:{:02.4f}'.format(metric_name, valid_logs[metric_name])
        train_logger.info(train_msg)
        valid_logger.info(valid_msg)

        scheduler.step()
        scheduler_D.step()