import json
import warnings

# torch
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs
import segmentation_models_pytorch as smp

# model
from src.models.DA_model import Discriminator, MobilenetDiscriminator
from src.models.segmentation import get_SegmentationModel
# dataset
from src.dataset.segmentation_dataset import DASegmentationDataset
from src.dataset.albmentation_augmentation import get_transforms, get_transformedImageSize
# training
from src.loss.get_loss import get_loss, get_metric
from src.utils.DAsegmentation_train_util import TrainEpoch, ValidEpoch
# utils
from src.utils.utils import get_pathes, get_optimizer, is_best_score, init_logging
from src.utils.utils import get_normParam, get_colorPalette, get_classWeight

warnings.simplefilter('ignore')

if __name__ == '__main__':

    # load config file
    config, train_logger, valid_logger, model_save_path, save_dir = init_logging()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.getint('batch_size')
    lr = config.getfloat('lr')
    decay_schedule = config.get('decay_schedule')
    gamma = config.getfloat('gamma')

    # datasets
    image_A_train_pathes = get_pathes(config.get('image_A_train'))
    label_A_train_pathes = get_pathes(config.get('label_A_train'))
    image_B_train_pathes = get_pathes(config.get('image_B_train'))
    label_B_train_pathes = get_pathes(config.get('label_B_train'))

    image_A_valid_pathes = get_pathes(config.get('image_A_valid'))
    label_A_valid_pathes = get_pathes(config.get('label_A_valid'))
    image_B_valid_pathes = get_pathes(config.get('image_B_valid'))
    label_B_valid_pathes = get_pathes(config.get('label_B_valid'))

    mean_A, std_A = get_normParam(config.get('mean_A'), config.get('std_A'))
    mean_B, std_B = get_normParam(config.get('mean_B'), config.get('std_B'))

    class_num = config.getint('class_num')
    label_type = config.get('label_type')
    color_palette = get_colorPalette(config.get('color_palette'))
    aug_settings_path = config.get('augmentation_setting_json')
    image_size = get_transformedImageSize(aug_settings_path)

    train_dataset = DASegmentationDataset(image_A_train_pathes, label_A_train_pathes,
                        image_B_train_pathes, label_B_train_pathes, class_num=class_num,
                        mean_A=mean_A, mean_B=mean_B, std_A=std_A, std_B=std_B,
                        color_palette=color_palette, label_type=label_type,
                        augmentation=get_transforms(aug_settings_path, train=True))
    valid_dataset = DASegmentationDataset(image_A_valid_pathes, label_A_valid_pathes,
                        image_B_valid_pathes, label_B_valid_pathes, class_num=class_num,
                        mean_A=mean_A, mean_B=mean_B, std_A=std_A, std_B=std_B,
                        augmentation=get_transforms(aug_settings_path, train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=config.getint('num_workers'))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, \
                            shuffle=False, num_workers=config.getint('num_workers'))

    # segmentation_model definition
    print('model : ', config.get('model'))
    print('encoder : ', config.get('encoder'))
    model = get_SegmentationModel(config.get('model'), config.get('encoder'), \
        depth=config.getint('depth'), class_num=config.getint('class_num'))

    # discriminator definition
    with torch.no_grad():
        sample = torch.rand((2, 3, image_size, image_size))
        sample = model.encoder.forward(sample)[-1]
    channels = config.get('discriminator_channels')
    channels = list(map(lambda x: int(x), channels.split('-')))
    model_D = Discriminator(sample, channels)

    # loss function
    loss_seg = config.get('loss_seg')
    loss_DA  = config.get('loss_DA')
    print('loss_seg : ', loss_seg)
    print('loss_DA  : ', loss_DA)
    loss = get_loss(loss_seg)
    loss_D  = get_loss(loss_DA)

    # metric function
    metrics = [get_metric(name) for name in config.get('metrics').split('-')]
    metrics_D = [get_metric(name) for name in config.get('metrics_D').split('-')]

    # optimizer
    optimizer_name = config.get('optimizer')
    print('optimizer : ', optimizer_name)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    optimizer_D = get_optimizer(optimizer_name, model_D.parameters(), lr)

    # scheduler
    decay_schedule = list(map(lambda x: int(x), decay_schedule.split('-')))
    scheduler = lrs.MultiStepLR(optimizer, milestones=decay_schedule, gamma=gamma)
    scheduler_D = lrs.MultiStepLR(optimizer_D, milestones=decay_schedule, gamma=gamma)

    # trainner
    train_epoch = TrainEpoch(
        model=model, loss=loss, metrics=metrics,
        model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
        optimizer=optimizer, optimizer_D=optimizer_D, device=DEVICE
    )

    valid_epoch = ValidEpoch(
        model=model, loss=loss, metrics=metrics,
        model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
        device=DEVICE
    )

    # training
    best_score = 0
    monitor_metric = config.get('monitor_metric')
    epochs = config.getint('epochs')
    for epoch in range(epochs):
        print('[Epoch {:03}]'.format(epoch))
        # training
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        if is_best_score(valid_logs, monitor_metric, best_score) and epoch>=int(epochs/2):
            torch.save(model.state_dict(), model_save_path)
            best_score = valid_logs[monitor_metric]

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