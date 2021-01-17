import warnings
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs

# model
from src.models.DA_model import Discriminator
from src.models import segmentation as seg
# dataset
from src.dataset.segmentation_dataset import SegmentationDataset, DASegmentationDataset
from src.dataset import albmentation_augmentation as aug
# training
from src.utils import opt_util
# utils
from src.utils import utils
import settings

if __name__ == '__main__':
    # init_training
    train_logger, valid_logger = utils.init_logging(settings.save_dir)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if 'MCDDA' in settings.task_name:
        from src.training.MCDDA_segmentation_trainer import TrainEpoch, ValidEpoch
    elif 'WDGR' in settings.task_name:
        from src.training.WDGR_segmentation_trainer import TrainEpoch, ValidEpoch
    elif 'DANN' in settings.task_name:
        from src.training.DANN_segmentation_trainer import TrainEpoch, ValidEpoch
    elif settings.task_name == 'segmentation':
        from src.training.segmentation_trainer import TrainEpoch, ValidEpoch
    else:
        assert False, f'No such task : {settings.task_name}'

    kargs = {'device': DEVICE}

    # datasets
    train_aug = aug.get_transforms(settings.aug_settings_path, train=True)
    valid_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    if settings.task_name == 'segmentation':
        train_image_pathes = utils.get_pathes(settings.train_image_path)
        train_label_pathes = utils.get_pathes(settings.train_label_path)
        valid_image_pathes = utils.get_pathes(settings.valid_image_path)
        valid_label_pathes = utils.get_pathes(settings.valid_label_path)

        train_dataset = SegmentationDataset(train_image_pathes, train_label_pathes,
                            class_num=settings.class_num, mean=settings.mean, std=settings.std,
                            label_type=settings.label_type, color_palette=settings.color_palette,
                            augmentation=train_aug)
        valid_dataset = SegmentationDataset(valid_image_pathes, valid_label_pathes,
                            class_num=settings.class_num, mean=settings.mean, std=settings.std,
                            label_type=settings.label_type, color_palette=settings.color_palette,
                            augmentation=valid_aug)
    else:
        image_A_train_pathes = utils.get_pathes(settings.image_A_train_path)
        label_A_train_pathes = utils.get_pathes(settings.label_A_train_path)
        image_B_train_pathes = utils.get_pathes(settings.image_B_train_path)
        label_B_train_pathes = utils.get_pathes(settings.label_B_train_path)

        image_A_valid_pathes = utils.get_pathes(settings.image_A_valid_path)
        label_A_valid_pathes = utils.get_pathes(settings.label_A_valid_path)
        image_B_valid_pathes = utils.get_pathes(settings.image_B_valid_path)
        label_B_valid_pathes = utils.get_pathes(settings.label_B_valid_path)


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

    # model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    if 'MCDDA' in settings.task_name:
        model = seg.SegmentationEncoder(encoder_name=settings.encoder, model_name=settings.model,
            depth=settings.depth, weights=settings.weights)
        with torch.no_grad():
            sample = model.forward(torch.rand((2, 3, settings.image_size, settings.image_size)))
        decoder_1 = seg.SegmentationDecoder(settings.model, sample, depth=settings.depth,
            activation=settings.activation, classes=settings.class_num)
        decoder_2 = seg.SegmentationDecoder(settings.model, sample, depth=settings.depth,
            activation=settings.activation, classes=settings.class_num)
        kargs.update({'model': model, 'decoder_1': decoder_1, 'decoder_2': decoder_2})
    else:
        model = seg.get_SegmentationModel(settings.model, settings.encoder, activation=settings.activation,\
            encoder_weights=settings.weights, depth=settings.depth, class_num=settings.class_num)
        kargs.update({'model': model})

    # discriminator definition
    if settings.task_name == 'cnngeometric':
        pass
    elif 'MCDDA' in settings.task_name:
        pass
    else:
        if 'WDGR' in settings.task_name:
            batchnorm = False
            gradient_reversal = False
            last_activation = None
        elif 'DANN' in settings.task_name:
            batchnorm = True
            gradient_reversal = True
            last_activation = 'sigmoid'
        with torch.no_grad():
            sample = torch.rand((2, 3, settings.image_size, settings.image_size))
            sample = model.encoder.forward(sample)[-1]
        model_D = Discriminator(sample, settings.discriminator_channels,
            batchnorm=batchnorm, gradient_reversal=gradient_reversal,
            use_GAP=True, last_activation=last_activation)
        kargs.update({'model_D': model_D})

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, settings.class_weight)
    kargs.update({'loss': loss})

    if settings.task_name != 'segmentation':
        print('loss_D : ', settings.loss_D)
        loss_D = opt_util.get_loss(settings.loss_D)
        kargs.update({'loss_D': loss_D})

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [opt_util.get_metric(name, ignore_channels=[0]) for name in settings.metrics]
    kargs.update({'metrics': metrics})
    if settings.task_name == 'segmentation':
        pass
    else:
        print('metrics_d : ', settings.metrics_D)
        metrics_D = [opt_util.get_metric(name) for name in settings.metrics_D]
        kargs.update({'metrics_D': metrics_D})

    # trainer
    valid_epoch = ValidEpoch(**kargs)

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = opt_util.get_optimizer(settings.optimizer, model.parameters(), settings.lr)
    kargs.update({'optimizer': optimizer})
    if 'MCDDA' in settings.task_name:
        optimizer_1 = opt_util.get_optimizer(settings.optimizer, decoder_1.parameters(), settings.lr)
        optimizer_2 = opt_util.get_optimizer(settings.optimizer, decoder_2.parameters(), settings.lr)
        kargs.update({'optimizer_1': optimizer_1, 'optimizer_2': optimizer_2})
    elif settings.task_name=='segmentation':
        pass
    else:
        optimizer_D = opt_util.get_optimizer(settings.optimizer_D, model_D.parameters(), settings.lr)
        kargs.update({'optimizer_D': optimizer_D})

    # scheduler
    print('scheduler : ', settings.scheduler_type)
    scheduler = opt_util.get_scheduler(settings.scheduler_type, optimizer, milestones=settings.decay_schedule, gamma=settings.gamma,
                                    T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
    schedulers = [scheduler]
    if 'MCDDA' in settings.task_name:
        scheduler_1 = opt_util.get_scheduler(settings.scheduler_type, optimizer_1, milestones=settings.decay_schedule, gamma=settings.gamma,
                                        T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
        schedulers.append(scheduler_1)
        scheduler_2 = opt_util.get_scheduler(settings.scheduler_type, optimizer_2, milestones=settings.decay_schedule, gamma=settings.gamma,
                                        T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
        schedulers.append(scheduler_2)
    elif settings.task_name=='segmentation':
        pass
    else:
        scheduler_D = opt_util.get_scheduler(settings.scheduler_type, optimizer_D, milestones=settings.decay_schedule, gamma=settings.gamma,
                                        T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
        schedulers.append(scheduler_D)

    # trainner
    if settings.task_name != 'segmentation':
        kargs.update({
            'modelupdate_freq': settings.modelupdate_freq,
            'discupdate_freq': settings.discupdate_freq,
        })
    train_epoch = TrainEpoch(**kargs)

    # training
    best_score = 0
    for epoch in range(settings.epochs):
        print('[Epoch {:03}]'.format(epoch))
        # training
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        if utils.is_best_score(valid_logs, settings.monitor_metric, best_score) and epoch>=int(settings.epochs/2):
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

        for s in schedulers:
            s.step()