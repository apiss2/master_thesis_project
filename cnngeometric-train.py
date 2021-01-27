import os
import warnings
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader

# model
from src.models.DA_model import Discriminator
from src.models.cnngeometric import CNNGeometric, CNNGeometricDecoder
from src.models.cnngeometric_base import FeatureExtraction
from src.transformation.core import GeometricTnf
# dataset
from src.dataset import albmentation_augmentation as aug
from src.dataset.geometric_transform_dataset import GeometricDataset, DAGeometricDataset
# training
from src.training.cnngeometric_trainer import TrainEpoch, ValidEpoch
from src.utils import opt_util
# utils
from src.utils import utils
import settings

if __name__ == '__main__':
    # init_training
    train_logger, valid_logger  = utils.init_logging(settings.save_dir)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if 'MCDDA' in settings.task_name:
        from src.training.MCDDA_cnngeometric_trainer import TrainEpoch, ValidEpoch
    elif 'WDGR' in settings.task_name:
        from src.training.WDGR_cnngeometric_trainer import TrainEpoch, ValidEpoch
    elif 'DANN' in settings.task_name:
        from src.training.DANN_cnngeometric_trainer import TrainEpoch, ValidEpoch
    elif settings.task_name == 'cnngeometric':
        from src.training.cnngeometric_trainer import TrainEpoch, ValidEpoch
    else:
        assert False, f'No such task : {settings.task_name}'

    kargs = {'device': DEVICE}

    # datasets
    train_aug = aug.get_transforms(settings.aug_settings_path, train=True)
    valid_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    if settings.task_name == 'cnngeometric':
        train_images_pathes = utils.get_pathes(settings.train_image_path)
        train_labels_pathes = utils.get_pathes(settings.train_label_path)
        valid_images_pathes = utils.get_pathes(settings.valid_image_path)
        valid_labels_pathes = utils.get_pathes(settings.valid_label_path)

        train_dataset = GeometricDataset(train_images_pathes, train_labels_pathes,\
                                        geometric=settings.geometric, augmentation=train_aug,\
                                        class_num=settings.class_num, mean=settings.mean, std=settings.std, \
                                        random_t_tps=settings.random_t_tps,\
                                        label_type=settings.label_type, color_palette=settings.color_palette)
        valid_dataset = GeometricDataset(image_pathes=valid_images_pathes, label_pathes=valid_labels_pathes,\
                                        geometric=settings.geometric, augmentation=valid_aug,\
                                        class_num=settings.class_num, mean=settings.mean, std=settings.std,\
                                        random_t_tps=settings.random_t_tps,\
                                        label_type=settings.label_type, color_palette=settings.color_palette)
    else:
        image_A_train_pathes = utils.get_pathes(settings.image_A_train_path)
        label_A_train_pathes = utils.get_pathes(settings.label_A_train_path)
        image_B_train_pathes = utils.get_pathes(settings.image_B_train_path)
        label_B_train_pathes = utils.get_pathes(settings.label_B_train_path)

        image_A_valid_pathes = utils.get_pathes(settings.image_A_valid_path)
        label_A_valid_pathes = utils.get_pathes(settings.label_A_valid_path)
        image_B_valid_pathes = utils.get_pathes(settings.image_B_valid_path)
        label_B_valid_pathes = utils.get_pathes(settings.label_B_valid_path)

        train_dataset = DAGeometricDataset(image_A_train_pathes, label_A_train_pathes,
                            image_B_train_pathes, label_B_train_pathes,
                            class_num=settings.class_num, color_palette=settings.color_palette,
                            label_type=settings.label_type, geometric=settings.geometric,
                            mean_A=settings.mean_A, std_A=settings.std_A,
                            mean_B=settings.mean_B, std_B=settings.std_B,
                            augmentation=train_aug)
        valid_dataset = DAGeometricDataset(image_A_valid_pathes, label_A_valid_pathes,
                            image_B_valid_pathes, label_B_valid_pathes,
                            class_num=settings.class_num, color_palette=settings.color_palette,
                            label_type=settings.label_type, geometric=settings.geometric,
                            mean_A=settings.mean_A, std_A=settings.std_A,
                            mean_B=settings.mean_B, std_B=settings.std_B,
                            augmentation=valid_aug)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                            shuffle=True, num_workers=settings.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=settings.batch_size,
                            shuffle=False, num_workers=settings.num_workers)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, device=DEVICE, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)
    kargs.update({'geometric_transform': geometric_transform})

    # model definition
    print('encoder : ', settings.encoder)
    if 'MCDDA' in settings.task_name:
        model = FeatureExtraction(name=settings.encoder, depth=settings.depth, weights=settings.weights)
        with torch.no_grad():
            sample = model.forward(torch.rand((2, 3, settings.image_size, settings.image_size)))
        decoder_1 = CNNGeometricDecoder(sample, output_dim=settings.cnn_output_dim)
        decoder_2 = CNNGeometricDecoder(sample, output_dim=settings.cnn_output_dim)
        kargs.update({'model': model, 'decoder_1': decoder_1, 'decoder_2': decoder_2, 'geometric':settings.geometric})
        if settings.pretrained_encoder_path is not None:
            state_dict = torch.load(settings.pretrained_encoder_path)
            state_dict['_fc.bias'] = None
            state_dict['_fc.weight'] = None
            model.model.load_state_dict(state_dict)
            kargs.update({'freeze_encoder': True})
    else:
        model = CNNGeometric(encoder=settings.encoder,
                        in_channels=settings.in_channels,
                        depth=settings.depth,
                        weights=settings.weights,
                        output_dim=settings.cnn_output_dim,
                        input_size=settings.image_size,
                        freeze_encoder=settings.freeze_encoder)
        if settings.pretrained_encoder_path is not None:
            state_dict = torch.load(settings.pretrained_encoder_path)
            state_dict['_fc.bias'] = None
            state_dict['_fc.weight'] = None
            model.encoder.model.load_state_dict(state_dict)
            kargs.update({'freeze_encoder': True})
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
            use_GAP=False, last_activation=last_activation)
        kargs.update({'model_D': model_D})

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, geometric=settings.geometric, device=DEVICE)
    kargs.update({'loss': loss})
    if settings.task_name == 'cnngeometric':
        pass
    else:
        print('loss_D  : ', settings.loss_D)
        if '2D' in settings.loss_D:
            loss_D  = opt_util.get_loss(settings.loss_D, sum=True)
        else:
            loss_D  = opt_util.get_loss(settings.loss_D)
        kargs.update({'loss_D': loss_D})

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [opt_util.get_metric(name, geometric=settings.geometric, device=DEVICE) for name in settings.metrics]
    kargs.update({'metrics': metrics})
    if settings.metrics_seg is not None:
        print('metrics_seg : ', settings.metrics_seg)
        if settings.class_num==1:
            metrics_seg = [opt_util.get_metric(name) for name in settings.metrics_seg]
        else:
            metrics_seg = [opt_util.get_metric(name, ignore_channels=[0]) for name in settings.metrics_seg]
        kargs.update({'metrics_seg': metrics_seg})
    else:
        metrics_seg = None
    if settings.task_name=='cnngeometric':
        pass
    else:
        print('metrics_D  : ', settings.metrics_D)
        metrics_D = [opt_util.get_metric(name) for name in settings.metrics_D]
        kargs.update({'metrics_D': metrics_D})

    # trainer
    valid_epoch = ValidEpoch(**kargs)

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = opt_util.get_optimizer(settings.optimizer, model.parameters(), settings.lr)
    kargs.update({'optimizer': optimizer})
    if settings.task_name=='cnngeometric':
        pass
    elif 'MCDDA' in settings.task_name:
        optimizer_1 = opt_util.get_optimizer(settings.optimizer, decoder_1.parameters(), settings.lr)
        optimizer_2 = opt_util.get_optimizer(settings.optimizer, decoder_2.parameters(), settings.lr)
        kargs.update({'optimizer_1': optimizer_1, 'optimizer_2': optimizer_2})
    else:
        optimizer_D = opt_util.get_optimizer(settings.optimizer_D, model_D.parameters(), settings.lr_D)
        kargs.update({'optimizer_D': optimizer_D})

    # scheduler
    schedulers = list()
    print('scheduler : ', settings.scheduler_type)
    scheduler = opt_util.get_scheduler(settings.scheduler_type, optimizer, milestones=settings.decay_schedule, gamma=settings.gamma,
                                    T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
    schedulers.append(scheduler)
    if settings.task_name=='cnngeometric':
        pass
    elif 'MCDDA' in settings.task_name:
        scheduler_1 = opt_util.get_scheduler(settings.scheduler_type, optimizer_1, milestones=settings.decay_schedule, gamma=settings.gamma,
                                        T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
        schedulers.append(scheduler_1)
        scheduler_2 = opt_util.get_scheduler(settings.scheduler_type, optimizer_2, milestones=settings.decay_schedule, gamma=settings.gamma,
                                        T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
        schedulers.append(scheduler_2)
    else:
        scheduler_D = opt_util.get_scheduler(settings.scheduler_type, optimizer_D, milestones=settings.decay_schedule, gamma=settings.gamma,
                                    T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)
        schedulers.append(scheduler_D)

    # trainner
    if settings.task_name != 'cnngeometric':
        kargs.update({
            'modelupdate_freq': settings.modelupdate_freq,
            'discupdate_freq': settings.discupdate_freq,
        })
    train_epoch = TrainEpoch(**kargs)

    # training
    for epoch in range(settings.epochs):
        print('[Epoch {:03}] (lr = {:03})'.format(epoch, optimizer.param_groups[0]['lr']))
        # training
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        save_flug = False
        if epoch==0:
            best_score = valid_logs[settings.monitor_metric]
        elif utils.is_best_score(valid_logs, settings.monitor_metric, best_score, maximize=False):
            if 'MCDDA' in settings.task_name:
                torch.save(train_epoch.model.state_dict(), os.path.join(settings.save_dir, 'best_encoder.pth'))
                torch.save(train_epoch.decoder_2.state_dict(), os.path.join(settings.save_dir, 'best_decoder_1.pth'))
                torch.save(train_epoch.decoder_1.state_dict(), os.path.join(settings.save_dir, 'best_decoder_2.pth'))
                best_score = valid_logs[settings.monitor_metric]
                save_flug=True
            elif 'WDGR' in settings.task_name:
                if abs(valid_logs['WassersteinGPLoss'])<2:
                    torch.save(train_epoch.model.state_dict(), settings.model_save_path)
                    best_score = valid_logs[settings.monitor_metric]
                    save_flug=True
            else:
                torch.save(train_epoch.model.state_dict(), settings.model_save_path)
                best_score = valid_logs[settings.monitor_metric]
                save_flug=True
        if save_flug:
            print('!!!Save model!!!')

        # logging
        train_msg = '[Epoch {:03}](Save model = {}) '.format(epoch, str(save_flug).rjust(5))
        valid_msg = '[Epoch {:03}](Save model = {}) '.format(epoch, str(save_flug).rjust(5))
        for metric_name in list(train_logs.keys()):
            train_msg += ' | {}:{:02.4f}'.format(metric_name, train_logs[metric_name])
            valid_msg += ' | {}:{:02.4f}'.format(metric_name, valid_logs[metric_name])
        train_logger.info(train_msg)
        valid_logger.info(valid_msg)

        for s in schedulers:
            s.step()

    if 'MCDDA' in settings.task_name:
        torch.save(train_epoch.model.state_dict(), os.path.join(settings.save_dir, 'last_encoder.pth'))
        torch.save(train_epoch.decoder_2.state_dict(), os.path.join(settings.save_dir, 'last_decoder_1.pth'))
        torch.save(train_epoch.decoder_1.state_dict(), os.path.join(settings.save_dir, 'last_decoder_2.pth'))
        save_flug=True
    else:
        torch.save(train_epoch.model.state_dict(), os.path.join(settings.save_dir, 'last_model.pth'))