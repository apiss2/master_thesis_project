import os
import warnings
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader

# model
from src.models.DA_model import Discriminator
from src.models.cnngeometric import CNNGeometric, CNNGeometricDecoder, CNNGeometric_MCDDA
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

from src.training.cnngeometric_multidomain_trainer import TrainEpoch, ValidEpoch

if __name__ == '__main__':
    # init_training
    train_logger, valid_logger  = utils.init_logging(settings.save_dir)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    kargs = {'device': DEVICE}

    # datasets
    train_aug = aug.get_transforms(settings.aug_settings_path, train=True)
    valid_aug = aug.get_transforms(settings.aug_settings_path, train=False)

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
    if 'MCDDA' in settings.experiment_name:
        encoder = FeatureExtraction(name=settings.encoder, depth=settings.depth, weights=settings.weights)
        with torch.no_grad():
            sample = encoder.forward(torch.rand((2, 3, settings.image_size, settings.image_size)))
        decoder = CNNGeometricDecoder(sample, output_dim=settings.cnn_output_dim)
        if settings.pretrained_encoder_path is not None:
            state_dict = torch.load(settings.pretrained_encoder_path)
            encoder.load_state_dict(state_dict)
            kargs.update({'freeze_encoder': True})
        model = CNNGeometric_MCDDA(encoder, decoder)
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

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, geometric=settings.geometric, device=DEVICE)
    kargs.update({'loss': loss})

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

    # trainer
    valid_epoch = ValidEpoch(**kargs)

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = opt_util.get_optimizer(settings.optimizer, model.parameters(), settings.lr)
    kargs.update({'optimizer': optimizer})

    # scheduler
    print('scheduler : ', settings.scheduler_type)
    scheduler = opt_util.get_scheduler(settings.scheduler_type, optimizer, milestones=settings.decay_schedule, gamma=settings.gamma,
                                    T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)

    # trainner
    train_epoch = TrainEpoch(**kargs)

    # training
    for epoch in range(settings.epochs):
        print('[Epoch {:03}] (lr = {:03})'.format(epoch, optimizer.param_groups[0]['lr']))
        # training
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        if epoch==0:
            best_score = valid_logs[settings.monitor_metric] 
        elif utils.is_best_score(valid_logs, settings.monitor_metric, best_score, maximize=False):
            print('! save model !')
            torch.save(train_epoch.model.state_dict(), settings.model_save_path)
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
