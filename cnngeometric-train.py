import warnings
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader

# model
from src.models.cnngeometric import CNNGeometric
from src.transformation.core import GeometricTnf
# dataset
from src.dataset import albmentation_augmentation as aug
from src.dataset.geometric_transform_dataset import GeometricDataset
# training
from src.training.cnngeometric_trainer import TrainEpoch, ValidEpoch
from src.utils import opt_util
# utils
from src.utils import utils
import settings

if __name__ == '__main__':
    # init_training
    train_logger, valid_logger  = utils.init_logging(settings.save_dir)
    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda" if use_cuda else "cpu"

    # datasets
    train_images_pathes = utils.get_pathes(settings.train_image_path)
    train_labels_pathes = utils.get_pathes(settings.train_label_path)
    valid_images_pathes = utils.get_pathes(settings.valid_image_path)
    valid_labels_pathes = utils.get_pathes(settings.valid_label_path)

    train_aug = aug.get_transforms(settings.aug_settings_path, train=True)
    valid_aug = aug.get_transforms(settings.aug_settings_path, train=False)

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

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                            shuffle=True, num_workers=settings.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=settings.batch_size,
                            shuffle=False, num_workers=settings.num_workers)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, use_cuda=use_cuda, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)

    # model definition
    model = CNNGeometric(encoder=settings.encoder, in_channels=settings.in_channels,
                        depth=settings.depth, weights=settings.weights,
                        output_dim=settings.cnn_output_dim, input_size=settings.image_size,
                        freeze_FeatureExtractionModel=settings.freeze)

    if settings.pretrained_model_path is not None:
        model.load_state_dict(torch.load(settings.pretrained_model_path))

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, geometric=settings.geometric)

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [opt_util.get_metric(name, geometric=settings.geometric) for name in settings.metrics]
    if settings.metrics_seg is not None:
        print('metrics_seg : ', settings.metrics_seg)
        metrics_seg = [opt_util.get_metric(name) for name in settings.metrics_seg]
    else:
        metrics_seg = None

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = opt_util.get_optimizer(settings.optimizer, model.parameters(), settings.lr)

    # scheduler
    print('scheduler : ', settings.scheduler_type)
    scheduler = opt_util.get_scheduler(settings.scheduler_type, optimizer, milestones=settings.decay_schedule, gamma=settings.gamma,
                                    T_max=settings.epochs, eta_min=settings.eta_min, warmupepochs=settings.warmupepochs)

    # trainner
    train_epoch = TrainEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        optimizer=optimizer, geometric_transform=geometric_transform,
        segmentation_metrics=metrics_seg
    )
    valid_epoch = ValidEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        geometric_transform=geometric_transform,
        segmentation_metrics=metrics_seg
    )

    #########################訓練###################################################
    # training
    best_score = 0
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
        for metric_name in list(train_logs.keys()):
            train_msg += ' | {}:{:02.4f}'.format(metric_name, train_logs[metric_name])
            valid_msg += ' | {}:{:02.4f}'.format(metric_name, valid_logs[metric_name])
        train_logger.info(train_msg)
        valid_logger.info(valid_msg)

        scheduler.step()