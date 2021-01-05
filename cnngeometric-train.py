import warnings
warnings.simplefilter('ignore')

# torch
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader

# model
from src.models.cnngeometric import CNNGeometric
from src.transformation.core import GeometricTnf
# dataset
from src.dataset.albmentation_augmentation import get_transforms, get_transformedImageSize
from src.dataset.geometric_transform_dataset import GeometricDataset
# training
from src.training.cnngeometric_trainer import TrainEpoch, ValidEpoch
from src.loss.get_loss import get_loss, get_metric
# utils
from src.utils.utils import get_pathes, is_best_score, init_logging
from src.utils.utils import get_optimizer

if __name__ == '__main__':
    # init_training
    from config import cnngeometric_settings as settings
    train_logger, valid_logger  = init_logging(settings.save_dir)

    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda" if use_cuda else "cpu"

    # datasets
    train_images_pathes = get_pathes(settings.train_image_path)
    train_labels_pathes = get_pathes(settings.train_label_path)
    valid_images_pathes = get_pathes(settings.valid_image_path)
    valid_labels_pathes = get_pathes(settings.valid_label_path)

    train_aug = get_transforms(settings.aug_settings_path, train=True)
    valid_aug = get_transforms(settings.aug_settings_path, train=False)

    train_dataset = GeometricDataset(train_images_pathes, train_labels_pathes,\
                                    geometric=settings.geometric, augmentation=train_aug,\
                                    class_num=settings.class_num, mean=settings.mean, std=settings.std, \
                                    random_t_tps=settings.random_t_tps,\
                                    label_type=settings.label_type, color_palette=settings.color_palette)
    valid_dataset = GeometricDataset(image_pathes=valid_images_pathes, label_pathes=train_labels_pathes,\
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
                        output_dim=settings.cnn_output_dim, input_size=settings.image_size, use_cuda=use_cuda,
                        freeze_FeatureExtractionModel=settings.freeze)

    if settings.pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path))

    # loss function
    print('loss : ', settings.loss)
    loss = get_loss(settings.loss, settings.class_weight)

    # metric function
    metrics = [get_metric(name) for name in settings.metrics]

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = get_optimizer(settings.optimizer, model.parameters(), settings.lr)

    # scheduler
    scheduler = lrs.MultiStepLR(optimizer, milestones=settings.decay_schedule, gamma=settings.gamma)

    # trainner
    train_epoch = TrainEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        optimizer=optimizer, geometric_transform=geometric_transform
    )
    valid_epoch = ValidEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        geometric_transform=geometric_transform
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
        if is_best_score(valid_logs, settings.monitor_metric, best_score):
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