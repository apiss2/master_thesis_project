import warnings
warnings.simplefilter('ignore')

# torch
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader

# model
from src.models.DA_model import Discriminator
from src.models.cnngeometric import CNNGeometric
from src.transformation.core import GeometricTnf
# dataset
from src.dataset.geometric_transform_dataset import DAGeometricDataset
from src.dataset import albmentation_augmentation as aug
# training
from src.utils import opt_util
from src.training.DANN_cnngeometric_trainer import TrainEpoch, ValidEpoch
# utils
from src.utils import utils


if __name__ == '__main__':
    # init_training
    from config import DANN_cnngeometric_settings as settings
    train_logger, valid_logger = utils.init_logging(settings.save_dir)

    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda" if use_cuda else "cpu"

    # datasets
    image_A_train_pathes = utils.get_pathes(settings.image_A_train_path)
    label_A_train_pathes = utils.get_pathes(settings.label_A_train_path)
    image_B_train_pathes = utils.get_pathes(settings.image_B_train_path)
    label_B_train_pathes = utils.get_pathes(settings.label_B_train_path)

    image_A_valid_pathes = utils.get_pathes(settings.image_A_valid_path)
    label_A_valid_pathes = utils.get_pathes(settings.label_A_valid_path)
    image_B_valid_pathes = utils.get_pathes(settings.image_B_valid_path)
    label_B_valid_pathes = utils.get_pathes(settings.label_B_valid_path)

    train_aug = aug.get_transforms(settings.aug_settings_path, train=True)
    valid_aug = aug.get_transforms(settings.aug_settings_path, train=False)

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

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,\
                            shuffle=True, num_workers=settings.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=settings.batch_size,\
                            shuffle=False, num_workers=settings.num_workers)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, use_cuda=use_cuda, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)

    # geometric model definition
    model = CNNGeometric(encoder=settings.encoder, in_channels=settings.in_channels,
                        depth=settings.depth, weights=settings.weights,
                        output_dim=settings.cnn_output_dim,
                        input_size=settings.image_size,
                        freeze_FeatureExtractionModel=settings.freeze)

    if settings.pretrained_model_path is not None:
        model.load_state_dict(torch.load(settings.pretrained_model_path))

    # discriminator definition
    with torch.no_grad():
        sample = torch.rand((2, 3, settings.image_size, settings.image_size))
        sample = model.FeatureExtraction.forward(sample)[-1]
    model_D = Discriminator(sample, settings.discriminator_channels)

    # loss function
    print('loss : ', settings.loss)
    print('loss_D  : ', settings.loss_D)
    loss = opt_util.get_loss(settings.loss)
    loss_D  = opt_util.get_loss(settings.loss_D)

    # metric function
    metrics = [opt_util.get_metric(name) for name in settings.metrics]
    metrics_seg = [opt_util.get_metric(name) for name in settings.metrics_seg]
    metrics_D = [opt_util.get_metric(name) for name in settings.metrics_D]

    # optimizer
    print('optimizer : ', settings.optimizer)
    optimizer = opt_util.get_optimizer(settings.optimizer, model.parameters(), settings.lr)
    optimizer_D = opt_util.get_optimizer(settings.optimizer, model_D.parameters(), settings.lr)

    # scheduler
    scheduler = lrs.MultiStepLR(optimizer, milestones=settings.decay_schedule, gamma=settings.gamma)
    scheduler_D = lrs.MultiStepLR(optimizer_D, milestones=settings.decay_schedule, gamma=settings.gamma)

    # trainner
    train_epoch = TrainEpoch(
        model=model, loss = loss, metrics = metrics,
        model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
        modelupdate_freq=settings.modelupdate_freq,
        discupdate_freq=settings.discupdate_freq,
        optimizer=optimizer, optimizer_D=optimizer_D,
        geometric_transform=geometric_transform, device=DEVICE,
    )
    valid_epoch = ValidEpoch(
        model=model, loss = loss, metrics = metrics, device=DEVICE,
        model_D=model_D, loss_D=loss_D, metrics_D=metrics_D,
        geometric_transform=geometric_transform
    )

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

        scheduler.step()
        scheduler_D.step()