import warnings
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader

# model
from src.models.cnngeometric import CNNGeometric
from src.transformation.core import GeometricTnf
# dataset
from src.dataset.albmentation_augmentation import get_transforms
from src.dataset.geometric_transform_dataset import GeometricDataset
# training
from src.training.cnngeometric_trainer import TestEpoch
from src.loss.get_loss import get_loss, get_metric
# utils
from src.utils.utils import get_pathes

if __name__ == '__main__':
    # init_training
    from config import cnngeometric_settings as settings

    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda" if use_cuda else "cpu"

    # datasets
    valid_images_pathes = get_pathes(settings.valid_image_path)
    valid_labels_pathes = get_pathes(settings.valid_label_path)

    valid_aug = get_transforms(settings.aug_settings_path, train=False)

    valid_dataset = GeometricDataset(image_pathes=valid_images_pathes, label_pathes=train_labels_pathes,\
                                    geometric=settings.geometric, augmentation=valid_aug,\
                                    class_num=settings.class_num, mean=settings.mean, std=settings.std,\
                                    random_t_tps=settings.random_t_tps,\
                                    label_type=settings.label_type, color_palette=settings.color_palette)

    valid_loader = DataLoader(valid_dataset, batch_size=1,
                            shuffle=False, num_workers=settings.num_workers)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, use_cuda=use_cuda, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)

    # model definition
    model = CNNGeometric(encoder=settings.encoder, in_channels=settings.in_channels,
                        depth=settings.depth, weights=settings.weights,
                        output_dim=settings.cnn_output_dim, input_size=settings.image_size, use_cuda=use_cuda,
                        freeze_FeatureExtractionModel=settings.freeze)

    model.load_state_dict(torch.load(pretrained_model_path))

    # loss function
    print('loss : ', settings.loss)
    loss = get_loss(settings.loss, settings.class_weight)

    # metric function
    metrics = [get_metric(name) for name in settings.metrics]

    # trainner
    test_epoch = TestEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        geometric_transform=geometric_transform
    )

    #########################訓練###################################################
    # evaluate
    test_logs = test_epoch.run(valid_loader)

    # logging
    test_msg = '[test] '.format(epoch)
    for metric_name in list(train_logs.keys()):
        train_msg += ' | {}:{:02.4f}'.format(metric_name, train_logs[metric_name])
        valid_msg += ' | {}:{:02.4f}'.format(metric_name, valid_logs[metric_name])
    train_logger.info(train_msg)
    valid_logger.info(valid_msg)
