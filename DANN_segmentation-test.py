import os
import json
import warnings
warnings.simplefilter('ignore')

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
from src.training.DANN_segmentation_trainer import TestEpoch
# utils
from src.utils.utils import get_pathes, init_logging

if __name__ == '__main__':
    # init_training
    from config import DANN_segmentation_settings as settings
    train_logger, valid_logger = init_logging(settings.save_dir)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    image_A_valid_pathes = get_pathes(settings.image_A_valid_path)
    label_A_valid_pathes = get_pathes(settings.label_A_valid_path)
    image_B_valid_pathes = get_pathes(settings.image_B_valid_path)
    label_B_valid_pathes = get_pathes(settings.label_B_valid_path)

    valid_aug = get_transforms(settings.aug_settings_path, train=False)

    valid_dataset = DASegmentationDataset(
                        image_A_valid_pathes, label_A_valid_pathes,
                        image_B_valid_pathes, label_B_valid_pathes,
                        mean_A=settings.mean_A, std_A=settings.std_A,
                        mean_B=settings.mean_B, std_B=settings.std_B,
                        class_num=settings.class_num, augmentation=valid_aug,
                        label_type=settings.label_type, color_palette=settings.color_palette
                        )

    valid_loader = DataLoader(valid_dataset, batch_size=1, \
                            shuffle=False, num_workers=0)

    # segmentation_model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    model = get_SegmentationModel(settings.model, settings.encoder, activation=settings.activation,\
        encoder_weights=settings.weights, depth=settings.depth, class_num=settings.class_num)
    model.load_state_dict(torch.load(settings.model_save_path))

    # loss function
    print('loss_seg : ', settings.loss_seg)
    loss = get_loss(settings.loss_seg, settings.class_weight)

    # metric function
    print('metrics_test : ', settings.metrics_test)
    metrics = [get_metric(name, ignore_channels=[0]) for name in settings.metrics_test]

    # trainner
    save_path = os.path.join(settings.save_dir, 'pred_image')
    os.makedirs(save_path, exist_ok=True)

    test_epoch_A = TestEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        target_modality='A',
        label_type=settings.label_type, color_palette=settings.color_palette,
        save_path=save_path, mean=settings.mean_A, std=settings.std_A
    )
    test_epoch_B = TestEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        target_modality='B',
        label_type=settings.label_type, color_palette=settings.color_palette,
        save_path=save_path, mean=settings.mean_A, std=settings.std_A
    )

    # evaluate
    logs_A = test_epoch_A.run(valid_loader)
    logs_B = test_epoch_B.run(valid_loader)

    for logs, modality, test_epoch in zip([logs_A, logs_B], ['A', 'B'], [test_epoch_A, test_epoch_B]):
        json_path = os.path.join(settings.save_dir, 'results_{}.json'.format(modality))
        with open(json_path, 'w') as f:
            json.dump(logs, f, indent=4)

        json_path = os.path.join(settings.save_dir, 'results_{}_all.json'.format(modality))
        with open(json_path, 'w') as f:
            json.dump(test_epoch.all_logs, f, indent=4)

