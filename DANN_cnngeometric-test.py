import os
import json
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
from src.training.DANN_cnngeometric_trainer import TestEpoch
# utils
from src.utils import utils
import settings

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda" if use_cuda else "cpu"

    # datasets
    image_A_test_pathes = utils.get_pathes(settings.image_A_test_path)
    label_A_test_pathes = utils.get_pathes(settings.label_A_test_path)
    image_B_test_pathes = utils.get_pathes(settings.image_B_test_path)
    label_B_test_pathes = utils.get_pathes(settings.label_B_test_path)
    test_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    test_dataset = DAGeometricDataset(image_A_test_pathes, label_A_test_pathes,
                        image_B_test_pathes, label_B_test_pathes,
                        class_num=settings.class_num, color_palette=settings.color_palette,
                        label_type=settings.label_type, geometric=settings.geometric,
                        mean_A=settings.mean_A, std_A=settings.std_A,
                        mean_B=settings.mean_B, std_B=settings.std_B,
                        augmentation=test_aug)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, use_cuda=use_cuda, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)

    # geometric model definition
    model = CNNGeometric(encoder=settings.encoder, in_channels=settings.in_channels,
                        depth=settings.depth, weights=settings.weights,
                        output_dim=settings.cnn_output_dim,
                        input_size=settings.image_size,
                        freeze_encoder=settings.freeze_encoder)
    model.load_state_dict(torch.load(settings.model_save_path))

    if settings.pretrained_model_path is not None:
        model.load_state_dict(torch.load(settings.pretrained_model_path))

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss)

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [opt_util.get_metric(name) for name in settings.metrics]
    if settings.metrics_seg is not None:
        print('metrics_seg : ', settings.metrics_seg)
        metrics_seg = [opt_util.get_metric(name) for name in settings.metrics_seg]
    else:
        metrics_seg = None

    # trainner
    save_path = os.path.join(settings.save_dir, 'pred_image')
    os.makedirs(save_path, exist_ok=True)

    epoch_multi_A = TestEpoch(
        model=model, loss = loss, metrics = metrics, device=DEVICE,
        target_modality='A', multi_modality=True,
        geometric_transform=geometric_transform, save_path=save_path,
        mean_A=settings.mean_A, std_A=settings.std_A,
        mean_B=settings.mean_B, std_B=settings.std_B,
        segmentation_metrics=metrics_seg
    )
    epoch_multi_B = TestEpoch(
        model=model, loss = loss, metrics = metrics, device=DEVICE,
        target_modality='B', multi_modality=True,
        geometric_transform=geometric_transform, save_path=save_path,
        mean_A=settings.mean_A, std_A=settings.std_A,
        mean_B=settings.mean_B, std_B=settings.std_B
        segmentation_metrics=metrics_seg
    )

    epoch_mono_A = TestEpoch(
        model=model, loss = loss, metrics = metrics, device=DEVICE,
        target_modality='A', multi_modality=False,
        geometric_transform=geometric_transform, save_path=save_path,
        mean_A=settings.mean_A, std_A=settings.std_A
        segmentation_metrics=metrics_seg
    )
    epoch_mono_B = TestEpoch(
        model=model, loss = loss, metrics = metrics, device=DEVICE,
        target_modality='B', multi_modality=False,
        geometric_transform=geometric_transform, save_path=save_path,
        mean_B=settings.mean_B, std_B=settings.std_B
        segmentation_metrics=metrics_seg
    )

    # training
    logs_multi_A = epoch_multi_A.run(test_loader)
    logs_multi_B = epoch_multi_B.run(test_loader)
    logs_multi = (logs_multi_A, logs_multi_B)
    all_logs_multi = (epoch_multi_A.all_logs, epoch_multi_B.all_logs)

    logs_mono_A = epoch_mono_A.run(test_loader)
    logs_mono_B = epoch_mono_B.run(test_loader)
    logs_mono = (logs_mono_A, logs_mono_B)
    all_logs_mono = (epoch_mono_A.all_logs, epoch_mono_B.all_logs)

    for registration_type, logs, all_logs in zip(['multi', 'mono'], \
                                                [logs_multi, logs_mono], [all_logs_multi, all_logs_mono]):
        for log, modality, all_log in zip(logs, ['A', 'B'], all_logs):
            json_path = os.path.join(settings.save_dir, f'results_{registration_type}_tgt{modality}.json')
            with open(json_path, 'w') as f:
                json.dump(log, f, indent=4)

            json_path = os.path.join(settings.save_dir, f'results_{registration_type}_tgt{modality}_all.json')
            with open(json_path, 'w') as f:
                json.dump(all_log, f, indent=4)
