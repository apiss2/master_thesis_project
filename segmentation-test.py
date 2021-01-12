import os
import json

# torch
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

# model
from src.models import segmentation as seg
# dataset
from src.dataset import albmentation_augmentation as aug
from src.dataset.segmentation_dataset import SegmentationDataset
# training
from src.utils import opt_util
from src.training.segmentation_trainer import TestEpoch
# utils
from src.utils import utils
import settings

if __name__ == '__main__':
    # init_training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    test_image_pathes = utils.get_pathes(settings.test_image_path)
    test_label_pathes = utils.get_pathes(settings.test_label_path)

    test_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    test_dataset = SegmentationDataset(test_image_pathes, test_label_pathes,
                        class_num=settings.class_num, mean=settings.mean, std=settings.std,
                        label_type=settings.label_type, color_palette=settings.color_palette,
                        augmentation=test_aug)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    model = seg.get_SegmentationModel(settings.model, settings.encoder, activation=settings.activation, \
        depth=settings.depth, class_num=settings.class_num)
    model.load_state_dict(torch.load(settings.model_save_path))

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, settings.class_weight)

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [opt_util.get_metric(name, ignore_channels=[0]) for name in settings.metrics]

    # trainner
    save_path = os.path.join(settings.save_dir, 'pred_image')
    os.makedirs(save_path, exist_ok=True)

    test_epoch = TestEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        color_palette=settings.color_palette, label_type=settings.label_type,
        save_path=save_path, mean=settings.mean, std=settings.std,
    )

    # evaluate
    logs = test_epoch.run(test_loader)

    json_path = os.path.join(settings.save_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(logs, f, indent=4)

    json_path = os.path.join(settings.save_dir, 'results_all.json')
    with open(json_path, 'w') as f:
        json.dump(test_epoch.all_logs, f, indent=4)
