import os
import json

# torch
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

# model
from src.models.segmentation import get_SegmentationModel
# dataset
from src.dataset.albmentation_augmentation import get_transforms
from src.dataset.segmentation_dataset import SegmentationDataset
# training
from src.training.simple_segmentation_trainer import TestEpoch
from src.loss.get_loss import get_loss, get_metric
# utils
from src.utils.utils import get_pathes

if __name__ == '__main__':
    # init_training
    from config import simple_segmentation_settings as settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    valid_image_pathes = get_pathes(settings.valid_image_path)
    valid_label_pathes = get_pathes(settings.valid_label_path)

    valid_aug = get_transforms(settings.aug_settings_path, train=False)

    valid_dataset = SegmentationDataset(valid_image_pathes, valid_label_pathes,
                        class_num=settings.class_num, mean=settings.mean, std=settings.std,
                        label_type=settings.label_type, color_palette=settings.color_palette,
                        augmentation=valid_aug)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    model = get_SegmentationModel(settings.model, settings.encoder, activation=settings.activation, \
        depth=settings.depth, class_num=settings.class_num)
    model.load_state_dict(torch.load(settings.model_save_path))

    # loss function
    print('loss : ', settings.loss)
    loss = get_loss(settings.loss, settings.class_weight)

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [get_metric(name) for name in settings.metrics]

    # trainner
    save_path = os.path.join(settings.save_dir, 'pred_image')
    os.makedirs(save_path, exist_ok=True)

    test_epoch = TestEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        color_palette=settings.color_palette, label_type=settings.label_type,
        save_path=save_path, mean=settings.mean, std=settings.std,
    )

    # evaluate
    logs = test_epoch.run(valid_loader)
    json_path = os.path.join(settings.save_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(logs, f, indent=4)

    json_path = os.path.join(settings.save_dir, 'results_all.json')
    with open(json_path, 'w') as f:
        json.dump(test_epoch.all_logs, f, indent=4)
