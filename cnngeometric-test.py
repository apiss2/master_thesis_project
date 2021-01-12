import os
import json
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
from src.utils import opt_util
from src.training.cnngeometric_trainer import TestEpoch
# utils
from src.utils import utils

if __name__ == '__main__':
    # init_training
    from config import cnngeometric_settings as settings

    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda" if use_cuda else "cpu"

    # datasets
    test_images_pathes = utils.get_pathes(settings.test_image_path)
    test_labels_pathes = utils.get_pathes(settings.test_label_path)

    test_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    test_dataset = GeometricDataset(image_pathes=test_images_pathes, label_pathes=test_labels_pathes,\
                                    geometric=settings.geometric, augmentation=test_aug,\
                                    class_num=settings.class_num, mean=settings.mean, std=settings.std,\
                                    random_t_tps=settings.random_t_tps,\
                                    label_type=settings.label_type, color_palette=settings.color_palette)

    test_loader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=settings.num_workers)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, use_cuda=use_cuda, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)

    # model definition
    model = CNNGeometric(encoder=settings.encoder, in_channels=settings.in_channels,
                        depth=settings.depth, weights=settings.weights,
                        output_dim=settings.cnn_output_dim, input_size=settings.image_size,
                        freeze_FeatureExtractionModel=settings.freeze)
    model.load_state_dict(torch.load(settings.model_save_path))

    if settings.pretrained_model_path is not None:
        model.load_state_dict(torch.load(settings.pretrained_model_path))

    # loss function
    print('loss : ', settings.loss)
    loss = opt_util.get_loss(settings.loss, settings.class_weight)

    # metric function
    print('metrics : ', settings.metrics)
    metrics = [opt_util.get_metric(name) for name in settings.metrics]

    # trainner
    save_path = os.path.join(settings.save_dir, 'pred_image')
    os.makedirs(save_path, exist_ok=True)

    test_epoch = TestEpoch(
        model=model, loss=loss, metrics=metrics, device=DEVICE,
        geometric_transform=geometric_transform,
        save_path=save_path, mean=settings.mean, std=settings.std
    )

    # evaluate
    logs = test_epoch.run(test_loader)

    json_path = os.path.join(settings.save_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(logs, f, indent=4)

    json_path = os.path.join(settings.save_dir, 'results_all.json')
    with open(json_path, 'w') as f:
        json.dump(test_epoch.all_logs, f, indent=4)