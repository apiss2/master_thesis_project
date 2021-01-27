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
from src.models.cnngeometric import CNNGeometric, CNNGeometricDecoder, CNNGeometric_MCDDA
from src.models.cnngeometric_base import FeatureExtraction
from src.transformation.core import GeometricTnf
# dataset
from src.dataset.geometric_transform_dataset import DAGeometricDataset, GeometricDataset
from src.dataset import albmentation_augmentation as aug
# training
from src.utils import opt_util
from src.training.DAtester import DAregistrationTester
# utils
from src.utils import utils
import settings

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    kargs = {'device': DEVICE}

    # datasets
    test_aug = aug.get_transforms(settings.aug_settings_path, train=False)

    image_A_test_pathes = utils.get_pathes(settings.image_A_test_path)
    label_A_test_pathes = utils.get_pathes(settings.label_A_test_path)
    image_B_test_pathes = utils.get_pathes(settings.image_B_test_path)
    label_B_test_pathes = utils.get_pathes(settings.label_B_test_path)

    test_dataset = DAGeometricDataset(image_A_test_pathes, label_A_test_pathes,
                        image_B_test_pathes, label_B_test_pathes,
                        class_num=settings.class_num, color_palette=settings.color_palette,
                        label_type=settings.label_type, geometric=settings.geometric,
                        mean_A=settings.mean_A, std_A=settings.std_A,
                        mean_B=settings.mean_B, std_B=settings.std_B,
                        augmentation=test_aug)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    geometric_transform = GeometricTnf(geometric_model=settings.geometric, device=DEVICE, \
                                        padding_mode=settings.padding_mode, size=settings.image_size)
    kargs.update({'geometric_transform': geometric_transform})

    # model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    if 'MCDDA' in settings.experiment_name:
        encoder = FeatureExtraction(name=settings.encoder, depth=settings.depth, weights=settings.weights)
        #encoder.load_state_dict(torch.load(settings.pretrained_encoder_path))
        with torch.no_grad():
            sample = encoder.forward(torch.rand((2, 3, settings.image_size, settings.image_size)))
        decoder = CNNGeometricDecoder(sample, output_dim=settings.cnn_output_dim)
        #decoder.load_state_dict(torch.load(settings.pretrained_decoder_path))
        model = CNNGeometric_MCDDA(encoder, decoder)
        model.load_state_dict(torch.load(settings.pretrained_model_path))
    else:
        model = CNNGeometric(encoder=settings.encoder,
                        in_channels=settings.in_channels,
                        depth=settings.depth,
                        weights=settings.weights,
                        output_dim=settings.cnn_output_dim,
                        input_size=settings.image_size,
                        freeze_encoder=settings.freeze_encoder,
                        make_dilated=settings.make_dilated).to(DEVICE)
        model.load_state_dict(torch.load(settings.pretrained_model_path))
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
        metrics_seg = [opt_util.get_metric(name, ignore_channels=[0]) for name in settings.metrics_seg]
        kargs.update({'metrics_seg': metrics_seg})
    else:
        metrics_seg = None

    # trainner
    save_path = os.path.join(settings.save_dir, 'pred_image')
    os.makedirs(save_path, exist_ok=True)
    kargs.update({
        'save_path':save_path,
        'mean_A': settings.mean_A,
        'std_A':settings.std_A,
        'mean_B':settings.mean_B,
        'std_B': settings.std_B,
        'label_type': settings.label_type,
        'color_palette': settings.color_palette
        })

    tester_multi_A = DAregistrationTester(
        target_modality='A', multi_modality=True, **kargs
    )
    tester_multi_B = DAregistrationTester(
        target_modality='B', multi_modality=True, **kargs
    )

    tester_mono_A = DAregistrationTester(
        target_modality='A', multi_modality=False, **kargs
    )
    tester_mono_B = DAregistrationTester(
        target_modality='B', multi_modality=False, **kargs
    )

    # training
    logs_multi_A = tester_multi_A.run(test_loader)
    logs_multi_B = tester_multi_B.run(test_loader)
    logs_multi = (logs_multi_A, logs_multi_B)
    all_logs_multi = (tester_multi_A.all_logs, tester_multi_B.all_logs)

    logs_mono_A = tester_mono_A.run(test_loader)
    logs_mono_B = tester_mono_B.run(test_loader)
    logs_mono = (logs_mono_A, logs_mono_B)
    all_logs_mono = (tester_mono_A.all_logs, tester_mono_B.all_logs)

    for registration_type, logs, all_logs in zip(['multi', 'mono'], \
                                                [logs_multi, logs_mono], [all_logs_multi, all_logs_mono]):
        for log, modality, all_log in zip(logs, ['A', 'B'], all_logs):
            json_path = os.path.join(settings.save_dir, f'results_{registration_type}_tgt{modality}.json')
            with open(json_path, 'w') as f:
                json.dump(log, f, indent=4)

            json_path = os.path.join(settings.save_dir, f'results_{registration_type}_tgt{modality}_all.json')
            with open(json_path, 'w') as f:
                json.dump(all_log, f, indent=4)
