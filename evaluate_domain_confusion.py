import os
import warnings
import numpy as np
from tqdm import tqdm
warnings.simplefilter('ignore')

# torch
import torch
from torch.utils.data import DataLoader

# model
from src.models.cnngeometric import CNNGeometric, CNNGeometric_MCDDA, CNNGeometricDecoder
from src.models.cnngeometric_base import FeatureExtraction
# dataset
from src.dataset.geometric_transform_dataset import DAGeometricDataset
from src.dataset import albmentation_augmentation as aug
# utils
from src.utils import utils
import settings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    # model definition
    print('model : ', settings.model)
    print('encoder : ', settings.encoder)
    if 'MCDDA' in settings.pretrained_model_path:
        if 'pretrain' in settings.pretrained_model_path:
            encoder = FeatureExtraction(name=settings.encoder, depth=settings.depth, weights=settings.weights)
            with torch.no_grad():
                sample = encoder.forward(torch.rand((2, 3, settings.image_size, settings.image_size)))[-1]
            decoder = CNNGeometricDecoder(sample, output_dim=settings.cnn_output_dim)
            model = CNNGeometric_MCDDA(encoder, decoder)
            model.load_state_dict(torch.load(settings.pretrained_model_path))
            model = model.encoder
        else:
            model = FeatureExtraction(name=settings.encoder, depth=settings.depth, weights=settings.weights)
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
        model = model.encoder

    model.to(DEVICE).eval()
    MR_features = list()
    CT_features = list()
    for batch in tqdm(test_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(DEVICE)
        x_A, x_B = batch['x_A'], batch['x_B']
        with torch.no_grad():
            feature_A = model(x_A)[-1][0]
            feature_B = model(x_B)[-1][0]
        feature_A = feature_A.cpu().detach().numpy().max(axis=(1, 2))
        feature_B = feature_B.cpu().detach().numpy().max(axis=(1, 2))
        MR_features.append(feature_A)
        CT_features.append(feature_B)

    all_features = MR_features.copy() + CT_features.copy()
    all_features = np.array(all_features)

    print('prepare PCA model...')
    model = PCA(2)
    model = model.fit(all_features)

    plt.figure(figsize=(8.0, 6.0), dpi=300)
    MR = model.transform(np.array(MR_features))
    CT = model.transform(np.array(CT_features))
    plt.scatter(MR[:,0],MR[:,1], label='MR')
    plt.scatter(CT[:, 0], CT[:, 1], label='CT')
    plt.savefig('./PCA.png', bbox_inches='tight')
    plt.show()

    print('prepare tSNE model...')
    model = TSNE(n_components=2)
    features_2d = model.fit_transform(all_features)
    half = int(features_2d.shape[0]/2)

    plt.figure(figsize=(8.0, 6.0), dpi=300)
    MR = features_2d[:half]
    CT = features_2d[half:]
    plt.scatter(MR[:,0],MR[:,1], label='MR')
    plt.scatter(CT[:30,0],CT[:30,1], label='CT')
    plt.savefig('./tSNE.png', bbox_inches='tight')
    plt.show()