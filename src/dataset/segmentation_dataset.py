import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T

from .base_dataset import Dataset_base

class SegmentationDataset(Dataset_base):
    def __init__(self, image_pathes, label_pathes, \
            augmentation=None, class_num=10,\
            color_palette=None, label_type='binary_label',\
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__(
            augmentation=augmentation, class_num=class_num,\
            color_palette=color_palette, label_type=label_type, mean=mean, std=std
        )
        assert len(image_pathes)==len(label_pathes), print('The number of images and labels must be the same.')
        self.image_pathes = image_pathes
        self.label_pathes = label_pathes

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, index):
        image = self._load_image(self.image_pathes[index])
        label = self._load_label(self.label_pathes[index])

        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        image = self.to_tensor(image)
        label = self.to_tensor(label)

        if self.normalize is not None:
            image = self.normalize(image)

        sample = {'x':image, 'y':label}
        return sample

class DASegmentationDataset(Dataset_base):
    def __init__(self, image_pathes_A, label_pathes_A, \
            image_pathes_B, label_pathes_B, \
            augmentation=None, class_num=10,\
            color_palette=None, label_type='binary_label',\
            mean_A=[0.485, 0.456, 0.406], std_A=[0.229, 0.224, 0.225],\
            mean_B=[0.485, 0.456, 0.406], std_B=[0.229, 0.224, 0.225]):
        super().__init__(
            augmentation=augmentation, class_num=class_num,\
            color_palette=color_palette, label_type=label_type, mean=None, std=None
        )
        assert len(image_pathes_A)==len(label_pathes_A), print('The number of images and labels must be the same in image A.')
        assert len(image_pathes_B)==len(label_pathes_B), print('The number of images and labels must be the same in image B.')
        assert len(image_pathes_A)==len(image_pathes_B), print('The number of images in A and B should be the same.')
        self.image_pathes_A = image_pathes_A
        self.label_pathes_A = label_pathes_A
        self.image_pathes_B = image_pathes_B
        self.label_pathes_B = label_pathes_B

        self.normalize = False if None in [mean_A, mean_B, std_A, std_B] else True
        self.normalize_A = T.Normalize(mean=mean_A, std=std_A) if self.normalize else None
        self.normalize_B = T.Normalize(mean=mean_B, std=std_B) if self.normalize else None

    def __len__(self):
        return len(self.image_pathes_A)

    def __getitem__(self, index):
        image_A = self._load_image(self.image_pathes_A[index])
        label_A = self._load_label(self.label_pathes_A[index])

        image_B = self._load_image(self.image_pathes_B[index])
        label_B = self._load_label(self.label_pathes_B[index])

        if self.augmentation is not None:
            augmented_A = self.augmentation(image=image_A, mask=label_A)
            image_A, label_A = augmented_A['image'], augmented_A['mask']

            augmented_B = self.augmentation(image=image_B, mask=label_B)
            image_B, label_B = augmented_B['image'], augmented_B['mask']

        image_A = self.to_tensor(image_A)
        label_A = self.to_tensor(label_A)
        image_B = self.to_tensor(image_B)
        label_B = self.to_tensor(label_B)

        if self.normalize:
            image_A = self.normalize_A(image_A)
            image_B = self.normalize_B(image_B)

        sample = {'x_A':image_A, 'y_A':label_A, \
                  'x_B':image_B, 'y_B':label_B}
        return sample


