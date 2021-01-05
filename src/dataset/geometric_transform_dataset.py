import torch
import numpy as np

from ..transformation.core import GeometricTnf
from .base_dataset import Dataset_base
from .segmentation_dataset import DASegmentationDataset

def generate_theta(geometric, random_t_tps):
    if geometric=='affine':
        rot_angle = (np.random.rand(1)-0.5)*2*np.pi/12 # between -np.pi/12 and np.pi/12
        sh_angle = (np.random.rand(1)-0.5)*2*np.pi/6 # between -np.pi/6 and np.pi/6
        lambda_1 = 1+(2*np.random.rand(1)-1)*0.25 # between 0.75 and 1.25
        lambda_2 = 1+(2*np.random.rand(1)-1)*0.25 # between 0.75 and 1.25
        tx=(2*np.random.rand(1)-1)*0.25  # between -0.25 and 0.25
        ty=(2*np.random.rand(1)-1)*0.25

        R_sh = np.array([[np.cos(sh_angle[0]),-np.sin(sh_angle[0])],
                            [np.sin(sh_angle[0]),np.cos(sh_angle[0])]])
        R_alpha = np.array([[np.cos(rot_angle[0]),-np.sin(rot_angle[0])],
                            [np.sin(rot_angle[0]),np.cos(rot_angle[0])]])

        D=np.diag([lambda_1[0],lambda_2[0]])
        A = R_alpha @ R_sh.transpose() @ D @ R_sh
        theta = np.array([A[0,0],A[0,1],tx,A[1,0],A[1,1],ty])
    if geometric=='hom':
        theta = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
        theta = theta+(np.random.rand(8)-0.5)*2*random_t_tps
    if geometric=='tps':
        theta = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
        theta = theta+(np.random.rand(18)-0.5)*2*random_t_tps
    return theta

class GeometricDataset(Dataset_base):
    def __init__(self, image_pathes:list, label_pathes:list=None,
                 geometric:str='affine', augmentation=None,
                 random_t_tps:float=0.4, class_num:int=10,\
                 color_palette:list=None, label_type:str='binary_label',\
                 mean:list=[0.485, 0.456, 0.406], std:list=[0.229, 0.224, 0.225]):
        super().__init__(
            augmentation=augmentation, class_num=class_num,\
            color_palette=color_palette, label_type=label_type, mean=mean, std=std
        )

        self.image_pathes = image_pathes
        self.label_pathes = label_pathes
        self.geometric = geometric
        self.random_t_tps = random_t_tps

        if self.label_pathes is not None:
            assert len(label_pathes)==len(image_pathes), 'The number of images and labels should be the same'
        if self.color_palette is not None:
            assert class_num==len(self.color_palette), \
                'The number of classes must match the number of color palettes.'

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        # load image
        image = self._load_image(self.image_pathes[idx])
        label = self._load_label(self.label_pathes[idx]) if self.label_pathes is not None else None

        # augmentation
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        image = self.to_tensor(image)
        label = self.to_tensor(label) if self.label_pathes is not None else None

        # normalization
        if self.normalize is not None:
            image = self.normalize(image)

        # generate theta
        theta = generate_theta(self.geometric, self.random_t_tps)
        theta = torch.Tensor(theta.astype(np.float32))

        sample = {'image':image, 'theta':theta}
        if label is not None:
            sample.update({'label':label})
        return sample

class DAGeometricDataset(DASegmentationDataset):
    def __init__(self, image_pathes_A:list, label_pathes_A:list, \
            image_pathes_B:list, label_pathes_B:list, \
            geometric:str='affine', random_t_tps:float=0.4, \
            augmentation=None, class_num:int=10,\
            color_palette:list=None, label_type:str='binary_label',\
            mean_A:list=[0.485, 0.456, 0.406], std_A:list=[0.229, 0.224, 0.225],\
            mean_B:list=[0.485, 0.456, 0.406], std_B:list=[0.229, 0.224, 0.225]):
        super().__init__(
            image_pathes_A=image_pathes_A, label_pathes_A=label_pathes_A,
            image_pathes_B=image_pathes_B, label_pathes_B=label_pathes_B,
            augmentation=augmentation, class_num=class_num,\
            color_palette=color_palette, label_type=label_type,\
            mean_A=mean_A, std_A=std_A, mean_B=mean_B, std_B=std_B
        )
        self.geometric = geometric
        self.random_t_tps = random_t_tps

    def __getitem__(self, idx):
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

        # generate theta
        theta = generate_theta(self.geometric, self.random_t_tps)
        theta = torch.Tensor(theta.astype(np.float32))

        sample = {'x_A':image_A, 'y_A':label_A, \
                  'x_B':image_B, 'y_B':label_B, 'theta':theta}
        return sample