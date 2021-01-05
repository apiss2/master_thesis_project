import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T

class Dataset_base(Dataset):
    def __init__(self, augmentation=None, class_num:int=1,\
            color_palette:list=None, label_type:str='binary_label',\
            mean:list=[0.485, 0.456, 0.406], std:list=[0.229, 0.224, 0.225]):
        self.class_num = class_num
        self.label_type = label_type
        self.color_palette = color_palette
        self.augmentation = augmentation
        self.to_tensor = T.ToTensor()
        self.normalize = None if None in [mean, std] else T.Normalize(mean=mean, std=std)

        if self.color_palette is not None:
            assert class_num==len(self.color_palette), \
                'The number of classes must match the number of color palettes.'

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        return NotImplementedError

    def _load_image(self, path):
        image = cv2.imread(path, 1).astype('uint8')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_label(self, path):
        if self.label_type == 'number_label':
            label = self._load_LabelNumberImage(path)
        elif self.label_type == 'color_label':
            label = self._load_ColorLabelImage(path)
        elif self.label_type == 'binary_label':
            label = self._load_BinaryLabel(path)
        else:
            raise ValueError("Unexpected label type")
        return label

    def _load_BinaryLabel(self, path):
        return cv2.imread(path, 0).astype('uint8')

    def _load_LabelNumberImage(self, path):
        label_raw = cv2.imread(path, 0).astype('uint8')
        h, w = label_raw.shape
        label = np.zeros((h,w,self.class_num))
        for i in range(self.class_num):
            label[..., i] = (label_raw==i)
        return label

    def _load_ColorLabelImage(self, path):
        label_raw = cv2.imread(path, 1).astype('uint8')
        label_raw = cv2.cvtColor(label_raw, cv2.COLOR_BGR2RGB)
        h,w = label_raw.shape[:2]
        label = np.zeros((h, w, len(self.color_palette)))
        for i, color in enumerate(self.color_palette):
            tmp = np.where((label_raw[...,0] == color[0])&
                           (label_raw[...,1] == color[1])&
                           (label_raw[...,2] == color[2]), 1, 0)
            label[..., i] = tmp
        return label