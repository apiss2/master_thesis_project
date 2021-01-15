import os
import glob
import json
import torch
import logging
import argparse
import numpy as np
import configparser
import matplotlib.pyplot as plt

def init_logging(save_dir):
    train_logger = logging.getLogger('train')
    valid_logger = logging.getLogger('valid')
    train_logger.setLevel(logging.INFO)
    valid_logger.setLevel(logging.INFO)
    train_handler = logging.FileHandler(os.path.join(save_dir,'train.log'))
    valid_handler = logging.FileHandler(os.path.join(save_dir,'valid.log'))
    train_logger.addHandler(train_handler)
    valid_logger.addHandler(valid_handler)
    return train_logger, valid_logger

def get_path(root_path, path):
    if path is None:
        return None
    return os.path.join(root_path, path)

def get_pathes(path, extension=None):
    if path is None:
        return None
    name = '*'
    if extension is not None:
        assert type(extension)==str, 'Please specify the extension as str'
        name += extension
    path = os.path.join(path, name)
    pathes = glob.glob(path)
    assert len(pathes)!=0, 'No such file. path:{}'.format(path)
    return pathes

def geometric_choice(geometric):
    if geometric is None:
        return None
    if geometric=='affine':
        cnn_output_dim = 6
    elif geometric=='hom':
        cnn_output_dim = 8
    elif geometric=='tps':
        cnn_output_dim = 18
    return cnn_output_dim

def get_colorPalette(path):
    if path is None:
        return None
    with open(path, 'r') as f:
        l = [[int(i) for i in s.strip().split(',')] for s in f.readlines()]
    return l

def get_normParam(mean, std):
    if mean == None or std == None:
        return None, None
    mean = list(map(lambda x: float(x), mean.split('-')))
    std  = list(map(lambda x: float(x), std.split('-' )))
    return mean, std

def get_classWeight(path):
    if path is None:
        return None
    with open(path, 'r') as f:
        l = [float(s) for s in f.readlines()]
    return l

def load_config(path):
    with open(path, 'rb') as f:
        config = json.load(f)
    return config

def expand_dim(tensor, dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))

def is_best_score(valid_logs, monitor_metric, best_score, maximize=True):
    if maximize:
        if valid_logs[monitor_metric] > best_score:
            return True
        else:
            return False
    else:
        if valid_logs[monitor_metric] < best_score:
            return True
        else:
            return False

def onehot2color(pred_image, color_palette):
    assert pred_image.shape[2]==len(color_palette), \
        'The number of classes of color_palette and image_channel do not match.'
    h, w = pred_image.shape[:2]
    label = np.zeros((h, w, 3))
    for i, colors in enumerate(color_palette):
        tmp = np.where(pred_image[...,i]==1, 1, 0)
        for channel, color in enumerate(colors):
            label[...,channel] += tmp*color/255
    return label.clip(0,1)