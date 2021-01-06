import os
import shutil
import argparse
import configparser

import sys
sys.path.append('../')

from src.utils.utils import get_normParam, get_colorPalette, get_classWeight

# load config file
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='path to config file')
parser.add_argument('--config_type', type=str, default='DEFAULT', help='path to config file')
args = parser.parse_args()
assert os.path.exists(args.config), 'no such config file'

config_ini = configparser.ConfigParser()
config_ini.read(args.config, encoding='utf-8')
experiment_name = args.config_type
config = config_ini[experiment_name]

# make dir
task_name = config.get('task_name')
save_dir = os.path.join('D:/result/', task_name, experiment_name)
os.makedirs(save_dir, exist_ok=True)

# save config
shutil.copy(args.config, os.path.join(save_dir, os.path.basename(args.config)))

############ settings ############

# util
model_save_path = os.path.join(save_dir, config.get('model_save_name'))

# training
monitor_metric = config.get('monitor_metric')
epochs = int(config.get('epochs'))

# dataset
train_image_path = config.get('train_image_path')
train_label_path = config.get('train_label_path')
valid_image_path = config.get('valid_image_path')
valid_label_path = config.get('valid_label_path')

mean, std = get_normParam(config.get('mean'), config.get('std'))
class_num = int(config.get('class_num'))
label_type = config.get('label_type')
color_palette = get_colorPalette(config.get('color_palette'))

aug_settings_path = config.get('augmentation_setting_json')

# save settings
if config.get('color_palette') is not None:
    shutil.copy(config.get('color_palette'), os.path.join(save_dir, os.path.basename(config.get('color_palette'))))
shutil.copy(aug_settings_path, os.path.join(save_dir, os.path.basename(aug_settings_path)))

# dataloader
batch_size = int(config.get('batch_size'))
num_workers = int(config.get('num_workers'))

# model
model = config.get('model')
encoder = config.get('encoder')
depth = int(config.get('depth'))
weights = config.get('weights')
activation = config.get('activation')

freeze = bool(config.get('freeze_FeatureExtractionModel'))
pretrained_model_path = config.get('pretrained_model_path')
padding_mode = config.get('padding_mode')

# loss func
loss = config.get('loss')
class_weight = get_classWeight(config.get('class_weight'))

if config.get('class_weight') is not None:
    shutil.copy(config.get('class_weight'), os.path.join(save_dir, os.path.basename(config.get('class_weight'))))

# metrics
metrics = config.get('metrics').split('-')

# optimizer
optimizer = config.get('optimizer')
lr = float(config.get('lr'))

# scheduler
decay_schedule = list(map(lambda x: int(x), config.get('decay_schedule').split('-')))
gamma = float(config.get('gamma'))


