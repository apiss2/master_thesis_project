import os
import shutil
import argparse
import configparser

import sys
sys.path.append('../')

from src.utils import utils
from src.dataset import albmentation_augmentation as aug

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
modelupdate_freq = int(config.get('modelupdate_freq'))

# dataset
image_A_train_path = config.get('image_A_train')
label_A_train_path = config.get('label_A_train')
image_B_train_path = config.get('image_B_train')
label_B_train_path = config.get('label_B_train')

image_A_valid_path = config.get('image_A_valid')
label_A_valid_path = config.get('label_A_valid')
image_B_valid_path = config.get('image_B_valid')
label_B_valid_path = config.get('label_B_valid')

image_A_test_path = config.get('image_A_test')
label_A_test_path = config.get('label_A_test')
image_B_test_path = config.get('image_B_test')
label_B_test_path = config.get('label_B_test')

mean_A, std_A = utils.get_normParam(config.get('MR_mean'), config.get('MR_std'))
mean_B, std_B = utils.get_normParam(config.get('CT_mean'), config.get('CT_std'))
class_num = int(config.get('class_num'))
label_type = config.get('label_type')
color_palette = utils.get_colorPalette(config.get('color_palette'))

aug_settings_path = config.get('augmentation_setting_json')
image_size = aug.get_transformedImageSize(aug_settings_path)

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

pretrained_model_path = config.get('pretrained_model_path')
padding_mode = config.get('padding_mode')

discriminator_channels = list(map(lambda x: int(x), config.get('discriminator_channels').split('-')))

# loss func
loss_seg = config.get('loss_seg')
loss_DA = config.get('loss_DA')
class_weight = utils.get_classWeight(config.get('class_weight'))

if config.get('class_weight') is not None:
    shutil.copy(config.get('class_weight'), os.path.join(save_dir, os.path.basename(config.get('class_weight'))))

# metrics
metrics = config.get('metrics').split('-')
metrics_D = config.get('metrics_D').split('-')
metrics_test = config.get('metrics_test').split('-')

# optimizer
optimizer = config.get('optimizer')
lr = float(config.get('lr'))

# scheduler
decay_schedule = list(map(lambda x: int(x), config.get('decay_schedule').split('-')))
gamma = float(config.get('gamma'))


