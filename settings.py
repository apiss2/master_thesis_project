import os
import shutil
import argparse
import configparser

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
save_root = config.get('save_root')
task_name = config.get('task_name')
save_dir = os.path.join(save_root, task_name, experiment_name)
os.makedirs(save_dir, exist_ok=True)

############ settings ############

# util
geometric = config.get('geometric')
cnn_output_dim = utils.geometric_choice(geometric)
model_save_path = os.path.join(save_dir, config.get('model_save_name'))

# training
monitor_metric = config.get('monitor_metric')
epochs = config.getint('epochs')

modelupdate_freq = config.getint('modelupdate_freq')
discupdate_freq = config.getint('discupdate_freq')

### image path ###
dataset_base_path = config.get('dataset_base_path')
assert dataset_base_path is not None, 'No dataset_base_path'

# train
train_image_path = utils.get_path(dataset_base_path, config.get('train_image_path'))
train_label_path = utils.get_path(dataset_base_path, config.get('train_label_path'))

image_A_train_path = utils.get_path(dataset_base_path, config.get('image_A_train'))
label_A_train_path = utils.get_path(dataset_base_path, config.get('label_A_train'))
image_B_train_path = utils.get_path(dataset_base_path, config.get('image_B_train'))
label_B_train_path = utils.get_path(dataset_base_path, config.get('label_B_train'))

# valid
valid_image_path = utils.get_path(dataset_base_path, config.get('valid_image_path'))
valid_label_path = utils.get_path(dataset_base_path, config.get('valid_label_path'))

image_A_valid_path = utils.get_path(dataset_base_path, config.get('image_A_valid'))
label_A_valid_path = utils.get_path(dataset_base_path, config.get('label_A_valid'))
image_B_valid_path = utils.get_path(dataset_base_path, config.get('image_B_valid'))
label_B_valid_path = utils.get_path(dataset_base_path, config.get('label_B_valid'))

# test
test_image_path = utils.get_path(dataset_base_path, config.get('test_image_path'))
test_label_path = utils.get_path(dataset_base_path, config.get('test_label_path'))

image_A_test_path = utils.get_path(dataset_base_path, config.get('image_A_test'))
label_A_test_path = utils.get_path(dataset_base_path, config.get('label_A_test'))
image_B_test_path = utils.get_path(dataset_base_path, config.get('image_B_test'))
label_B_test_path = utils.get_path(dataset_base_path, config.get('label_B_test'))

# dataset
mean, std = utils.get_normParam(config.get('mean'), config.get('std'))
mean_A, std_A = utils.get_normParam(config.get('MR_mean'), config.get('MR_std'))
mean_B, std_B = utils.get_normParam(config.get('CT_mean'), config.get('CT_std'))

class_num = config.getint('class_num')
label_type = config.get('label_type')
color_palette_path = utils.get_path(dataset_base_path, config.get('color_palette'))
color_palette = utils.get_colorPalette(color_palette_path)

aug_settings_path = config.get('augmentation_setting_json')
image_size = aug.get_transformedImageSize(aug_settings_path)

random_t_tps = config.getfloat('random_t_tps') # geometric only

# dataloader
batch_size = config.getint('batch_size')
num_workers = config.getint('num_workers')

# model
model = config.get('model')
encoder = config.get('encoder')
in_channels = config.getint('in_channels')
depth = config.getint('depth')
weights = config.get('weights')
activation = config.get('activation')

freeze_encoder = config.getboolean('freeze_encoder')
pretrained_model_path = config.get('pretrained_model_path')
padding_mode = config.get('padding_mode')

if config.get('discriminator_channels') is not None:
    discriminator_channels = list(map(lambda x: int(x), config.get('discriminator_channels').split('-')))

# loss func
loss = config.get('loss')
loss_D = config.get('loss_D')
class_weight_path = utils.get_path(dataset_base_path, config.get('class_weight'))
class_weight = utils.get_classWeight(class_weight_path)

# metrics
metrics = config.get('metrics').split('-')
metrics_D = config.get('metrics_D').split('-') if config.get('metrics_D') is not None else None
metrics_seg = config.get('metrics_seg').split('-') if config.get('metrics_seg') is not None else None
metrics_test = config.get('metrics_test').split('-') if config.get('metrics_test') is not None else None

# optimizer
optimizer = config.get('optimizer')
lr = config.getfloat('lr')
optimizer_D = config.get('optimizer_D')
lr_D = config.getfloat('lr_D')

# scheduler
scheduler_type = config.get('scheduler_type')
if config.get('decay_schedule') is not None:
    decay_schedule = list(map(lambda x: int(x), config.get('decay_schedule').split('-')))
gamma = config.getfloat('gamma')
warmupepochs = config.getint('warmupepochs')
eta_min = config.getfloat('eta_min')

### save settings ###
shutil.copy(args.config, os.path.join(save_dir, os.path.basename(args.config)))
if color_palette_path is not None:
    shutil.copy(color_palette_path, os.path.join(save_dir, os.path.basename(color_palette_path)))
if class_weight_path is not None:
    shutil.copy(class_weight_path, os.path.join(save_dir, os.path.basename(class_weight_path)))
if aug_settings_path is not None:
    shutil.copy(aug_settings_path, os.path.join(save_dir, os.path.basename(aug_settings_path)))

