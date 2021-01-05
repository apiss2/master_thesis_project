import json
import albumentations as A
# https://qiita.com/kurilab/items/b69e1be8d0224ae139ad#huesaturationvalue

def get_transforms(settings, train:bool=True):
    if type(settings) == str:
        with open(settings, 'r') as f:
            settings = json.load(f)
    preprocess_funcs = get_augmentation(settings['preprocess'])
    after_funcs = get_augmentation(settings['after'])
    if train:
        transform_list = get_augmentation(settings['augmentation'])
        transforms = preprocess_funcs + transform_list + after_funcs
    else:
        transforms = preprocess_funcs + after_funcs

    return A.Compose(transforms)

def get_transformedImageSize(settings):
    if type(settings) == str:
        with open(settings, 'r') as f:
            settings = json.load(f)
    return settings['output_size']

def get_augmentation(settings:dict):
    transforms = list()
    for d in settings:
        name = d['transform']
        params = d['params']
        if name == 'resize':
            transforms.append(get_resize(params))
        elif name == 'resizecrop':
            transforms.append(get_randomCropResize(params))
        elif name == 'crop':
            transforms.append(get_crop(params))
        elif name == 'flip':
            transforms.append(get_flip(params))
        elif name == 'affine':
            transforms.append(get_affine(params))
        elif name == 'distortion':
            transforms.append(get_distortion(params))
        elif name == 'noise':
            transforms.append(get_noise(params))
        elif name == 'brightnesscontrast':
            transforms.append(get_BrightnessContrast(params))
        elif name == 'dropout':
            transforms.append(get_dropout(params))
        else:
            assert False, 'Invalid augmentation name: {}'.format(name)
    return transforms

def get_randomCropResize(params:dict):
    height = params['height']
    width  = params['width']
    scale = params['scale']
    ratio  = params['ratio']
    trans = A.RandomResizedCrop(height=height, width=width, \
        scale=scale, ratio=ratio, always_apply=True)
    return trans

def get_resize(params:dict):
    height = params['height']
    width  = params['width']
    trans = A.Resize(height=height, width=width, always_apply=True)
    return trans

def get_BrightnessContrast(params:dict):
    bright = params['brightness_limit']
    contrast  = params['contrast_limit']
    probability = params['probability']
    trans = A.RandomBrightnessContrast(brightness_limit=bright,
        contrast_limit=contrast, p=probability)
    return trans

def get_affine(params:dict):
    scale = params['scale']
    shift = params['shift']
    rotate = params['rotate']
    border_mode = params['border_mode']
    probability = params['probability']
    trans = A.ShiftScaleRotate(shift_limit=shift, scale_limit=scale, \
        rotate_limit=rotate, border_mode=border_mode, p=probability)
    return trans

def get_flip(params:dict):
    flip_type = params['type'].lower()
    probability = params['probability']
    if flip_type=='holizontal':
        trans = A.HorizontalFlip(p=probability)
    elif flip_type=='vertical':
        trans = A.VerticalFlip(p=probability)
    elif flip_type=='flip':
        trans = A.VerticalFlip(p=probability)
    else:
        ValueError('Unexpected flip type')
    return trans

def get_dropout(params:dict):
    probability = params['probability']
    max_holes = params['max_holes']
    min_holes = params['min_holes']
    max_height = params['max_height']
    min_height = params['min_height']
    max_width = params['max_width']
    min_width = params['min_width']
    fill_value = params['fill_value']
    trans = A.CoarseDropout(p=probability, fill_value=fill_value,
        max_holes=max_holes, min_holes=min_holes,
        max_height=max_height, min_height=min_height,
        max_width=max_width, min_width=min_width)
    return trans

def get_colorOperator(params:dict):
    # add more transforms
    color_type = params['type'].lower()
    probability = params['probability']
    if color_type=='huesaturation':
        hue_shift_limit = params['hue_shift_limit']
        sat_shift_limit = params['sat_shift_limit']
        val_shift_limit = params['val_shift_limit']
        trans = A.HueSaturationValue(p=probability,
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit)
    elif color_type=='rgbshift':
        r = params['r']
        g = params['g']
        b = params['b']
        trans = A.RGBShift(p=probability, r_shift_limit=r,
            g_shift_limit=g, b_shift_limit=b)
    else:
        ValueError('Unexpected flip type')
    return trans

def get_distortion(params:dict):
    distortion_type = params['type'].lower()
    probability = params['probability']
    if distortion_type=='optical':
        distort_limit = params['distort_limit']
        shift_limit = params['shift_limit']
        trans = A.OpticalDistortion(p=probability,
            distort_limit=distort_limit, shift_limit=shift_limit)
    elif distortion_type=='grid':
        num_steps = params['num_steps']
        border_mode = params['border_mode']
        trans = A.GridDistortion(p=probability,
            num_steps=num_steps, border_mode=border_mode)
    elif distortion_type=='elastic':
        alpha = params['alpha']
        sigma = params['sigma']
        alpha_affine = params['alpha_affine']
        trans = A.ElasticTransform(p=probability,
            alpha=alpha, sigma=sigma, alpha_affine=alpha_affine)
    else:
        ValueError('Unexpected flip type')
    return trans

def get_noise(params:dict):
    noise_type = params['type'].lower()
    probability = params['probability']
    if noise_type=='additivegaussiannoise':
        trans = A.IAAAdditiveGaussianNoise(p=probability)
    elif noise_type=='gaussnoise':
        trans = A.GaussNoise(p=probability)
    elif noise_type=='compression':
        h = params['quality_upper']
        l = params['quality_lower']
        compression_type = params['compression_type']
        trans = A.ImageCompression(p=probability,
            quality_lower=l, quality_upper=h,
            compression_type=compression_type)
    elif noise_type=='compression':
        h = params['scale_max']
        l = params['scale_min']
        trans = A.ImageCompression(p=probability,
            scale_min=l, scale_max=h)
    else:
        ValueError('Unexpected flip type')
    return trans

def get_crop(params:dict):
    crop_type = params['type'].lower()

    if crop_type=='crop':
        x_min = params['x_min']
        y_min = params['y_min']
        x_max = params['x_max']
        y_max = params['y_max']
        trans = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True)
    else:
        height = params['height']
        width  = params['width']

        if crop_type=='centercrop':
            trans  = A.CenterCrop(height, width, always_apply=True)
        elif crop_type=='randomcrop':
            trans  = A.RandomCrop(height, width, always_apply=True)
        elif crop_type=='cropnonemptymask':
            ignore_values  = params['ignore_values'] if 'ignore_values' in params.keys() else None
            ignore_channels  = params['ignore_channels'] if 'ignore_channels' in params.keys() else None
            trans  = A.CropNonEmptyMaskIfExists(height, width,
                                ignore_values = ignore_values,
                                ignore_channels = ignore_channels,
                                always_apply=True)
        else:
            ValueError('Unexpected crop type')

    return trans