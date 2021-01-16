from typing import List
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder


def get_SegmentationModel(model, encoder, \
    depth=4, class_num=1, encoder_weights='imagenet', activation=None):
    if activation=='None':
        activation = None

    if depth == 5:
        decoder_channels = [256, 128, 64, 32, 16]
    elif depth == 4:
        decoder_channels = [256, 128, 64, 32]
    elif depth == 3:
        decoder_channels = [256, 128, 64]

    if model=='Unet':
        return smp.Unet(encoder, encoder_weights=encoder_weights, \
                    encoder_depth=depth, decoder_channels=decoder_channels, \
                    classes=class_num, activation=activation)
    elif model=='PAN':
        return smp.PAN(encoder, encoder_weights=encoder_weights, \
                    classes=class_num, activation=activation)
    elif model=='DeepLabV3':
        return smp.DeepLabV3(encoder, encoder_weights=encoder_weights, \
                    encoder_depth=depth,\
                    classes=class_num, activation=activation)
    elif model=='DeepLabV3Plus':
        return smp.DeepLabV3Plus(encoder, encoder_weights=encoder_weights, \
                    classes=class_num, activation=activation)
    elif model=='PSPNet':
        return smp.PSPNet(encoder, encoder_weights=encoder_weights, \
                    classes=class_num, activation=activation)
    elif model=='Linknet':
        return smp.PSPNet(encoder, encoder_weights=encoder_weights, \
                    classes=class_num, activation=activation)
    elif model=='FPN':
        return smp.PSPNet(encoder, encoder_weights=encoder_weights, \
                    classes=class_num, activation=activation)
    else:
        assert False, 'Unexpected model_name: {}'.format(model)


class SegmentationEncoder(SegmentationModel):
    def __init__(self, encoder_name:str, model_name:str, depth:int=5,
        weights:str='imagenet', encoder_output_stride:int=8):
        super(SegmentationEncoder, self).__init__()
        self.model = get_encoder(name=encoder_name, depth=depth, weights=weights)
        if model_name == 'DeepLabV3':
            self.model.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
                )
        elif model_name == 'DeepLabV3plus':
            if encoder_output_stride==8:
                stage_list=[4, 5]
                dilation_list=[2, 4]
            elif encoder_output_stride==16:
                stage_list=[5]
                dilation_list=[2]
            self.model.make_dilated(
                stage_list=stage_list,
                dilation_list=dilation_list
                )
        elif model_name == 'Unet':
            pass
        else:
            assert False, f'Unexpected model name: {model_name}'

    def forward(self, x):
        return self.model(x)


class SegmentationDecoder(SegmentationModel):
    def __init__(self, model_name:str, input_sample, classes:int, depth:int=5,
        activation:str=None, decoder_channels: List[int] = (256, 128, 64, 32, 16)
        ):
        super(SegmentationDecoder, self).__init__()
        if model_name == 'DeepLabV3':
            decoder_channels = 256
            self.decoder = DeepLabV3Decoder(
                in_channels=input_sample[-1].size()[1],
                out_channels=decoder_channels
            )
            upsampling = 8
            kernel_size = 1
        elif model_name == 'DeepLabV3plus':
            decoder_channels = 256
            encoder_channels = [s.size()[1] for s in input_sample]
            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=encoder_channels,
                out_channels=decoder_channels
            )
            upsampling = 4
            kernel_size = 1
        elif model_name == 'Unet':
            self.decoder = UnetDecoder(
                encoder_channels=input_sample[-1].size()[1],
                decoder_channels=decoder_channels,
                n_blocks=depth,
                use_batchnorm=True,
                center=True, #if encoder_name.startswith("vgg") else False,
                attention_type=None, # or 'scse'
            )
            upsampling = 0
            kernel_size = 3
        else:
            assert False, f'Unexpected model name: {model_name}'

        activation = activation if activation != 'None' else None
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=kernel_size,
            upsampling=upsampling,
        )

    def forward(self, features):
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks

    def predict(self, x):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x
