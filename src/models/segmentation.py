import segmentation_models_pytorch as smp

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