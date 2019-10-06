import segmentation_models_pytorch as smp

def init_network():
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=4, activation='sigmoid')
    return model