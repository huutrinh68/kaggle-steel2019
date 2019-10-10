import segmentation_models_pytorch as smp

def init_network(args):
    model = smp.Unet(args.model, encoder_weights='imagenet', classes=4, activation='sigmoid')
    return model