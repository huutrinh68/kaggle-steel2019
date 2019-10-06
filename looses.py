import segmentation_models_pytorch as smp

def get_criterion():
    return smp.utils.losses.BCEDiceLoss(eps=1.)