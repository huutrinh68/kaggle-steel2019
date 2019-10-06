import numpy as np 

def dice_score(predict, target, smooth=1e-5):
    smooth = 1.

    pflat = predict.view(-1)
    tflat = target.view(-1)
    pflat = pflat.cpu().detach()
    tflat = tflat.cpu().detach()
    intersection = (pflat * tflat).sum()
    dice_score = (2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth)

    return dice_score