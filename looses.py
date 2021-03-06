import segmentation_models_pytorch as smp
import torch 
import torch.nn as nn


def get_criterion(args, log):

    if args.loss_type == 'bcelogit':
        criterion = torch.nn.BCEWithLogitsLoss()
    if args.loss_type == 'bcedice':
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        
    log.write(f'criterion   = {args.loss_type}\n')
    return criterion