'''Get optimizer for model'''
import torch

def get_optimizer(model, type):
    '''
    return optimizer
    ----
    params
        model: module
        type: str
    return
        optimizer
    '''
    if type.lower() == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 5e-4},
            {'params': model.decoder.parameters(), 'lr': 5e-3},
            ])
    return optimizer
