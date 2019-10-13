'''Get optimizer for model'''
import torch

def get_optimizer(model, type, log):
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
        # optimizer = torch.optim.Adam([
        #     {'params': model.encoder.parameters(), 'lr': 5e-4},
        #     {'params': model.decoder.parameters(), 'lr': 5e-3},
        #     ])
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    ## log
    log.write(f'\n--------------------\n')
    log.write(f'\noptimizer   = {type}\n')
    log.write(f'encoder lr  = 5e-4\n')
    log.write(f'decoder lr  = 5e-3\n')

    return optimizer
