import os
import numpy as np
import torch 
import random

import sys 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: 
	        mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: 
            is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# create seed random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# report checkpoint when load checkpoint
def report_checkpoint(checkpoint):
    print('Best Epoch    :', checkpoint['best_epoch'])
    print('Best Dice     :', checkpoint['best_dice'])
    print('Best Dice Arr :', checkpoint['best_dice_arr'])

# accumulate when using ema
def accumulate(model1, model2, decay=0.99):
    par1 = model1.state_dict()
    par2 = model2.state_dict()

    with torch.no_grad():
        for k in par1.keys():
            par1[k].data.copy_(par1[k].data * decay + par2[k].data * (1 - decay))