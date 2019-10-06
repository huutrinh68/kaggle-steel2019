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

# save ktop epoch
def save_top_epochs(model_out_dir, model, best_dice_arr, valid_dice, best_epoch, epoch, best_dice, ema=False):
    best_dice_arr = best_dice_arr.copy()

    if ema:
        suffix = '_ema'
    else:
        suffix = ''
    
    min_dice = np.min(best_dice_arr)
    last_ix = len(best_dice_arr) - 1

    def get_top_path(ix):
        return os.path.join(model_out_dir, f'top{ix+1}{suffix}.pth')

    if valid_dice > min_dice:
        min_ix = last_ix
        for ix, score in enumerate(best_dice_arr):
            if score < valid_dice:
                min_ix = ix
                break
        
        lowest_path = get_top_path(last_ix)
        if os.path.exists(lowest_path):
            os.remove(lowest_path)
        
        for ix in range(last_ix - 1, min_ix - 1, -1):
            score = best_dice_arr[ix]
            best_dice_arr[ix + 1] = score
            if score > 0 and os.path.exists(get_top_path(ix)):
                os.rename(get_top_path(ix), get_top_path(ix + 1))
        
        best_dice_arr[min_ix] = valid_dice

        model_name = f'top{min_ix+1}'

        save_model(model, model_out_dir, epoch, model_name, best_dice_arr, is_best=False,
                   optimizer=None, best_epoch=best_epoch, best_dice=best_dice, ema=ema)
        
    return best_dice_arr

# save epoch
def save_model(model, model_out_dir, epoch, model_name, best_dice_arr, is_best=False, 
               optimizer=None, best_epoch=None, best_dice=None, ema=False):
    if type(model) == nn.DataParallel: 
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    if ema:
        model_fpath = os.path.join(model_out_dir, f'{model_name}_ema.pth')
    else:
        model_fpath = os.path.join(model_out_dir, f'{model_name}.pth')
    torch.save({
        'state_dict': state_dict,
        'best_epoch': best_epoch,
        'epoch': epoch,
        'best_dice': best_dice,
        'best_dice_arr': best_dice_arr,
    }, model_fpath)

    optim_fpath = os.path.join(model_out_dir, f'{model_name}_optim.pth')
    if optimizer is not None:
        torch.save({
            'optimizer': optimizer.state_dict(),
        }, optim_fpath)

    if is_best:
        if ema:
            best_model_fpath = os.path.join(model_out_dir, 'final_ema.pth')
        else:
            best_model_fpath = os.path.join(model_out_dir, 'final.pth')
        shutil.copyfile(model_fpath, best_model_fpath)
        if optimizer is not None:
            best_optim_fpath = os.path.join(model_out_dir, 'final_optim.pth')
            shutil.copyfile(optim_fpath, best_optim_fpath)