import torch
import random
import os
import sys
import numpy as np
import shutil
from config import config


# create seed random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# save best model
def save_checkpoint(state, is_best_loss, fold, epoch):
    filename = config["checkpoint"]["path"] + configs["checkpoint"]["model_name"] + os.sep + str(fold) + os.sep + str(epoch) + "_checkpoint.pt"
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename, "{}/{}_fold_{}_model_best_loss.pt".
        format(configs["checkpoint"]["best_model"], configs["checkpoint"]["model_name"], str(fold)))
    # if is_best_f1:
    #     shutil.copyfile(filename, "{}/{}_fold_{}_model_best_f1.pth.tar".
    #     format(configs["checkpoint"]["best_model"], configs["checkpoint"]["model_name"], str(fold)))


# create folder  save checkpoint
def create_folder():
    n_splits = config.n_splits
    for fold in range(n_splits):
        fold_path = config.output_path + config.model_name + os.sep + str(fold)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
    if not os.path.exists(config.best_model):
        os.mkdir(config.best_model)

    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

# evaluate meters
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


# print logger
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


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError
