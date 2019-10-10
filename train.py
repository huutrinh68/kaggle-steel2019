import argparse
import os 
import time
import json

import torch 
import numpy as np

from dataset import get_dataloader, get_dataframe
from models.resnet18 import init_network
from optimizers import get_optimizer
from schedulers import get_scheduler
from looses import get_criterion
from utils.helpers import AverageMeter, Logger, seed_everything, \
    report_checkpoint, accumulate, save_top_epochs, save_model
from metrics import dice_score

import torch.nn as nn
import gc
from apex import amp

import warnings
warnings.filterwarnings('ignore')

# train ########################
def train(model, ema_model, train_loader, optimizer, criterion, epoch, lr, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()

    # switch to train mode
    model.train() 
    
    num_its = len(train_loader)
    end = time.time()
    
    for idx, (inputs, labels) in enumerate(train_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        # move data to device
        inputs = inputs.to(args.device, dtype=torch.float)
        labels = labels.to(args.device, dtype=torch.float)
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item()) # or losses.update(loss.item(), inputs.size(0))

            loss = loss / args.accumulate_step
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        probs = torch.sigmoid(outputs)
        dice = dice_score(probs, labels)
        # dice = dice_score(outputs, labels)
        dices.update(dice.item()) # or dices.update(dice.item(), inputs.size(0))

        if args.ema:
            if epoch >= args.ema_start:
                accumulate(ema_model, model, decay=args.ema_decay)
            else:
                accumulate(ema_model, model, decay=0)

        # measure elapsed time
        batch_time.update(time.time() -end)
        end = time.time()

        # update params by accumulate gradient
        if idx ==0 or (idx+1) % args.accumulate_step == 0 or (idx+1) == num_its:
            optimizer.step() 
            # model.zero_grad() 
            optimizer.zero_grad()
            
            print('\r', end='', flush=True)
            print('\r%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  |  ... ' % \
                  (epoch - 1 + (idx + 1) / num_its, idx + 1, lr, losses.avg, dices.avg), \
                   end='', flush=True)


    return idx, losses.avg, dices.avg


# valid ########################
def valid(model, valid_loader, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for it, (inputs, labels) in enumerate(valid_loader, 0):
        # measure data loading time
        data_time.update(time.time() -end)

        # move variable to device
        inputs = inputs.to(args.device, dtype=torch.float)
        labels = labels.to(args.device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item()) # or losses.update(loss.item(), inputs.size(0))

        probs = torch.sigmoid(outputs)
        dice = dice_score(probs, labels)
        # dice = dice_score(outputs, labels)
        dices.update(dice.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, dices.avg


def main(args):
    # set device ###################
    if 'cuda' in args.device:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
        torch.backends.cudnn.benchmark=True
    device = torch.device(args.device)

    # set log ######################
    log = Logger()

    if args.folds is None or len(args.folds)== 0:
        folds = [None]
    for i, fold in enumerate(folds):
        torch.cuda.empty_cache()

        # set directory
        directory_prefix = f'{args.model}'
        if fold is not None:
            directory_prefix += f'_fold{fold}'
        log_dir = os.path.join('runs', directory_prefix)
        os.makedirs(log_dir, exist_ok=True)

        model_out_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(model_out_dir, exist_ok=True)

        # write setting json file
        config_fname = os.path.join(log_dir, f'{directory_prefix}.json')
        with open(config_fname, 'w') as f:
            train_session_args = vars(args)
            f.write(json.dumps(train_session_args, indent=2))

        # set up log
        log.open(os.path.join(log_dir, f'log.train_fold{fold}.txt'), mode='a')
        log.write(f'training fold_{i}\n')

        # train setup
        start_epoch = 0
        best_epoch = 0
        best_dice = 0
        best_dice_arr = np.zeros(3)
        early_stopping_count = 0
        seed_everything(args.seed)

        # data #########################
        total_df = get_dataframe(args)

        train_loader = get_dataloader(total_df=total_df, phase='train', args=args)
        valid_loader = get_dataloader(total_df=total_df, phase='valid', args=args)
        
        log.write(f"train length: {len(train_loader)}\n")
        log.write(f"valid length: {len(valid_loader)}\n")


        # model ########################
        model = init_network()
        model = model.to(device)

        if args.ema:
            ema_model = copy.deepcopy(model)
            ema_model = ema_model.to(device)
        else:
            ema_model = None

        # optimizer ####################
        optimizer = get_optimizer(model, 'adam')

        # f16 ##########################
        # Initialization
        # opt_level = 'O1'
        # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        # scheduler ####################
        scheduler = get_scheduler(optimizer, 'reducelronplateau', args)

        # criterion ####################
        criterion = get_criterion()

        if args.resume:
            model_fpath = os.path.join(model_out_dir, args.resume)
            if os.path.isfile(model_fpath):
                # load checkpoint weights and update model and optimizer
                log.write(f">> Loading checkpoint:\n>> '{model_fpath}'\n")
                
                # show report
                checkpoint = torch.load(model_fpath)
                report_checkpoint(checkpoint)

                # re-itinial train setting
                start_epoch = checkpoint['epoch']
                best_epoch = checkpoint['best_epoch']
                best_dice_arr = checkpoint['best_dice_arr']
                best_dice = np.max(best_dice_arr)
                
                # load model
                if type(model) == nn.DataParallel: 
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])

                # load optimizer
                optimizer_fpath = model_fpath.replace('.pth', '_optim.pth')
                if os.path.exists(optimizer_fpath):
                    log.write(f">> Loading checkpoit:\n>> '{optimizer_fpath}'\n")
                    optimizer.load_state_dict(torch.load(optimizer_fpath)['optimizer'])
                
                # if using ema
                if args.ema:
                    ema_model_fpath =model_fpath.replace('.pth', '_ema.pth')
                    if os.path.join(ema_model_fpath):
                        log.write(f">> Loading checkpoint:\n>>{ema_model_fpath}")
                        ema_model.load_state_dict(torch.load(ema_model_fpath)['state_dict'])

                log.write(f">> Loaded checkpoint:\n>> '{model_fpath}' (epoch {checkpoint['epoch']})\n")

                del checkpoint
                gc.collect()
            else:
                log.write(f">> No checkpoint found at '{model_fpath}'\n")
        
        # train model ##################
        log.write('** start training here! **\n')
        log.write('\n')
        log.write('epoch    iter      rate     | smooth_loss/dice | valid_loss/dice | best_epoch/best_score |  min \n')
        log.write('----------------------------------------------------------------------------------------------- \n')
        start_epoch += 1
        for epoch in range(start_epoch, args.epochs+1):
            end = time.time()
            # lr for encoder
            lr = optimizer.param_groups[0]['lr']

            # train ####################
            idx, train_loss, train_dice = train(model, ema_model, train_loader, optimizer, criterion, epoch, lr, args)

            # valid ####################
            with torch.no_grad():
                if args.ema:
                    valid_loss, valid_dice = valid(ema_model, valid_loader, criterion, args)
                else:
                    valid_loss, valid_dice = valid(model, valid_loader, criterion, args)

            # decay lr if valid_loss doesn't decrease
            scheduler.step(valid_loss)

            # remember best loss and save checkpoint
            is_best = valid_dice >= best_dice
            if is_best:
                best_epoch = epoch
                best_dice = valid_dice
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            
            if args.ema:
                save_top_epochs(model_out_dir, ema_model, best_dice_arr, valid_dice,
                                best_epoch, epoch, best_dice, ema=True)

            best_dice_arr = save_top_epochs(model_out_dir, model, best_dice_arr, valid_dice, 
                                            best_epoch, epoch, best_dice, ema=False)

            print('\r', end='', flush=True)
            log.write('%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  |  %0.4f  %6.4f |  %6.1f     %6.4f    | %3.1f min \n' % \
                (epoch, idx + 1, lr, train_loss, train_dice, valid_loss, valid_dice, best_epoch, best_dice, (time.time() - end) / 60))

            # savemodel
            model_savename = f'{epoch:03d}'
            if args.ema:
                save_model(ema_model, model_out_dir, epoch, model_savename, best_dice_arr, is_best=is_best,
                           optimizer=optimizer, best_epoch=best_epoch, best_dice=best_dice, ema=True)
            save_model(model, model_out_dir, epoch, model_savename, best_dice_arr, is_best=is_best, 
                       optimizer=optimizer, best_epoch=best_epoch, best_dice=best_dice, ema=False)      
            
            if args.early_stopping:
                if early_stopping_count >= args.early_stopping:
                    print('='*30, '>early stopped!')
                    break

            time.sleep(0.01)



# main ###############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for Steel kaggle competetion.')
    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    parser.add_argument('-m', '--model', default='efficientnet-b5', type=str, help='model arch')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='train on device')
    parser.add_argument('-f', '--factor', default=0.75, type=float, help='factor to decrease lr')
    parser.add_argument('-p', '--patience', default=2, type=int, help='patience epoch number')
    parser.add_argument('-e', '--epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--num_workers', default=0, type=int, help='number worker')
    parser.add_argument('--batch_size', default=2, type=int, help='batchsize number')
    parser.add_argument('--accumulate_step', default=2, type=int, help='accumulate_step')
    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--ema_start', type=int, default=0)
    parser.add_argument('--folds', action='append', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42, help='seed number')
    parser.add_argument('--resume', default='top1.pth', type=str, 
                        help='name of the latest checkpoint (default: None)')
    parser.add_argument('--early_stopping', type=int, default=10, help='number epoch need to early stop')
                        
    args = parser.parse_args()

    main(args)
    print('\nsucess!')