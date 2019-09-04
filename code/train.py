from tqdm import tqdm
import torch 
from utils import AverageMeter, time_to_str
from sklearn.metrics import f1_score
from timeit import default_timer as timer
from config import config
from tensorboardX import SummaryWriter

writer = SummaryWriter()

def train_model(model, epoch, optimizer, criterion, train_loader, valid_loss, device, best_results, log, start, classification=True):
    if epoch == 1:
        for param in model.parameters():
                param.requires_grad = True
    avg_loss = 0.
    optimizer.zero_grad()
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train() 
    for idx, batch in enumerate(train_loader):
        if classification:
            inputs = batch["image"]
            labels = batch["label"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
        else:
            inputs = batch["image"]
            labels = batch["label"].view(-1, 1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        
        f1_batch = f1_score(labels.cpu(), outputs.argmax(-1).cpu().detach().numpy(), average='macro')
        f1.update(f1_batch, inputs.size(0))

        # update params by accumulate gradient
        if (idx+1) % 5 == 0:
            optimizer.step() 
            optimizer.zero_grad() 
            
            print('\r',end='',flush=True)
            message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "train", idx/len(train_loader) + epoch, epoch,
                    losses.avg, f1.avg,
                    valid_loss[0], valid_loss[1],
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
            print(message , end='',flush=True)

    # writer.add_scalar(configs["checkpoint"]["train_loss"], losses.avg, epoch)
    # writer.add_scalar(configs["checkpoint"]["train_f1"], f1.avg, epoch)

    log.write("\n")
        
    return [losses.avg, f1.avg]


def valid_model(model, epoch, criterion, val_loader, train_loss, device, best_results, log, start, classification=True):
    avg_val_loss = 0.
    losses = AverageMeter()
    f1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if classification:
                inputs = batch["image"]
                labels = batch["label"]
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
            else:
                inputs = batch["image"]
                labels = batch["label"].view(-1, 1)
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            f1_batch = f1_score(labels.cpu(), outputs.argmax(-1).cpu().detach().numpy(), average='macro')
            f1.update(f1_batch, inputs.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", idx/len(val_loader) + epoch, epoch,
                    train_loss[0], train_loss[1],
                    losses.avg, f1.avg,
                    str(best_results[0])[:8], str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")

    # writer.add_scalar(configs["checkpoint"]["eval_loss"], losses.avg, epoch)
    # writer.add_scalar(configs["checkpoint"]["eval_f1"], f1.avg, epoch)

        
    return [losses.avg, f1.avg]
