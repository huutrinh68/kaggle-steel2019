import os
import sys
import time
import json
import gc
from tqdm import tqdm
from datetime import datetime
from timeit import default_timer as timer
import warnings
import torch
import torch.nn.init as init
import torch.nn as nn
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import Adam, SGD, RMSprop
import torch.functional as F
import torch.nn.functional as F
from utils import seed_everything, save_checkpoint, create_folder, Logger, AverageMeter, time_to_str
from dataset import MyDataset, train_transform, test_transform
from train import train_model, valid_model
from model import my_model
from submission import create_submission
from config import config
torch.backends.cudnn.benchmark=True
warnings.filterwarnings('ignore')
print(os.listdir("../input/severstal-steel-defect-detection/"))

# 1. setting device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda:1"
start = timer()


# 2. setting parameters
tta                 = config.n_tta
num_classes         = config.n_classes
img_size            = config.img_size
batch_size          = config.batch_size
lr                  = config.lr
n_epochs            = config.n_epochs
best_avg_loss       = config.best_avg_loss
weight_decay        = config.weight_decay
step_size           = config.n_step
gamma               = config.gamma
num_workers         = config.n_workers
test_size           = config.test_size
random_state        = config.random_state
n_splits            = config.n_splits
patience            = config.patience
# 2.2. input data path
test_img            = config.test_img
train_img           = config.train_img
train_csv           = config.train_csv
sample_submission   = config.sample_submission_csv

# 2.3. prepare folder to save checkpoint and setting seed
create_folder()

seed_everything(random_state)
# 2.4. logger
log = Logger()
log.open("{}/{}_log_train.txt".format(config.logs, config.model_name), mode="a")
log.write("\n---------------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-" * 56))
log.write("                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n")
log.write("mode     iter     epoch    |         loss    metrics        |         loss    metrics       |         loss    metrics       | time       |\n")
log.write("-----------------------------------------------------------------------------------------------------------------------------------------|\n")

# 3. prepare data to train, valid, test
df = pd.read_csv(train_csv)

train_df, valid_df  = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

# 3.1. data to train
trainset            = MyDataset(train_df, transform=train_transform())
train_loader        = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# 3.2 data to valid
validset            = MyDataset(valid_df, transform=train_transform())
valid_loader        = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# 4. create model and optimizer, criterion, scheduler setting
# 4.1 model
model = my_model()
model.to(device)
# 4.2 optimizer, criterion, scheduler setting
# criterion = nn.MSELoss()
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

valid_metrics = [np.inf, 0]
best_results = [np.inf, 0]

# 5. train and validate for each epoch
for epoch in range(n_epochs):
        start_time   = time.time()
        # train
        train_metrics= train_model(model, epoch, optimizer, criterion, train_loader, valid_metrics, device, best_results, log, start, classification=True)
        # validate
        # valid_metrics = valid_model(model, epoch, criterion, valid_loader, train_metrics, device, best_results, log, start, classification=True)
        # elapsed_time = time.time() - start_time 
        # # check results
        # is_best_loss = val_metrics[0] < best_results[0]
        # best_results[0] = min(val_metrics[0], best_results[0])
        # is_best_f1 = val_metrics[1] > best_results[1]
        # best_results[1] = max(val_metrics[1], best_results[1])

        # # save model
        # save_checkpoint({
        #                 "epoch":epoch + 1,
        #                 "model_name":configs["checkpoint"]["model_name"],
        #                 "state_dict":model.state_dict(),
        #                 "best_loss":best_results[0],
        #                 "optimizer":optimizer.state_dict(),
        #                 "fold":fold,
        # }, configs, is_best_loss, fold, epoch)

        # # print logs
        # print('\r',end='', flush=True)
        # log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
        #         "best", epoch+1, epoch+1,
        #         train_metrics[0], train_metrics[1],
        #         val_metrics[0], val_metrics[1],
        #         str(best_results[0])[:8], str(best_results[1])[:8],
        #         time_to_str((timer() - start), 'min'))
        #         )
        # log.write("\n")
        # time.sleep(0.01)

        # # change learning rate
        # scheduler.step(epoch)


# # # 6. create submission file
# # # 6.1 data to test
# # # test_transform  = test_transform()
# # testset         = MyDataset(pd.read_csv(sample_submission), configs, transform=test_transform(configs))
# # test_loader     = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
# # create_submission(model, device, configs, test_loader)
