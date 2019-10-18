import os
import sys
from shutil import copyfile
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import pdb

from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensor
import torch.utils.data as data
print('torch version:', torch.__version__)
#######################################################################
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../input/segmentation-models-pytorch')
sys.path.append('../input/pretrainedmodels')
sys.path.append('../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master')
sys.path.append('../input/efficientnet-pytorch-b0-b7')
sys.path.append('../input/albumentations')

from segmentation_models_pytorch import Unet, FPN

!mkdir -p /tmp/.cache/torch/checkpoints/
!cp ../input/efficientnet-pytorch-b0-b7/efficientnet-b5-b6417697.pth /tmp/.cache/torch/checkpoints/efficientnet-b5-b6417697.pth

##### net ##############################################################
def Net():
    
    model = Unet(
        encoder_name='resnet152', 
        encoder_weights=None, 
        classes=4, 
        activation='sigmoid')
    # model = FPN(
    #     encoder_name='efficientnet-b5', 
    #     encoder_weights='imagenet', 
    #     classes=4, 
    #     activation='sigmoid')
    
    return model

class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"

# initialize test dataloader
num_workers = 2
batch_size = 4
# best_threshold = 0.5
# min_size = 3500

best_threshold = [0.5,0.5,0.6,0.5]
min_size = [800,1000,3000,3500]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# Initialize mode and load trained weights
ckpt_path = "../input/model_dump1/model.pth"
device = torch.device("cuda")
model = Unet("efficientnet-b5", encoder_weights=None, classes=4, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
    for fname, preds in zip(fnames, batch_preds):
        for cls, pred in enumerate(preds):
            pred, num = post_process(pred, best_threshold[cls], min_size[cls])
            rle = mask2rle(pred)
            name = fname + f"_{cls+1}"
            predictions.append([name, rle])

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("submission.csv", index=False)

if 1:
    df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
    df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
    pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
    pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
    pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
    pos4 = ((df['Class']==4) & (df['Label']==1)).sum()

    num_image = len(df)//4
    num = len(df)
    pos = (df['Label']==1).sum()
    neg = num-pos

    print('')
    print('\t\tnum_image = %5d(1801)'%num_image)
    print('\t\tnum  = %5d(7204)'%num)
    print('\t\tneg  = %5d(6172)  %0.3f'%(neg,neg/num))
    print('\t\tpos  = %5d(1032)  %0.3f'%(pos,pos/num))
    print('\t\tpos1 = %5d( 128)  %0.3f  %0.3f'%(pos1,pos1/num_image,pos1/pos))
    print('\t\tpos2 = %5d(  43)  %0.3f  %0.3f'%(pos2,pos2/num_image,pos2/pos))
    print('\t\tpos3 = %5d( 741)  %0.3f  %0.3f'%(pos3,pos3/num_image,pos3/pos))
    print('\t\tpos4 = %5d( 120)  %0.3f  %0.3f'%(pos4,pos4/num_image,pos4/pos))
