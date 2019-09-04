import numpy as np 
import cv2 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from os.path import isfile
import albumentations
from albumentations.pytorch import ToTensor
import pandas as pd
import os
import torch
from config import config

def train_transform():
    # return albumentations.Compose([
    # albumentations.Resize(224, 224),
    # albumentations.RandomRotate90(p=0.5),
    # albumentations.Transpose(p=0.5),
    # albumentations.Flip(p=0.5),
    # albumentations.OneOf([
    #     albumentations.CLAHE(clip_limit=2), albumentations.RandomBrightness(), albumentations.RandomContrast(),
    #     albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 
    # albumentations.HueSaturationValue(p=0.5), 
    # albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
    # # albumentations.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=365, p=1.0),
    # # albumentations.RandomBrightnessContrast(p=1.0),
    # albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ToTensor()])
    return albumentations.Compose([
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=365, p=1.0),
    albumentations.RandomBrightnessContrast(p=1.0),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensor()])


def test_transform():
    return albumentations.Compose([
    albumentations.Resize(config.img_size, config.img_size),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensor()])


def make_mask(row_id, df):
    image_names = [str(i).split("_")[0] for i in df.iloc[row_id:row_id+4, 0].values]
    # if not (image_names[0] == image_names[1] == image_names[2] == image_names[3]):
    #     raise ValueError
        
    labels = df.iloc[row_id:row_id+4, 1].values
    masks = np.zeros((256, 1600, 4), dtype=np.float32)

    for idx, label in enumerate(labels):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256*1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos-1:(pos+le-1)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order="F")
    return image_names[0], masks


class MyDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image = cv2.imread(os.path.join(config.train_img, image_id))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            # mask = augmented["mask"]
            # mask = mask[0].permute(2,0,1)
        
        return {"image": image, "label": mask}


# if __name__ == "__main__":
#     df = pd.read_csv(config.train_csv)
#     image_id, mask = make_mask(0, df)
#     print(image_id, mask)

