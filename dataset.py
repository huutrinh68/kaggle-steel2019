import os 
import cv2
import pandas as pd 
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from utils.dataset_utils import make_mask
from augmentations import get_augmetation


DATA_DIR = '../data'

class SteelDataset(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.phase = phase
        self.transforms = get_augmetation(self.phase)

    def __getitem__(self, idx):
        fname, mask = make_mask(idx, self.df)
        image_path = os.path.join(DATA_DIR, 'train_images', fname)
        img = cv2.imread(image_path)

        # augmetation
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.df)


def get_dataframe(args):
    '''
    create dataframe for training, validation
    ----------
    return
    df: dataframe
        processed dataframe
    '''

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    if args.debug:
        print('debug mode...\n')
        df = df.iloc[:100]
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    return df

def get_dataloader(total_df, phase, args):
    '''
    Get train, valid dataloader
    ----------
    parameter
    total_df: dataframe
        from train.csv
    phase: str
        'train', 'valid', or 'test'
    ----------
    return
        dataloader: Dataloader
    '''

    train_df, valid_df = train_test_split(
        total_df, 
        test_size=0.2, 
        stratify=total_df['defects'], 
        random_state=42)
        
    if phase == 'train':
        train_dataset = SteelDataset(train_df, phase)
        return DataLoader(
            dataset = train_dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
    
    if phase == 'valid':
        valid_dataset = SteelDataset(valid_df, phase)
        return DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )