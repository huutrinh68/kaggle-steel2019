import albumentations
from albumentations.torch import ToTensor

def get_augmetation(phase):
    if phase == 'train':
        train_augmetations = albumentations.Compose([
            albumentations.RandomResizedCrop(256, 400),
            # albumentations.Resize(256, 1600),
            albumentations.HorizontalFlip(),
            ToTensor(),
        ], p=1.0)
        return train_augmetations
    elif phase == 'valid':
        valid_augmetations = albumentations.Compose([
            albumentations.RandomResizedCrop(256, 400),
            # albumentations.Resize(256, 1600),
            ToTensor(),
        ], p=1.0)
        return valid_augmetations