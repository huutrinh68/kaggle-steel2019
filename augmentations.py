import albumentations as A
from albumentations.torch import ToTensor

def get_augmetation(phase):
    if phase == 'train':
        train_augmetations = A.Compose([
            A.RandomResizedCrop(256, 400),
            # A.Resize(256, 1600),
            A.HorizontalFlip(),
            ToTensor(),
        ], p=1.0)
        return train_augmetations
    elif phase == 'valid':
        valid_augmetations = A.Compose([
            A.RandomResizedCrop(256, 400),
            # A.Resize(256, 1600),
            ToTensor(),
        ], p=1.0)
        return valid_augmetations