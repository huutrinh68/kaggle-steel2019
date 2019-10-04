import albumentations as A
from albumentations.torch import ToTensor

def get_augmetation(phase):
    if phase == 'train':
        return train_augmetations = A.Compose([
            A.HorizontalFlip(),
            A.ToTensor(),
        ], p=1.0)
    elif phase == 'valid':
        return valid_augmetations = A.Compose([
            ToTensor(),
        ], p=1.0)