import albumentations
from albumentations.torch import ToTensor

def get_augmetation(phase):
    if phase == 'train':
        train_augmetations = [
            # albumentations.RandomResizedCrop(256, 400),
            albumentations.Resize(256, 256),
            albumentations.HorizontalFlip(),
            ToTensor(),
        ]
        return albumentations.Compose(train_augmetations, p=1)
    elif phase == 'valid':
        valid_augmetations = [
            # albumentations.RandomResizedCrop(256, 400), 256, 1600
            albumentations.Resize(256, 256),
            ToTensor(),
        ]
        return albumentations.Compose(valid_augmetations, p=1)