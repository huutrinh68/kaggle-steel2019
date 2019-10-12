import albumentations
from albumentations.torch import ToTensor

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def get_augmetation(phase):
    if phase == 'train':
        train_augmetations = [
            # albumentations.RandomResizedCrop(256, 400),
            # albumentations.Resize(256, 256),
            albumentations.Resize(256, 512),
            albumentations.HorizontalFlip(),
            albumentations.Normalize(MEAN, STD),
            ToTensor(),
        ]
        return albumentations.Compose(train_augmetations, p=1)
    elif phase == 'valid':
        valid_augmetations = [
            # albumentations.RandomResizedCrop(256, 400), 256, 1600
            # albumentations.Resize(256, 256),
            albumentations.Resize(256, 512),
            albumentations.Normalize(MEAN, STD),
            ToTensor(),
        ]
        return albumentations.Compose(valid_augmetations, p=1)