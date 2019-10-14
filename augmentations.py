import albumentations
from albumentations.torch import ToTensor

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

def get_augmetation(phase):
    
    list_transforms = []

    if phase == 'train':
        list_transforms.extend(
            [
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip(),
            ]
        )
    
    list_transforms.extend(
        [
            albumentations.Normalize(mean=IMAGE_RGB_MEAN, std=IMAGE_RGB_STD),
            ToTensor(),
        ]
    )

    return albumentations.Compose(list_transforms, p=1)