import albumentations
from albumentations.torch import ToTensor

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = [256, 400]

def get_augmetation(phase):
    
    list_transforms = []

    if phase == 'train':
        list_transforms.extend(
            [
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip(),
                albumentations.ShiftScaleRotate(shift_limit=0.03, scale_limit=0, rotate_limit=(-3,3), border_mode=0),
                albumentations.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], border_mode=0),
                albumentations.RandomCrop(*IMAGE_SIZE),
                albumentations.RandomBrightness(limit=(-0.25, 0.25)),
                albumentations.RandomContrast(limit=(-0.15, 0.4)),
                albumentations.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            ]
        )
    
    list_transforms.extend(
        [
            albumentations.Normalize(mean=IMAGE_RGB_MEAN, std=IMAGE_RGB_STD),
            ToTensor(),
        ]
    )

    return albumentations.Compose(list_transforms, p=1)