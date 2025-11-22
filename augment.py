
import albumentations as A
import cv2
import numpy as np

transform = A.Compose([
    A.RandomRotate90(p=0.4),
    A.Rotate(limit=25, p=0.6),
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.RGBShift(15,15,15,p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize()
])

def augment_image(image_np):
    """
    image_np: RGB uint8 image
    returns: augmented image as float32 numpy array
    """
    augmented = transform(image=image_np)
    return augmented["image"]
