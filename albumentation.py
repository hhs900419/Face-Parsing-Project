import albumentations as albu
import cv2

RESIZE_SIZE = 448

def get_training_augmentation():
    train_transform = [
        albu.Rotate(limit=20,p=0.5,border_mode=cv2.BORDER_CONSTANT),
        albu.OneOf([
            albu.RandomBrightnessContrast(),
            albu.HueSaturationValue(),
            albu.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
            # albu.ColorJitter(p=0.5),
            albu.Sharpen(),
        ], p=0.35),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=0.3),
            albu.GaussNoise(p=0.3),
            albu.GaussianBlur(p=0.3),
        ], p=0.3)
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
