# src/training/augs.py
import albumentations as A

def get_train_transforms():
    """
    Albumentations pipeline for face augmentation during training.
    - Lighting/night: RandomGamma, RandomBrightnessContrast
    - Weather: RandomRain, RandomFog
    - Occlusion: CoarseDropout (simulates masks/sunglasses)
    - Robustness: GaussNoise, MotionBlur
    NOTE: Use only during training.
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.OneOf([
            A.RandomRain(blur_value=3, brightness_coefficient=0.8, p=0.6),
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=0.6),
        ], p=0.3),
        A.CoarseDropout(max_holes=2, max_height=24, max_width=48, min_holes=1, p=0.45),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.35),
        A.HorizontalFlip(p=0.5),
    ])

def get_val_transforms():
    """No augmentation for validation / test."""
    return A.Compose([])
