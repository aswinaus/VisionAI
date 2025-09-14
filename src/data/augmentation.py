"""
Image augmentation utilities for data enhancement.
"""

import random
from typing import Union, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class ImageAugmenter:
    """
    Image augmentation utilities for training data enhancement.
    """
    
    def __init__(self, augmentation_config: dict = None):
        """
        Initialize augmenter with configuration.
        
        Args:
            augmentation_config: Dictionary with augmentation parameters
        """
        self.config = augmentation_config or self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default augmentation configuration."""
        return {
            "rotation_range": 15,
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.8, 1.2),
            "saturation_range": (0.8, 1.2),
            "hue_range": (-0.1, 0.1),
            "flip_probability": 0.5,
            "blur_probability": 0.1,
            "noise_probability": 0.1
        }
    
    def random_rotation(self, image: Image.Image, max_angle: int = None) -> Image.Image:
        """Apply random rotation to image."""
        max_angle = max_angle or self.config.get("rotation_range", 15)
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    def random_brightness(self, image: Image.Image, brightness_range: Tuple = None) -> Image.Image:
        """Apply random brightness adjustment."""
        brightness_range = brightness_range or self.config.get("brightness_range", (0.8, 1.2))
        factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def random_contrast(self, image: Image.Image, contrast_range: Tuple = None) -> Image.Image:
        """Apply random contrast adjustment."""
        contrast_range = contrast_range or self.config.get("contrast_range", (0.8, 1.2))
        factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def random_flip(self, image: Image.Image, probability: float = None) -> Image.Image:
        """Apply random horizontal flip."""
        probability = probability or self.config.get("flip_probability", 0.5)
        if random.random() < probability:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    def random_blur(self, image: Image.Image, probability: float = None) -> Image.Image:
        """Apply random blur."""
        probability = probability or self.config.get("blur_probability", 0.1)
        if random.random() < probability:
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return image
    
    def get_torchvision_transforms(self, is_training: bool = True) -> transforms.Compose:
        """
        Get torchvision-based augmentation pipeline.
        
        Args:
            is_training: Whether to include training augmentations
            
        Returns:
            Composed transforms
        """
        if not is_training:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=self.config.get("flip_probability", 0.5)),
            transforms.ColorJitter(
                brightness=self.config.get("brightness_range", (0.8, 1.2)),
                contrast=self.config.get("contrast_range", (0.8, 1.2)),
                saturation=self.config.get("saturation_range", (0.8, 1.2)),
                hue=self.config.get("hue_range", (-0.1, 0.1))
            ),
            transforms.RandomRotation(self.config.get("rotation_range", 15)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_albumentations_transforms(self, is_training: bool = True) -> Union[A.Compose, None]:
        """
        Get albumentations-based augmentation pipeline.
        
        Args:
            is_training: Whether to include training augmentations
            
        Returns:
            Albumentations compose object or None if not available
        """
        if not HAS_ALBUMENTATIONS:
            print("Albumentations not available, use torchvision transforms instead")
            return None
        
        if not is_training:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        return A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=self.config.get("flip_probability", 0.5)),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.Rotate(limit=self.config.get("rotation_range", 15), p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=self.config.get("blur_probability", 0.1)),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])