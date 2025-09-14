"""
Image preprocessing utilities for vision AI tasks.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import torchvision.transforms as transforms


class ImagePreprocessor:
    """
    Image preprocessing utilities for computer vision tasks.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target size for resized images (width, height)
        """
        self.target_size = target_size
    
    def resize_image(self, image: Union[Image.Image, np.ndarray], 
                    size: Optional[Tuple[int, int]] = None) -> Union[Image.Image, np.ndarray]:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            size: Target size (uses default if None)
            
        Returns:
            Resized image
        """
        size = size or self.target_size
        
        if isinstance(image, Image.Image):
            return image.resize(size, Image.Resampling.LANCZOS)
        elif isinstance(image, np.ndarray):
            return cv2.resize(image, size)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image
    
    def get_standard_transforms(self, is_training: bool = False) -> transforms.Compose:
        """
        Get standard preprocessing transforms.
        
        Args:
            is_training: Whether to include training augmentations
            
        Returns:
            Composed transforms
        """
        if is_training:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])