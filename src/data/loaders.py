"""
Data loaders for handling various image formats and dataset structures.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms


class ImageDataLoader:
    """
    Flexible image data loader supporting various formats and structures.
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, 
                 image_extensions: List[str] = None):
        """
        Initialize image data loader.
        
        Args:
            data_dir: Directory containing images
            batch_size: Batch size for data loading
            image_extensions: Supported image file extensions
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        self.image_paths = self._find_images()
        
    def _find_images(self) -> List[Path]:
        """Find all image files in the directory."""
        image_paths = []
        
        for ext in self.image_extensions:
            pattern = f"**/*{ext}"
            image_paths.extend(self.data_dir.glob(pattern))
            # Also check uppercase extensions
            pattern_upper = f"**/*{ext.upper()}"
            image_paths.extend(self.data_dir.glob(pattern_upper))
        
        return sorted(image_paths)
    
    def load_single_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_batch(self, start_idx: int = 0, batch_size: Optional[int] = None) -> List[Image.Image]:
        """
        Load a batch of images.
        
        Args:
            start_idx: Starting index for the batch
            batch_size: Size of the batch (uses default if None)
            
        Returns:
            List of PIL Images
        """
        batch_size = batch_size or self.batch_size
        end_idx = min(start_idx + batch_size, len(self.image_paths))
        
        images = []
        for i in range(start_idx, end_idx):
            image = self.load_single_image(self.image_paths[i])
            if image is not None:
                images.append(image)
        
        return images
    
    def get_image_info(self) -> Dict[str, Any]:
        """Get information about the loaded images."""
        if not self.image_paths:
            return {"num_images": 0, "extensions": [], "total_size": 0}
        
        # Sample a few images to get average dimensions
        sample_size = min(10, len(self.image_paths))
        sample_images = [self.load_single_image(path) for path in self.image_paths[:sample_size]]
        sample_images = [img for img in sample_images if img is not None]
        
        if sample_images:
            avg_width = sum(img.width for img in sample_images) / len(sample_images)
            avg_height = sum(img.height for img in sample_images) / len(sample_images)
        else:
            avg_width = avg_height = 0
        
        # Get file extensions present
        extensions = set(path.suffix.lower() for path in self.image_paths)
        
        return {
            "num_images": len(self.image_paths),
            "extensions": list(extensions),
            "avg_width": int(avg_width),
            "avg_height": int(avg_height),
            "sample_paths": [str(path) for path in self.image_paths[:5]]
        }


class DatasetLoader(Dataset):
    """
    PyTorch Dataset class for structured datasets with labels.
    """
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None,
                 labels_file: Optional[str] = None, class_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing images (can be structured by class folders)
            transform: Torchvision transforms to apply
            labels_file: CSV file with image names and labels
            class_to_idx: Mapping from class names to indices
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        
        if labels_file and os.path.exists(labels_file):
            self.samples, self.class_to_idx = self._load_from_csv(labels_file, class_to_idx)
        else:
            self.samples, self.class_to_idx = self._load_from_folders()
        
        self.classes = list(self.class_to_idx.keys())
        self.num_classes = len(self.classes)
    
    def _load_from_csv(self, labels_file: str, class_to_idx: Optional[Dict[str, int]]) -> Tuple[List, Dict]:
        """Load samples from CSV file with labels."""
        df = pd.read_csv(labels_file)
        
        # Assume CSV has columns: 'image' and 'label'
        if 'image' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file must have 'image' and 'label' columns")
        
        # Create class to index mapping
        unique_labels = sorted(df['label'].unique())
        if class_to_idx is None:
            class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        samples = []
        for _, row in df.iterrows():
            image_path = self.data_dir / row['image']
            if image_path.exists():
                label_idx = class_to_idx.get(row['label'])
                if label_idx is not None:
                    samples.append((str(image_path), label_idx))
        
        return samples, class_to_idx
    
    def _load_from_folders(self) -> Tuple[List, Dict]:
        """Load samples from folder structure (each subfolder is a class)."""
        samples = []
        class_folders = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not class_folders:
            # Single folder with mixed images - create dummy class
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(self.data_dir.glob(f"*{ext}"))
                image_files.extend(self.data_dir.glob(f"*{ext.upper()}"))
            
            samples = [(str(path), 0) for path in image_files]
            class_to_idx = {"default": 0}
        else:
            # Multiple class folders
            class_to_idx = {folder.name: idx for idx, folder in enumerate(sorted(class_folders))}
            
            for class_folder in class_folders:
                class_idx = class_to_idx[class_folder.name]
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    image_files = list(class_folder.glob(f"*{ext}")) + list(class_folder.glob(f"*{ext.upper()}"))
                    
                    for image_path in image_files:
                        samples.append((str(image_path), class_idx))
        
        return samples, class_to_idx
    
    def __len__(self) -> int:
        """Return size of dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        distribution = {}
        idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        for _, label_idx in self.samples:
            class_name = idx_to_class[label_idx]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        
        return distribution
    
    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                         num_workers: int = 0) -> DataLoader:
        """
        Create a PyTorch DataLoader.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )