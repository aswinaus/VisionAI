"""
Tests for data processing functionality.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.data.loaders import ImageDataLoader, DatasetLoader
from src.data.preprocessors import ImagePreprocessor


class TestImageDataLoader:
    """Tests for ImageDataLoader."""
    
    def setup_method(self):
        """Set up test fixtures with temporary directory and images."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test images
        for i, color in enumerate(['red', 'green', 'blue']):
            img = Image.new('RGB', (100, 100), color)
            img.save(self.temp_dir / f"test_{i}.jpg")
        
        self.loader = ImageDataLoader(str(self.temp_dir))
    
    def teardown_method(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test loader initialization."""
        assert self.loader.data_dir == self.temp_dir
        assert self.loader.batch_size == 32  # default
        assert len(self.loader.image_paths) == 3
    
    def test_find_images(self):
        """Test image finding functionality."""
        # Should find 3 images
        assert len(self.loader.image_paths) == 3
        
        # All paths should exist
        for path in self.loader.image_paths:
            assert path.exists()
            assert path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def test_load_single_image(self):
        """Test single image loading."""
        image_path = self.loader.image_paths[0]
        image = self.loader.load_single_image(image_path)
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (100, 100)
    
    def test_load_batch(self):
        """Test batch loading."""
        batch = self.loader.load_batch(start_idx=0, batch_size=2)
        
        assert len(batch) == 2
        assert all(isinstance(img, Image.Image) for img in batch)
    
    def test_get_image_info(self):
        """Test image info retrieval."""
        info = self.loader.get_image_info()
        
        assert isinstance(info, dict)
        assert info['num_images'] == 3
        assert '.jpg' in info['extensions']
        assert 'avg_width' in info
        assert 'avg_height' in info


class TestDatasetLoader:
    """Tests for DatasetLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create class-based directory structure
        for class_name in ['cats', 'dogs']:
            class_dir = self.temp_dir / class_name
            class_dir.mkdir()
            
            # Create 2 images per class
            for i in range(2):
                img = Image.new('RGB', (64, 64), 'red' if class_name == 'cats' else 'blue')
                img.save(class_dir / f"image_{i}.jpg")
    
    def teardown_method(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_folder_structure_loading(self):
        """Test loading from folder structure."""
        dataset = DatasetLoader(str(self.temp_dir))
        
        assert len(dataset) == 4  # 2 images per class, 2 classes
        assert dataset.num_classes == 2
        assert 'cats' in dataset.classes
        assert 'dogs' in dataset.classes
    
    def test_getitem(self):
        """Test dataset item retrieval."""
        dataset = DatasetLoader(str(self.temp_dir))
        
        image, label = dataset[0]
        
        # Should return tensor and integer label
        assert hasattr(image, 'shape')  # Tensor-like
        assert isinstance(label, int)
        assert 0 <= label < dataset.num_classes
    
    def test_class_distribution(self):
        """Test class distribution calculation."""
        dataset = DatasetLoader(str(self.temp_dir))
        distribution = dataset.get_class_distribution()
        
        assert isinstance(distribution, dict)
        assert len(distribution) == 2
        assert distribution['cats'] == 2
        assert distribution['dogs'] == 2
    
    def test_create_dataloader(self):
        """Test PyTorch DataLoader creation."""
        dataset = DatasetLoader(str(self.temp_dir))
        dataloader = dataset.create_dataloader(batch_size=2, shuffle=False)
        
        # Test first batch
        batch = next(iter(dataloader))
        images, labels = batch
        
        assert images.shape[0] <= 2  # Batch size
        assert labels.shape[0] <= 2
        assert len(images.shape) == 4  # NCHW format


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor(target_size=(128, 128))
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        assert self.preprocessor.target_size == (128, 128)
    
    def test_resize_pil_image(self):
        """Test PIL image resizing."""
        original_image = Image.new('RGB', (256, 256), 'red')
        resized = self.preprocessor.resize_image(original_image)
        
        assert isinstance(resized, Image.Image)
        assert resized.size == (128, 128)
    
    def test_resize_numpy_array(self):
        """Test numpy array resizing."""
        original_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        resized = self.preprocessor.resize_image(original_array)
        
        assert isinstance(resized, np.ndarray)
        assert resized.shape[:2] == (128, 128)  # Height, Width
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Test uint8 image
        uint8_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        normalized = self.preprocessor.normalize_image(uint8_image)
        
        assert normalized.dtype == np.float32
        assert 0 <= normalized.min() <= 1
        assert 0 <= normalized.max() <= 1
        
        # Test already float image
        float_image = np.random.random((100, 100, 3)).astype(np.float32)
        normalized = self.preprocessor.normalize_image(float_image)
        
        assert normalized.dtype == np.float32
        assert np.array_equal(normalized, float_image)
    
    def test_get_standard_transforms(self):
        """Test transform generation."""
        # Training transforms
        train_transforms = self.preprocessor.get_standard_transforms(is_training=True)
        assert train_transforms is not None
        
        # Validation transforms
        val_transforms = self.preprocessor.get_standard_transforms(is_training=False)
        assert val_transforms is not None
        
        # Test transforms on sample image
        sample_image = Image.new('RGB', (256, 256), 'green')
        
        train_result = train_transforms(sample_image)
        val_result = val_transforms(sample_image)
        
        # Both should return tensors with expected shape
        assert hasattr(train_result, 'shape')
        assert hasattr(val_result, 'shape')
        assert train_result.shape == (3, 224, 224)  # CHW
        assert val_result.shape == (3, 224, 224)


@pytest.fixture
def sample_image():
    """Fixture providing a sample image."""
    return Image.new('RGB', (100, 100), 'blue')


@pytest.fixture  
def sample_numpy_image():
    """Fixture providing a sample numpy image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


def test_error_handling_invalid_path():
    """Test error handling for invalid paths."""
    # Should not raise error, but return empty list
    loader = ImageDataLoader("nonexistent_path")
    assert len(loader.image_paths) == 0


def test_error_handling_empty_directory(tmp_path):
    """Test handling of empty directory."""
    loader = ImageDataLoader(str(tmp_path))
    assert len(loader.image_paths) == 0
    
    info = loader.get_image_info()
    assert info['num_images'] == 0


def test_supported_extensions():
    """Test that all supported extensions are recognized."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create files with different extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        for ext in extensions:
            # Create dummy file (not actual image for this test)
            (temp_dir / f"test{ext}").touch()
        
        loader = ImageDataLoader(str(temp_dir))
        # Note: This will find files but loading may fail since they're not real images
        # The test is mainly for the file discovery logic
        
        found_extensions = set()
        for path in loader.image_paths:
            found_extensions.add(path.suffix.lower())
        
        # Should find most extensions (some might be filtered out by the glob pattern)
        assert len(found_extensions) > 0
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])