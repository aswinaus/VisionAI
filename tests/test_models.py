"""
Tests for model functionality.
"""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from src.models.base import BaseModel
from src.models.classification import ImageClassifier, CustomClassifier
from src.models.registry import ModelRegistry


class TestImageClassifier:
    """Tests for ImageClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "name": "test_classifier",
            "architecture": "resnet18",
            "num_classes": 10,
            "pretrained": False,  # Don't download weights for tests
            "class_names": [f"class_{i}" for i in range(10)]
        }
        self.model = ImageClassifier(self.config)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model_name == "test_classifier"
        assert self.model.num_classes == 10
        assert len(self.model.class_names) == 10
        assert self.model.architecture == "resnet18"
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        output = self.model.forward(x)
        
        assert output.shape == (batch_size, self.model.num_classes)
        assert torch.is_tensor(output)
    
    def test_preprocess(self):
        """Test image preprocessing."""
        # Create test image
        test_image = Image.new('RGB', (256, 256), color='red')
        
        processed = self.model.preprocess(test_image)
        
        assert torch.is_tensor(processed)
        assert processed.shape == (3, 224, 224)  # CHW format
        assert processed.dtype == torch.float32
    
    def test_predict(self):
        """Test end-to-end prediction."""
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        result = self.model.predict(test_image)
        
        assert isinstance(result, dict)
        assert "predictions" in result
        assert "top_class" in result
        assert "top_confidence" in result
        assert len(result["predictions"]) <= 5  # Top-5 predictions
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.model.get_model_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["num_classes"] == 10


class TestCustomClassifier:
    """Tests for CustomClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "name": "test_custom",
            "num_classes": 2,
            "input_size": (224, 224),
            "dropout_rate": 0.3,
            "hidden_size": 128
        }
        self.model = CustomClassifier(self.config)
    
    def test_custom_model_initialization(self):
        """Test custom model initialization."""
        assert self.model.model_name == "test_custom"
        assert self.model.num_classes == 2
        assert self.model.dropout_rate == 0.3
        assert self.model.hidden_size == 128
    
    def test_custom_forward_pass(self):
        """Test forward pass."""
        batch_size = 3
        x = torch.randn(batch_size, 3, 224, 224)
        
        output = self.model.forward(x)
        
        assert output.shape == (batch_size, 2)
        assert torch.is_tensor(output)
    
    def test_custom_predict(self):
        """Test prediction."""
        test_image = Image.new('RGB', (224, 224), color='green')
        
        result = self.model.predict(test_image)
        
        assert isinstance(result, dict)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "percentage" in result


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_models_dir = Path("test_models")
        self.temp_models_dir.mkdir(exist_ok=True)
        self.registry = ModelRegistry(str(self.temp_models_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_models_dir.exists():
            shutil.rmtree(self.temp_models_dir)
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry) == 0  # Should start empty
        assert self.registry.models_dir.exists()
    
    def test_register_model(self):
        """Test model registration."""
        config = {
            "name": "test_model",
            "type": "classification",
            "architecture": "resnet18",
            "num_classes": 5,
            "pretrained": False
        }
        
        self.registry.register_model("test_model", "classification", config)
        
        assert len(self.registry) == 1
        assert "test_model" in self.registry
        assert "test_model" in self.registry.get_available_models()
    
    def test_create_default_models(self):
        """Test creation of default models."""
        self.registry.create_default_models()
        
        available = self.registry.get_available_models()
        assert len(available) > 0
        assert "imagenet_resnet50" in available
        assert "binary_classifier" in available
        assert "multiclass_classifier" in available
    
    def test_get_model_info(self):
        """Test getting model information."""
        config = {
            "name": "info_test",
            "type": "classification",
            "architecture": "resnet18",
            "description": "Test model"
        }
        
        self.registry.register_model("info_test", "classification", config)
        info = self.registry.get_model_info("info_test")
        
        assert isinstance(info, dict)
        assert info["config"]["name"] == "info_test"
        assert info["config"]["description"] == "Test model"


@pytest.fixture
def sample_image():
    """Fixture providing a sample test image."""
    return Image.new('RGB', (224, 224), color='red')


@pytest.fixture
def sample_config():
    """Fixture providing a sample model configuration."""
    return {
        "name": "fixture_model",
        "architecture": "resnet18",
        "num_classes": 3,
        "pretrained": False,
        "class_names": ["class_a", "class_b", "class_c"]
    }


def test_model_device_handling(sample_config):
    """Test model device handling."""
    model = ImageClassifier(sample_config)
    
    # Test device property
    assert hasattr(model, 'device')
    
    # Test device switching (if CUDA available)
    if torch.cuda.is_available():
        model.to_device('cuda')
        assert 'cuda' in str(model.device)
    
    model.to_device('cpu')
    assert str(model.device) == 'cpu'


def test_model_save_load_config(sample_config, tmp_path):
    """Test model configuration saving and loading."""
    model = ImageClassifier(sample_config)
    
    # Save model (this tests the save functionality)
    save_path = tmp_path / "test_model.pt"
    model.save_model(str(save_path))
    
    assert save_path.exists()
    
    # Test YAML config file creation
    yaml_path = save_path.with_suffix('.yaml')
    assert yaml_path.exists()


def test_error_handling_invalid_image():
    """Test error handling for invalid image inputs."""
    config = {
        "name": "error_test",
        "architecture": "resnet18", 
        "num_classes": 2,
        "pretrained": False
    }
    model = ImageClassifier(config)
    
    # Test with invalid image input
    with pytest.raises(Exception):
        model.preprocess("invalid_image_data")


if __name__ == "__main__":
    pytest.main([__file__])