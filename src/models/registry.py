"""
Model registry for managing and organizing different vision models.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import importlib
from .base import BaseModel
from .classification import ImageClassifier, CustomClassifier


class ModelRegistry:
    """
    Central registry for managing vision models in the showcase.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model registry.
        
        Args:
            models_dir: Directory containing model files and configurations
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_classes: Dict[str, Type[BaseModel]] = {
            "classification": ImageClassifier,
            "custom_classification": CustomClassifier,
        }
        self._load_model_configs()
    
    def _load_model_configs(self) -> None:
        """Load all model configurations from the models directory."""
        config_files = list(self.models_dir.glob("*.yaml")) + list(self.models_dir.glob("*.yml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                model_name = config.get("name", config_file.stem)
                self._models[model_name] = {
                    "config": config,
                    "config_path": str(config_file),
                    "model_path": str(config_file.with_suffix(".pt")),
                    "loaded": False,
                    "instance": None
                }
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")
    
    def register_model(self, name: str, model_type: str, config: Dict[str, Any], 
                      model_path: Optional[str] = None) -> None:
        """
        Register a new model in the registry.
        
        Args:
            name: Unique model name
            model_type: Type of model (e.g., 'classification', 'detection')
            config: Model configuration dictionary
            model_path: Optional path to saved model weights
        """
        if model_type not in self._model_classes:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(self._model_classes.keys())}")
        
        config["name"] = name
        config["type"] = model_type
        
        # Save configuration
        config_path = self.models_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Register in memory
        self._models[name] = {
            "config": config,
            "config_path": str(config_path),
            "model_path": model_path or str(self.models_dir / f"{name}.pt"),
            "loaded": False,
            "instance": None,
            "type": model_type
        }
    
    def load_model(self, name: str, force_reload: bool = False) -> BaseModel:
        """
        Load a model by name.
        
        Args:
            name: Model name
            force_reload: Whether to force reload if already loaded
            
        Returns:
            Loaded model instance
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found in registry. "
                           f"Available models: {list(self._models.keys())}")
        
        model_info = self._models[name]
        
        # Return cached instance if already loaded
        if model_info["loaded"] and not force_reload:
            return model_info["instance"]
        
        # Load model
        config = model_info["config"]
        model_type = config.get("type", "classification")
        
        if model_type not in self._model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create model instance
        model_class = self._model_classes[model_type]
        model = model_class(config)
        
        # Load weights if available
        model_path = model_info["model_path"]
        if os.path.exists(model_path):
            try:
                model.load_model(model_path)
                print(f"Loaded weights for model '{name}' from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load weights for '{name}': {e}")
        
        # Cache loaded model
        model_info["instance"] = model
        model_info["loaded"] = True
        
        return model
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary containing model information
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        model_info = self._models[name].copy()
        
        # Add runtime info if model is loaded
        if model_info["loaded"]:
            model = model_info["instance"]
            runtime_info = model.get_model_info()
            model_info.update(runtime_info)
        
        return model_info
    
    def unload_model(self, name: str) -> None:
        """
        Unload a model to free memory.
        
        Args:
            name: Model name to unload
        """
        if name in self._models and self._models[name]["loaded"]:
            self._models[name]["instance"] = None
            self._models[name]["loaded"] = False
            print(f"Unloaded model '{name}'")
    
    def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        for name in self._models:
            self.unload_model(name)
    
    def create_default_models(self) -> None:
        """Create default model configurations for demonstration."""
        
        # Default ImageNet classifier
        imagenet_config = {
            "name": "imagenet_resnet50",
            "type": "classification",
            "architecture": "resnet50",
            "num_classes": 1000,
            "pretrained": True,
            "input_size": [224, 224],
            "description": "ResNet-50 trained on ImageNet dataset",
            "class_names": []  # Will be loaded from ImageNet classes
        }
        
        # Custom binary classifier example
        binary_config = {
            "name": "binary_classifier",
            "type": "custom_classification",
            "num_classes": 2,
            "input_size": [224, 224],
            "dropout_rate": 0.3,
            "hidden_size": 256,
            "description": "Custom binary classifier example",
            "class_names": ["Class_0", "Class_1"]
        }
        
        # Small multi-class classifier
        multiclass_config = {
            "name": "multiclass_classifier",
            "type": "classification",
            "architecture": "resnet18",
            "num_classes": 10,
            "pretrained": True,
            "input_size": [224, 224],
            "description": "ResNet-18 for 10-class classification",
            "class_names": [f"Class_{i}" for i in range(10)]
        }
        
        # Register default models
        for config in [imagenet_config, binary_config, multiclass_config]:
            if config["name"] not in self._models:
                self.register_model(
                    name=config["name"],
                    model_type=config["type"],
                    config=config
                )
        
        print("Created default model configurations")
    
    def add_model_class(self, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Add a new model class to the registry.
        
        Args:
            model_type: String identifier for the model type
            model_class: Class that inherits from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
        
        self._model_classes[model_type] = model_class
        print(f"Registered model type '{model_type}'")
    
    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._models)
    
    def __contains__(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._models
    
    def __str__(self) -> str:
        """String representation of the registry."""
        if not self._models:
            return "ModelRegistry: No models registered"
        
        lines = ["ModelRegistry:"]
        for name, info in self._models.items():
            status = "loaded" if info["loaded"] else "not loaded"
            model_type = info["config"].get("type", "unknown")
            lines.append(f"  - {name} ({model_type}): {status}")
        
        return "\n".join(lines)