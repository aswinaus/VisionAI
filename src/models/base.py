"""
Base model class providing common functionality for all vision models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all vision models in the showcase.
    
    Provides common functionality for model loading, saving, and inference.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            model_config: Dictionary containing model configuration
        """
        super().__init__()
        self.config = model_config
        self.model_name = model_config.get("name", "unknown")
        self.num_classes = model_config.get("num_classes", 1000)
        self.input_size = model_config.get("input_size", (224, 224))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output tensor
        """
        pass
    
    @abstractmethod
    def preprocess(self, image) -> torch.Tensor:
        """
        Preprocess input image for the model.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        pass
    
    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        """
        Postprocess model output into human-readable format.
        
        Args:
            output: Raw model output tensor
            
        Returns:
            Dictionary containing processed results
        """
        pass
    
    def predict(self, image) -> Dict[str, Any]:
        """
        End-to-end prediction pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Prediction results
        """
        self.eval()
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess(image)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            input_tensor = input_tensor.to(self.device)
            
            # Forward pass
            output = self.forward(input_tensor)
            
            # Postprocess
            result = self.postprocess(output)
            
        return result
    
    def save_model(self, path: str, save_config: bool = True) -> None:
        """
        Save model state and configuration.
        
        Args:
            path: Path to save the model
            save_config: Whether to save model configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name,
        }, path)
        
        # Optionally save config separately
        if save_config:
            config_path = path.with_suffix('.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def load_model(self, path: str) -> None:
        """
        Load model state from file.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config.update(checkpoint.get('config', {}))
        self.model_name = checkpoint.get('model_name', self.model_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "config": self.config
        }
    
    def to_device(self, device: Optional[str] = None) -> 'BaseModel':
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda', 'cpu', or None for auto)
            
        Returns:
            Self for method chaining
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.to(self.device)
        return self