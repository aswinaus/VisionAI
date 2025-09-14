"""
Image classification models for various computer vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Union
from .base import BaseModel


class ImageClassifier(BaseModel):
    """
    General-purpose image classification model supporting multiple architectures.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize image classifier.
        
        Args:
            model_config: Configuration dictionary containing:
                - architecture: Model architecture name (e.g., 'resnet50', 'efficientnet_b0')
                - num_classes: Number of output classes
                - pretrained: Whether to use pretrained weights
                - class_names: List of class names
        """
        super().__init__(model_config)
        
        self.architecture = model_config.get("architecture", "resnet50")
        self.pretrained = model_config.get("pretrained", True)
        self.class_names = model_config.get("class_names", [])
        
        # Build model
        self.model = self._build_model()
        
        # Define preprocessing transforms
        self.transform = self._get_transforms()
    
    def _build_model(self) -> nn.Module:
        """Build the classification model based on architecture."""
        if self.architecture == "resnet50":
            model = models.resnet50(pretrained=self.pretrained)
            if self.num_classes != 1000:
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        elif self.architecture == "resnet18":
            model = models.resnet18(pretrained=self.pretrained)
            if self.num_classes != 1000:
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        elif self.architecture == "efficientnet_b0":
            try:
                model = models.efficientnet_b0(pretrained=self.pretrained)
                if self.num_classes != 1000:
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            except AttributeError:
                # Fallback to ResNet if EfficientNet not available
                print("EfficientNet not available, falling back to ResNet50")
                model = models.resnet50(pretrained=self.pretrained)
                if self.num_classes != 1000:
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        elif self.architecture == "vgg16":
            model = models.vgg16(pretrained=self.pretrained)
            if self.num_classes != 1000:
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
        
        else:
            # Default to ResNet50
            print(f"Architecture {self.architecture} not recognized, using ResNet50")
            model = models.resnet50(pretrained=self.pretrained)
            if self.num_classes != 1000:
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        return model
    
    def _get_transforms(self) -> transforms.Compose:
        """Get preprocessing transforms for the model."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def preprocess(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        Preprocess input image for classification.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            
        Returns:
            Preprocessed tensor
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8 and image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            image = Image.fromarray((image * 255).astype(np.uint8))
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return self.transform(image)
    
    def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        """
        Postprocess model output into classification results.
        
        Args:
            output: Raw model output tensor
            
        Returns:
            Dictionary containing classification results
        """
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get top-k predictions
        top_k = min(5, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        
        # Convert to numpy for easier handling
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Prepare results
        predictions = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
            predictions.append({
                "rank": i + 1,
                "class_id": int(idx),
                "class_name": class_name,
                "confidence": float(prob),
                "percentage": float(prob * 100)
            })
        
        return {
            "predictions": predictions,
            "top_class": predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "model_name": self.model_name,
            "architecture": self.architecture
        }


class CustomClassifier(BaseModel):
    """
    Custom CNN classifier for specialized tasks.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize custom classifier."""
        super().__init__(model_config)
        
        self.dropout_rate = model_config.get("dropout_rate", 0.5)
        self.hidden_size = model_config.get("hidden_size", 512)
        
        # Build custom architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through custom model."""
        features = self.features(x)
        output = self.classifier(features)
        return output
    
    def preprocess(self, image) -> torch.Tensor:
        """Preprocess image using standard transforms."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        """Postprocess model output."""
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        class_name = (self.config.get("class_names", [])[predicted_class] 
                     if predicted_class < len(self.config.get("class_names", []))
                     else f"Class_{predicted_class}")
        
        return {
            "predicted_class": class_name,
            "class_id": predicted_class,
            "confidence": confidence,
            "percentage": confidence * 100,
            "model_name": self.model_name
        }