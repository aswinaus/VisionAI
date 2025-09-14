"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Configuration manager for loading and managing application settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        # Default configuration
        default_config = {
            "models_dir": "models",
            "data_dir": "data", 
            "output_dir": "outputs",
            "logging": {
                "level": "INFO",
                "file": "logs/vision_ai.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "inference": {
                "batch_size": 32,
                "device": "auto",  # auto, cuda, cpu
                "max_image_size": 1024
            },
            "web": {
                "port": 8501,
                "host": "localhost",
                "title": "Vision AI Showcase"
            }
        }
        
        # Load from file if specified
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge with defaults (file config takes precedence)
                config = self._merge_configs(default_config, file_config)
                return config
                
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            default: Default configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Output file path (uses original path if None)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        if not save_path:
            raise ValueError("No output path specified and no original config path available")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update_from_env(self, prefix: str = "VISION_AI_") -> None:
        """
        Update configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                self.set(config_key, value)
    
    def validate_config(self) -> Dict[str, str]:
        """
        Validate configuration and return any issues.
        
        Returns:
            Dictionary of validation issues (empty if valid)
        """
        issues = {}
        
        # Check required directories
        for dir_key in ['models_dir', 'data_dir', 'output_dir']:
            dir_path = self.get(dir_key)
            if dir_path:
                path = Path(dir_path)
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        issues[dir_key] = f"Cannot create directory {dir_path}: {e}"
        
        # Check logging configuration
        log_file = self.get('logging.file')
        if log_file:
            log_path = Path(log_file)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues['logging.file'] = f"Cannot create log directory: {e}"
        
        # Validate inference settings
        device = self.get('inference.device', 'auto')
        if device not in ['auto', 'cuda', 'cpu']:
            issues['inference.device'] = f"Invalid device '{device}', must be 'auto', 'cuda', or 'cpu'"
        
        batch_size = self.get('inference.batch_size', 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            issues['inference.batch_size'] = "Batch size must be a positive integer"
        
        return issues
    
    def create_sample_config(self, output_path: str) -> None:
        """
        Create a sample configuration file with documentation.
        
        Args:
            output_path: Path for the sample config file
        """
        sample_config = """# Vision AI Showcase Configuration File

# Directory settings
models_dir: "models"        # Directory containing model files
data_dir: "data"           # Directory containing datasets
output_dir: "outputs"      # Directory for output files

# Logging configuration
logging:
  level: "INFO"            # Log level: DEBUG, INFO, WARNING, ERROR
  file: "logs/vision_ai.log"  # Log file path
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Inference settings
inference:
  batch_size: 32           # Default batch size for inference
  device: "auto"           # Device to use: auto, cuda, cpu
  max_image_size: 1024     # Maximum image size for processing

# Web interface settings
web:
  port: 8501              # Port for Streamlit app
  host: "localhost"       # Host for web server
  title: "Vision AI Showcase"  # Application title

# Model-specific settings
models:
  default_timeout: 30     # Timeout for model loading (seconds)
  cache_size: 3           # Number of models to keep in memory
  
# Data processing settings
data:
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
  max_batch_size: 100     # Maximum batch size for data loading
  preprocessing:
    resize_method: "lanczos"  # Resize method: lanczos, bilinear, nearest
    normalize: true       # Whether to normalize images
"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(sample_config)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(config_path={self.config_path}, keys={list(self.config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ConfigManager(config_path={self.config_path}, config={self.config})"