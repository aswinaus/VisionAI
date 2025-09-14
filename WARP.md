# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a vision AI showcase project designed to demonstrate computer vision capabilities. The project will include machine learning models, image processing pipelines, and demonstration interfaces.

## Common Development Commands

### Project Initialization
```powershell
# Initialize the project structure
mkdir src, tests, configs, data, models, notebooks, scripts, docs
mkdir data/raw, data/processed, data/samples
mkdir src/models, src/data, src/training, src/inference, src/utils, src/web

# Create initial requirements.txt
@"
torch>=1.11.0
torchvision>=0.12.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=6.2.0
black>=21.0.0
isort>=5.9.0
flake8>=3.9.0
mypy>=0.910
"@ | Out-File -FilePath requirements.txt -Encoding utf8

# Create setup.py for development installation
@"
from setuptools import setup, find_packages

setup(
    name="vision-ai-showcase",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
    ],
)
"@ | Out-File -FilePath setup.py -Encoding utf8
```

### Environment Setup
```powershell
# Create virtual environment (Python)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (after creating requirements.txt)
pip install -r requirements.txt

# Install in development mode (after creating setup.py)
pip install -e .
```

### Running the Application
```powershell
# Start the main application
python main.py

# Run with specific configuration
python main.py --config config/development.yaml

# Run inference on single image
python inference.py --image path/to/image.jpg --model models/best_model.pt

# Start web interface (if using Streamlit/Gradio)
streamlit run app.py
# or
python app.py
```

### Testing
```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/

# Test specific model performance
python scripts/evaluate_model.py --model models/model.pt --test-data data/test/
```

### Data Management
```powershell
# Download datasets
python scripts/download_data.py

# Process raw data
python scripts/preprocess_data.py --input data/raw --output data/processed

# Generate training/validation splits
python scripts/split_data.py --data data/processed --split 0.8/0.1/0.1

# Validate dataset integrity
python scripts/validate_data.py --data data/processed
```

### Training and Model Development
```powershell
# Start training
python train.py --config configs/training.yaml

# Resume training from checkpoint
python train.py --config configs/training.yaml --resume checkpoints/latest.pt

# Evaluate trained model
python evaluate.py --model models/best_model.pt --data data/test

# Export model for deployment
python scripts/export_model.py --model models/best_model.pt --format onnx
```

### Linting and Formatting
```powershell
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

### Docker Operations
```powershell
# Build Docker image
docker build -t vision-ai-showcase .

# Run container locally
docker run -p 8000:8000 -v ${PWD}/data:/app/data vision-ai-showcase

# Run with GPU support
docker run --gpus all -p 8000:8000 vision-ai-showcase

# Build and run with docker-compose
docker-compose up --build
```

## Architecture Overview

### Project Structure
```
├── src/                    # Source code
│   ├── models/            # Model definitions and architectures
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training loops and utilities
│   ├── inference/         # Inference pipelines
│   ├── utils/             # Utility functions
│   └── web/               # Web interface components
├── tests/                 # Test files
├── configs/               # Configuration files (YAML/JSON)
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed data
│   └── samples/          # Sample images for testing
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── .gitignore           # Git ignore patterns
└── WARP.md              # This file
```

### Key Components

#### Model Layer
- **Model Definitions**: Neural network architectures (CNN, Vision Transformers, etc.)
- **Model Registry**: Centralized model management and versioning
- **Model Loaders**: Utilities for loading pre-trained and custom models

#### Data Pipeline
- **Data Loaders**: Efficient data loading with transforms and augmentations
- **Preprocessing**: Image normalization, resizing, and format conversion
- **Augmentation**: Training-time data augmentation strategies

#### Inference Pipeline
- **Batch Processing**: Handle multiple images efficiently
- **Real-time Processing**: Single image inference with minimal latency
- **Post-processing**: Result formatting and visualization

#### Web Interface
- **Upload Interface**: File upload and drag-drop functionality
- **Result Display**: Image visualization with predictions/annotations
- **Model Selection**: Switch between different trained models

## Development Guidelines

### Model Development
- Store model configurations in `configs/` directory
- Save checkpoints in `models/` with descriptive names
- Include model metadata (accuracy, training date, dataset) in filenames
- Use consistent input/output formats across models

### Data Handling
- Keep raw and processed data separate
- Document data sources and preprocessing steps
- Implement data validation before training
- Use consistent image formats and naming conventions

### Code Organization
- Separate model training from inference code
- Create reusable components for common operations
- Use configuration files for hyperparameters
- Implement proper error handling for file I/O operations

### Performance Considerations
- Batch processing for multiple images
- Model optimization (quantization, pruning) for deployment
- Efficient memory usage for large images
- GPU utilization when available

### Testing Strategy
- Unit tests for individual components
- Integration tests for full pipelines
- Model performance regression tests
- Data pipeline validation tests

## Configuration Management

### Environment Variables
```powershell
# Model paths
$env:MODEL_PATH = "models/"
$env:DATA_PATH = "data/"

# Training settings
$env:BATCH_SIZE = "32"
$env:LEARNING_RATE = "0.001"

# Inference settings
$env:DEVICE = "cuda"  # or "cpu"
$env:MAX_IMAGE_SIZE = "1024"
```

### Config File Structure
Configuration files should follow this pattern:
- `configs/training.yaml` - Training hyperparameters
- `configs/model.yaml` - Model architecture settings  
- `configs/data.yaml` - Dataset and preprocessing settings
- `configs/inference.yaml` - Inference pipeline settings

### Getting Started Commands
```powershell
# Create .gitignore file
@"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
venv/
env/
.env

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pt
models/*.pth
models/*.onnx
!models/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding utf8

# Create placeholder files to maintain directory structure
New-Item -Path "data/raw/.gitkeep" -ItemType File -Force
New-Item -Path "data/processed/.gitkeep" -ItemType File -Force
New-Item -Path "models/.gitkeep" -ItemType File -Force
```

## Troubleshooting

### Common Issues
- **GPU Memory**: Reduce batch size or image resolution
- **Missing Dependencies**: Check `requirements.txt` versions
- **Data Path Issues**: Use absolute paths in configuration files
- **Model Loading**: Verify model file format and compatibility

### Debug Commands
```powershell
# Check Python environment
python --version
pip list | Select-String "torch"

# Check GPU availability (after installing PyTorch)
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Validate project structure
Get-ChildItem -Path . -Directory | Select-Object Name

# Test imports (after creating source files)
python -c "import src; print('Source imports working')"
```
