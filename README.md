# Vision AI Showcase

A comprehensive computer vision demonstration showcasing various AI models and capabilities for image classification, object detection, and more.

## 🌟 Features

- **Multiple Model Support**: Classification, detection, and segmentation models
- **Web Interface**: Interactive Streamlit web application 
- **Command Line Interface**: Powerful CLI for batch processing
- **Model Registry**: Centralized model management and configuration
- **Flexible Data Loading**: Support for various image formats and dataset structures
- **Real-time Inference**: Fast prediction pipelines with GPU acceleration
- **Extensible Architecture**: Easy to add new models and capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for acceleration

### Installation

1. **Clone the repository** (or download and extract):
   ```powershell
   git clone <repository-url>
   cd vision-ai-showcase
   ```

2. **Create virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Initialize the project**:
   ```powershell
   python main.py --mode setup
   ```

### Running the Application

#### Web Interface (Recommended)
```powershell
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

#### Command Line Interface
```powershell
# Interactive demo mode
python main.py --mode demo

# Run inference on a single image
python main.py --mode inference --input path/to/image.jpg --model imagenet_resnet50

# Process a directory of images
python main.py --mode inference --input data/samples --output results
```

## 📁 Project Structure

```
vision-ai-showcase/
├── src/                    # Source code
│   ├── models/            # Model definitions and architectures
│   │   ├── base.py        # Base model class
│   │   ├── classification.py  # Classification models
│   │   ├── detection.py   # Object detection models
│   │   ├── segmentation.py # Image segmentation models
│   │   └── registry.py    # Model registry
│   ├── data/              # Data loading and preprocessing
│   │   ├── loaders.py     # Data loaders
│   │   ├── preprocessors.py # Image preprocessing
│   │   └── augmentation.py # Data augmentation
│   ├── inference/         # Inference pipelines
│   │   └── pipeline.py    # Main inference pipeline
│   ├── utils/             # Utility functions
│   │   ├── config.py      # Configuration management
│   │   └── logging.py     # Logging utilities
│   └── web/               # Web interface components
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed data
│   └── samples/          # Sample images for testing
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── main.py              # Main CLI application
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎯 Usage Examples

### Web Interface

1. Launch the web app: `streamlit run app.py`
2. Select a model from the sidebar
3. Upload an image or use sample images
4. Click "Analyze Image" to see results
5. Download results as JSON

### Command Line

#### Basic Inference
```powershell
# Classify a single image
python main.py --mode inference --input cat.jpg --model imagenet_resnet50

# Process multiple images
python main.py --mode inference --input ./images/ --model multiclass_classifier
```

#### Interactive Demo
```powershell
python main.py --mode demo
```

#### Configuration
```powershell
# Use custom configuration
python main.py --config configs/custom.yaml --mode inference --input image.jpg
```

## 🔧 Configuration

The application uses YAML configuration files located in the `configs/` directory.

### Key Configuration Options

```yaml
# models_dir: Directory for model files
models_dir: "models"

# data_dir: Directory for datasets  
data_dir: "data"

# inference: Inference settings
inference:
  batch_size: 32
  device: "auto"  # auto, cuda, cpu
  max_image_size: 1024

# logging: Logging configuration
logging:
  level: "INFO"
  file: "logs/vision_ai.log"
```

## 🧠 Models

The showcase supports multiple model types:

### Built-in Models

1. **ImageNet ResNet-50**: Pre-trained classification model
2. **Multi-class Classifier**: ResNet-18 for custom classes
3. **Binary Classifier**: Custom CNN for binary classification

### Adding Custom Models

1. Create a model class inheriting from `BaseModel`
2. Implement required methods: `forward()`, `preprocess()`, `postprocess()`
3. Register the model in the registry
4. Create a configuration YAML file

Example:
```python
from src.models.base import BaseModel

class MyCustomModel(BaseModel):
    def forward(self, x):
        # Your model implementation
        pass
    
    def preprocess(self, image):
        # Image preprocessing
        pass
    
    def postprocess(self, output):
        # Result processing
        pass
```

## 📊 Data Management

### Supported Formats
- JPEG, PNG, BMP, TIFF, WebP
- RGB and grayscale images
- Various resolutions (automatically resized)

### Data Organization

#### Option 1: Folder Structure (for classification)
```
data/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
└── class2/
    ├── image3.jpg
    └── image4.jpg
```

#### Option 2: CSV Labels
```csv
image,label
image1.jpg,cat
image2.jpg,dog
image3.jpg,cat
```

#### Option 3: Mixed Images (for general inference)
```
data/samples/
├── image1.jpg
├── image2.png
└── image3.bmp
```

## 🚀 Development

### Adding New Features

1. **New Model Type**: Extend `BaseModel` and add to registry
2. **New Data Loader**: Implement in `src/data/loaders.py`
3. **New Preprocessing**: Add to `src/data/preprocessors.py`
4. **New Web Features**: Modify `app.py`

### Testing

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```powershell
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Detected
```powershell
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

#### Memory Issues
- Reduce batch size in configuration
- Use CPU instead of GPU: set `device: "cpu"` in config
- Resize images to smaller dimensions

#### Model Loading Errors
- Ensure model files exist in `models/` directory
- Check model configuration YAML files
- Verify model compatibility with PyTorch version

#### Import Errors
```powershell
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Debug Mode

```powershell
# Enable verbose logging
python main.py --verbose --mode demo

# Check system info
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Linux
- **Python**: 3.8+
- **RAM**: 8GB (4GB for CPU-only)
- **Storage**: 2GB free space

### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **Storage**: 10GB+ free space (for models and data)

## 📚 API Reference

### ModelRegistry
```python
from src.models.registry import ModelRegistry

registry = ModelRegistry("models/")
model = registry.load_model("imagenet_resnet50")
```

### InferencePipeline
```python
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline(model)
result = pipeline.predict_single("image.jpg")
```

### ConfigManager
```python
from src.utils.config import ConfigManager

config = ConfigManager("configs/default.yaml")
batch_size = config.get("inference.batch_size", 32)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `black`, `isort`, `flake8`, `pytest`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the web interface framework
- The computer vision community for inspiration and resources

## 📞 Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. Join our community discussions

---

**Happy coding! 🚀**
