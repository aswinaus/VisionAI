"""
Main entry point for the Vision AI Showcase application.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from src.models.registry import ModelRegistry
from src.data.loaders import ImageDataLoader
from src.inference.pipeline import InferencePipeline
from src.utils.config import ConfigManager
from src.utils.logging import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vision AI Showcase Application")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="imagenet_resnet50",
        help="Model name to use for inference"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input image or directory path"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "demo", "setup"],
        default="demo",
        help="Application mode"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_models(config: Dict[str, Any]) -> ModelRegistry:
    """Initialize and setup model registry."""
    models_dir = config.get("models_dir", "models")
    registry = ModelRegistry(models_dir)
    
    # Create default models if none exist
    if len(registry) == 0:
        print("No models found. Creating default model configurations...")
        registry.create_default_models()
    
    return registry


def run_inference_mode(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run inference on input images."""
    if not args.input:
        print("Error: --input is required for inference mode")
        sys.exit(1)
    
    # Setup model registry
    registry = setup_models(config)
    
    # Load model
    try:
        model = registry.load_model(args.model)
        print(f"Loaded model: {args.model}")
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        print(f"Available models: {registry.get_available_models()}")
        sys.exit(1)
    
    # Setup inference pipeline
    pipeline = InferencePipeline(model)
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")
        result = pipeline.predict_single(str(input_path))
        
        # Save result
        output_file = output_dir / f"{input_path.stem}_result.json"
        pipeline.save_result(result, str(output_file))
        print(f"Result saved to: {output_file}")
        
    elif input_path.is_dir():
        # Batch inference
        print(f"Processing directory: {input_path}")
        loader = ImageDataLoader(str(input_path))
        results = pipeline.predict_batch_from_loader(loader)
        
        # Save results
        output_file = output_dir / "batch_results.json"
        pipeline.save_results(results, str(output_file))
        print(f"Results saved to: {output_file}")
        
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)


def run_demo_mode(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run interactive demo mode."""
    print("Starting Vision AI Showcase Demo...")
    
    # Setup model registry
    registry = setup_models(config)
    
    print(f"Available models: {registry.get_available_models()}")
    
    # Simple interactive loop
    while True:
        print("\n" + "="*50)
        print("Vision AI Showcase Demo")
        print("="*50)
        print("1. List available models")
        print("2. Load and test a model")
        print("3. Process sample images")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            models = registry.get_available_models()
            print(f"\nAvailable models: {models}")
            for model_name in models:
                info = registry.get_model_info(model_name)
                print(f"  - {model_name}: {info['config'].get('description', 'No description')}")
        
        elif choice == "2":
            model_name = input("Enter model name: ").strip()
            try:
                model = registry.load_model(model_name)
                print(f"Successfully loaded model: {model_name}")
                model_info = model.get_model_info()
                print(f"Model info: {model_info}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        elif choice == "3":
            # Process sample images if available
            sample_dir = Path("data/samples")
            if sample_dir.exists():
                print(f"Processing images from: {sample_dir}")
                # This would process sample images
                loader = ImageDataLoader(str(sample_dir))
                info = loader.get_image_info()
                print(f"Found {info['num_images']} sample images")
            else:
                print("No sample images directory found at data/samples")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


def run_setup_mode(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run setup mode to initialize the project."""
    print("Setting up Vision AI Showcase...")
    
    # Create necessary directories
    directories = ["models", "data/samples", "outputs", "logs"]
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        print(f"Created directory: {dir_path}")
    
    # Setup model registry and create default models
    registry = setup_models(config)
    print("Model registry initialized")
    
    # Create sample config files if they don't exist
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    default_config_path = config_dir / "default.yaml"
    if not default_config_path.exists():
        default_config = {
            "models_dir": "models",
            "data_dir": "data",
            "output_dir": "outputs",
            "logging": {
                "level": "INFO",
                "file": "logs/vision_ai.log"
            }
        }
        with open(default_config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default config: {default_config_path}")
    
    print("Setup completed successfully!")


def main() -> None:
    """Main application entry point."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(verbose=args.verbose)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    logger.info(f"Starting Vision AI Showcase in {args.mode} mode")
    
    try:
        if args.mode == "inference":
            run_inference_mode(args, config)
        elif args.mode == "demo":
            run_demo_mode(args, config)
        elif args.mode == "setup":
            run_setup_mode(args, config)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()