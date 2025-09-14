"""
Model evaluation and benchmarking script.
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from src.models.registry import ModelRegistry
from src.inference.pipeline import InferencePipeline
from src.data.loaders import ImageDataLoader
from src.utils.config import ConfigManager


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Vision AI Model Performance")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name to evaluate"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/samples",
        help="Path to test data directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--num-runs", "-n",
        type=int,
        default=3,
        help="Number of benchmark runs for averaging"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Configuration file path"
    )
    
    return parser.parse_args()


def evaluate_model_accuracy(pipeline: InferencePipeline, test_images: List[str]) -> Dict[str, Any]:
    """
    Evaluate model accuracy (placeholder implementation).
    
    Args:
        pipeline: Inference pipeline
        test_images: List of test image paths
        
    Returns:
        Dictionary with accuracy metrics
    """
    # This is a simplified accuracy evaluation
    # In a real scenario, you would have ground truth labels
    
    results = []
    successful_predictions = 0
    
    print(f"Processing {len(test_images)} images for accuracy evaluation...")
    
    for i, image_path in enumerate(test_images):
        try:
            result = pipeline.predict_single(image_path)
            
            # Check if prediction was successful
            if "top_class" in result and "top_confidence" in result:
                successful_predictions += 1
                results.append({
                    "image": Path(image_path).name,
                    "prediction": result["top_class"],
                    "confidence": result["top_confidence"],
                    "success": True
                })
            else:
                results.append({
                    "image": Path(image_path).name,
                    "error": "No valid prediction",
                    "success": False
                })
            
        except Exception as e:
            results.append({
                "image": Path(image_path).name,
                "error": str(e),
                "success": False
            })
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(test_images):
            print(f"Processed {i + 1}/{len(test_images)} images")
    
    success_rate = successful_predictions / len(test_images) if test_images else 0
    
    return {
        "total_images": len(test_images),
        "successful_predictions": successful_predictions,
        "failed_predictions": len(test_images) - successful_predictions,
        "success_rate": success_rate,
        "detailed_results": results
    }


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    print(f"Starting model evaluation for: {args.model}")
    print(f"Test data directory: {args.data}")
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Initialize model registry
    models_dir = config.get("models_dir", "models")
    registry = ModelRegistry(models_dir)
    
    # Create default models if none exist
    if len(registry) == 0:
        print("No models found. Creating default models...")
        registry.create_default_models()
    
    # Check if model exists
    if args.model not in registry:
        print(f"Error: Model '{args.model}' not found.")
        print(f"Available models: {registry.get_available_models()}")
        return
    
    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = registry.load_model(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create inference pipeline
    pipeline = InferencePipeline(model)
    
    # Load test data
    if not Path(args.data).exists():
        print(f"Error: Test data directory does not exist: {args.data}")
        return
    
    loader = ImageDataLoader(args.data)
    test_images = [str(path) for path in loader.image_paths]
    
    if not test_images:
        print("No test images found in the specified directory")
        return
    
    # Limit number of images if specified
    if args.max_images and args.max_images < len(test_images):
        test_images = test_images[:args.max_images]
        print(f"Limited to {args.max_images} images")
    
    print(f"Found {len(test_images)} test images")
    
    # Run evaluation
    evaluation_results = {
        "model_name": args.model,
        "model_info": model.get_model_info(),
        "test_data_path": args.data,
        "num_test_images": len(test_images),
        "evaluation_timestamp": time.time(),
    }
    
    # 1. Accuracy Evaluation
    print("\n=== Accuracy Evaluation ===")
    accuracy_results = evaluate_model_accuracy(pipeline, test_images)
    evaluation_results["accuracy"] = accuracy_results
    
    print(f"Success Rate: {accuracy_results['success_rate']:.2%}")
    print(f"Successful Predictions: {accuracy_results['successful_predictions']}/{accuracy_results['total_images']}")
    
    # 2. Performance Benchmark
    print("\n=== Performance Benchmark ===")
    try:
        benchmark_results = pipeline.benchmark_model(test_images, num_runs=args.num_runs)
        evaluation_results["performance"] = benchmark_results
        
        print(f"Average Inference Time: {benchmark_results['average_inference_time']:.4f} seconds")
        print(f"Images per Second: {benchmark_results['images_per_second']:.2f}")
        print(f"Min/Max Time: {benchmark_results['min_inference_time']:.4f}s / {benchmark_results['max_inference_time']:.4f}s")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        evaluation_results["performance"] = {"error": str(e)}
    
    # 3. Model Statistics
    print("\n=== Model Information ===")
    model_info = model.get_model_info()
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"Device: {model_info['device']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {output_path}")
    
    # Summary
    print(f"\nSummary for {args.model}:")
    print(f"  - Success Rate: {accuracy_results['success_rate']:.2%}")
    if 'performance' in evaluation_results and 'average_inference_time' in evaluation_results['performance']:
        avg_time = evaluation_results['performance']['average_inference_time']
        print(f"  - Average Inference Time: {avg_time:.4f}s")
        print(f"  - Throughput: {1.0/avg_time:.2f} images/second")


if __name__ == "__main__":
    main()