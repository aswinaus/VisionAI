"""
Inference pipeline for running predictions with vision models.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Union
from PIL import Image
import torch

from ..models.base import BaseModel
from ..data.loaders import ImageDataLoader


class InferencePipeline:
    """
    Pipeline for running inference with vision models.
    """
    
    def __init__(self, model: BaseModel):
        """
        Initialize inference pipeline.
        
        Args:
            model: Vision model instance
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode
        
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing prediction results
        """
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
        
        # Load and predict
        image = Image.open(image_path).convert('RGB')
        return self._predict_image(image, metadata={"image_path": str(image_path)})
    
    def predict_single_pil(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a PIL Image.
        
        Args:
            image: PIL Image instance
            
        Returns:
            Dictionary containing prediction results
        """
        return self._predict_image(image, metadata={"image_source": "PIL_Image"})
    
    def _predict_image(self, image: Image.Image, metadata: Dict = None) -> Dict[str, Any]:
        """
        Internal method to run prediction on PIL Image.
        
        Args:
            image: PIL Image instance
            metadata: Additional metadata to include in result
            
        Returns:
            Prediction result dictionary
        """
        start_time = time.time()
        
        # Run model prediction
        result = self.model.predict(image)
        
        inference_time = time.time() - start_time
        
        # Add metadata
        result.update({
            "inference_time": inference_time,
            "image_size": image.size,
            "timestamp": time.time(),
            **(metadata or {})
        })
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                error_result = {
                    "image_path": image_path,
                    "error": str(e),
                    "success": False,
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        return results
    
    def predict_batch_from_loader(self, loader: ImageDataLoader, max_images: int = None) -> List[Dict[str, Any]]:
        """
        Run inference using ImageDataLoader.
        
        Args:
            loader: ImageDataLoader instance
            max_images: Maximum number of images to process
            
        Returns:
            List of prediction results
        """
        image_paths = loader.image_paths
        if max_images:
            image_paths = image_paths[:max_images]
        
        return self.predict_batch([str(path) for path in image_paths])
    
    def predict_batch_with_progress(self, image_paths: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Run batch inference with progress tracking.
        
        Args:
            image_paths: List of image paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                error_result = {
                    "image_path": image_path,
                    "error": str(e),
                    "success": False,
                    "timestamp": time.time()
                }
                results.append(error_result)
            
            # Progress callback
            if progress_callback:
                progress = (i + 1) / total
                progress_callback(progress, i + 1, total)
        
        return results
    
    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save prediction result to file.
        
        Args:
            result: Prediction result dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save batch prediction results to file.
        
        Args:
            results: List of prediction results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create summary
        summary = {
            "total_images": len(results),
            "successful_predictions": sum(1 for r in results if r.get("success", True)),
            "failed_predictions": sum(1 for r in results if not r.get("success", True)),
            "average_inference_time": None,
            "results": results
        }
        
        # Calculate average inference time for successful predictions
        successful_times = [r["inference_time"] for r in results 
                           if "inference_time" in r and r.get("success", True)]
        if successful_times:
            summary["average_inference_time"] = sum(successful_times) / len(successful_times)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def benchmark_model(self, test_images: List[str], num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            test_images: List of test image paths
            num_runs: Number of runs for averaging
            
        Returns:
            Benchmark results
        """
        if not test_images:
            raise ValueError("No test images provided")
        
        all_times = []
        successful_runs = 0
        
        for run in range(num_runs):
            run_times = []
            
            for image_path in test_images:
                try:
                    start_time = time.time()
                    self.predict_single(image_path)
                    inference_time = time.time() - start_time
                    run_times.append(inference_time)
                except Exception:
                    continue
            
            if run_times:
                all_times.extend(run_times)
                successful_runs += 1
        
        if not all_times:
            return {"error": "No successful predictions in benchmark"}
        
        # Calculate statistics
        avg_time = sum(all_times) / len(all_times)
        min_time = min(all_times)
        max_time = max(all_times)
        
        return {
            "num_test_images": len(test_images),
            "num_runs": num_runs,
            "successful_runs": successful_runs,
            "total_predictions": len(all_times),
            "average_inference_time": avg_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "images_per_second": 1.0 / avg_time if avg_time > 0 else 0,
            "model_info": self.model.get_model_info()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model.get_model_info()
    
    def set_device(self, device: str) -> None:
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda', 'cpu')
        """
        self.model.to_device(device)