"""
Data validation and analysis script for Vision AI datasets.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from src.data.loaders import ImageDataLoader, DatasetLoader


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate and analyze Vision AI datasets")
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data_validation_report.json",
        help="Output file for validation report"
    )
    
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Perform detailed image validation (slower)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of images to analyze in detail"
    )
    
    return parser.parse_args()


def validate_image_file(image_path: Path) -> Dict[str, Any]:
    """
    Validate a single image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Validation results dictionary
    """
    result = {
        "path": str(image_path),
        "exists": image_path.exists(),
        "readable": False,
        "valid_image": False,
        "size": None,
        "mode": None,
        "format": None,
        "file_size_bytes": None,
        "errors": []
    }
    
    if not result["exists"]:
        result["errors"].append("File does not exist")
        return result
    
    try:
        result["file_size_bytes"] = image_path.stat().st_size
        
        with Image.open(image_path) as img:
            result["readable"] = True
            result["valid_image"] = True
            result["size"] = img.size  # (width, height)
            result["mode"] = img.mode
            result["format"] = img.format
            
            # Check for potential issues
            if img.size[0] < 32 or img.size[1] < 32:
                result["errors"].append(f"Very small image: {img.size}")
            
            if img.mode not in ['RGB', 'RGBA', 'L', 'LA']:
                result["errors"].append(f"Unusual color mode: {img.mode}")
                
            if result["file_size_bytes"] < 1000:  # < 1KB
                result["errors"].append("Very small file size")
                
    except Exception as e:
        result["errors"].append(f"Image loading error: {str(e)}")
    
    return result


def analyze_dataset_structure(data_path: Path) -> Dict[str, Any]:
    """
    Analyze the structure of a dataset directory.
    
    Args:
        data_path: Path to dataset directory
        
    Returns:
        Analysis results dictionary
    """
    analysis = {
        "path": str(data_path),
        "exists": data_path.exists(),
        "is_directory": data_path.is_dir(),
        "total_files": 0,
        "image_files": 0,
        "subdirectories": 0,
        "file_extensions": defaultdict(int),
        "directory_structure": {},
        "potential_classes": [],
        "structure_type": "unknown"
    }
    
    if not analysis["exists"] or not analysis["is_directory"]:
        return analysis
    
    # Analyze directory structure
    subdirs = []
    all_files = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            subdirs.append(item.name)
            analysis["subdirectories"] += 1
            
            # Count files in subdirectory
            subdir_files = []
            for subitem in item.iterdir():
                if subitem.is_file():
                    subdir_files.append(subitem.name)
                    all_files.append(subitem)
                    
            analysis["directory_structure"][item.name] = {
                "file_count": len(subdir_files),
                "sample_files": subdir_files[:5]  # First 5 files as samples
            }
            
        elif item.is_file():
            all_files.append(item)
    
    analysis["total_files"] = len(all_files)
    
    # Count file extensions and image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    for file_path in all_files:
        ext = file_path.suffix.lower()
        analysis["file_extensions"][ext] += 1
        
        if ext in image_extensions:
            analysis["image_files"] += 1
    
    # Determine structure type
    if subdirs:
        analysis["potential_classes"] = sorted(subdirs)
        if len(subdirs) >= 2:
            analysis["structure_type"] = "classification_folders"
        else:
            analysis["structure_type"] = "single_folder_with_subdir"
    else:
        if analysis["image_files"] > 0:
            analysis["structure_type"] = "flat_image_directory"
        else:
            analysis["structure_type"] = "no_images_found"
    
    return analysis


def compute_image_statistics(image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics from image validation results.
    
    Args:
        image_results: List of image validation results
        
    Returns:
        Statistics dictionary
    """
    valid_images = [r for r in image_results if r["valid_image"]]
    
    if not valid_images:
        return {"error": "No valid images found"}
    
    # Extract dimensions
    widths = [r["size"][0] for r in valid_images if r["size"]]
    heights = [r["size"][1] for r in valid_images if r["size"]]
    file_sizes = [r["file_size_bytes"] for r in valid_images if r["file_size_bytes"]]
    
    # Color modes
    modes = [r["mode"] for r in valid_images if r["mode"]]
    mode_counts = defaultdict(int)
    for mode in modes:
        mode_counts[mode] += 1
    
    # File formats
    formats = [r["format"] for r in valid_images if r["format"]]
    format_counts = defaultdict(int)
    for fmt in formats:
        format_counts[fmt] += 1
    
    stats = {
        "total_valid_images": len(valid_images),
        "dimensions": {
            "width": {
                "min": min(widths) if widths else 0,
                "max": max(widths) if widths else 0,
                "mean": np.mean(widths) if widths else 0,
                "std": np.std(widths) if widths else 0
            },
            "height": {
                "min": min(heights) if heights else 0,
                "max": max(heights) if heights else 0,
                "mean": np.mean(heights) if heights else 0,
                "std": np.std(heights) if heights else 0
            }
        },
        "file_size_bytes": {
            "min": min(file_sizes) if file_sizes else 0,
            "max": max(file_sizes) if file_sizes else 0,
            "mean": np.mean(file_sizes) if file_sizes else 0,
            "std": np.std(file_sizes) if file_sizes else 0
        },
        "color_modes": dict(mode_counts),
        "file_formats": dict(format_counts),
        "aspect_ratios": []
    }
    
    # Calculate aspect ratios
    if widths and heights:
        aspect_ratios = [w/h for w, h in zip(widths, heights)]
        stats["aspect_ratios"] = {
            "min": min(aspect_ratios),
            "max": max(aspect_ratios),
            "mean": np.mean(aspect_ratios),
            "std": np.std(aspect_ratios)
        }
    
    return stats


def main():
    """Main validation function."""
    args = parse_arguments()
    
    print(f"Starting data validation for: {args.data}")
    
    data_path = Path(args.data)
    
    # Initialize validation report
    validation_report = {
        "data_path": str(data_path),
        "validation_timestamp": None,
        "dataset_analysis": {},
        "image_validation": {},
        "statistics": {},
        "recommendations": [],
        "errors": []
    }
    
    import time
    validation_report["validation_timestamp"] = time.time()
    
    # 1. Analyze dataset structure
    print("\n=== Dataset Structure Analysis ===")
    structure_analysis = analyze_dataset_structure(data_path)
    validation_report["dataset_analysis"] = structure_analysis
    
    print(f"Structure Type: {structure_analysis['structure_type']}")
    print(f"Total Files: {structure_analysis['total_files']}")
    print(f"Image Files: {structure_analysis['image_files']}")
    print(f"Subdirectories: {structure_analysis['subdirectories']}")
    
    if structure_analysis["potential_classes"]:
        print(f"Potential Classes: {structure_analysis['potential_classes']}")
    
    # File extensions summary
    if structure_analysis["file_extensions"]:
        print("File Extensions:")
        for ext, count in structure_analysis["file_extensions"].items():
            print(f"  {ext}: {count} files")
    
    # 2. Image-level validation (if requested)
    if args.check_images and structure_analysis["image_files"] > 0:
        print("\n=== Image Validation ===")
        
        # Load images using appropriate loader
        try:
            if structure_analysis["structure_type"] == "classification_folders":
                dataset = DatasetLoader(str(data_path))
                image_paths = [sample[0] for sample in dataset.samples]
            else:
                loader = ImageDataLoader(str(data_path))
                image_paths = [str(path) for path in loader.image_paths]
            
            # Limit number of images to check
            if args.max_samples and len(image_paths) > args.max_samples:
                image_paths = image_paths[:args.max_samples]
                print(f"Checking {args.max_samples} out of {structure_analysis['image_files']} images")
            
            # Validate each image
            image_results = []
            for i, image_path in enumerate(image_paths):
                result = validate_image_file(Path(image_path))
                image_results.append(result)
                
                # Progress indicator
                if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
                    print(f"Validated {i + 1}/{len(image_paths)} images")
            
            # Count validation results
            valid_count = sum(1 for r in image_results if r["valid_image"])
            error_count = sum(1 for r in image_results if r["errors"])
            
            validation_report["image_validation"] = {
                "total_checked": len(image_results),
                "valid_images": valid_count,
                "invalid_images": len(image_results) - valid_count,
                "images_with_errors": error_count,
                "detailed_results": image_results
            }
            
            print(f"Valid Images: {valid_count}/{len(image_results)}")
            print(f"Images with Issues: {error_count}")
            
            # 3. Compute statistics
            if valid_count > 0:
                print("\n=== Image Statistics ===")
                stats = compute_image_statistics(image_results)
                validation_report["statistics"] = stats
                
                if "dimensions" in stats:
                    dim_stats = stats["dimensions"]
                    print(f"Image Dimensions:")
                    print(f"  Width: {dim_stats['width']['min']}-{dim_stats['width']['max']} (avg: {dim_stats['width']['mean']:.1f})")
                    print(f"  Height: {dim_stats['height']['min']}-{dim_stats['height']['max']} (avg: {dim_stats['height']['mean']:.1f})")
                
                if "color_modes" in stats:
                    print(f"Color Modes: {dict(stats['color_modes'])}")
                
                if "file_formats" in stats:
                    print(f"File Formats: {dict(stats['file_formats'])}")
            
        except Exception as e:
            error_msg = f"Image validation failed: {str(e)}"
            print(f"Error: {error_msg}")
            validation_report["errors"].append(error_msg)
    
    # 4. Generate recommendations
    print("\n=== Recommendations ===")
    recommendations = []
    
    if structure_analysis["image_files"] == 0:
        recommendations.append("No image files found. Check if the path is correct.")
    elif structure_analysis["image_files"] < 10:
        recommendations.append("Very few images found. Consider adding more data for better model training.")
    
    if structure_analysis["structure_type"] == "classification_folders":
        # Check class balance
        class_counts = {name: info["file_count"] 
                       for name, info in structure_analysis["directory_structure"].items()}
        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            if max_count > min_count * 3:  # Imbalance threshold
                recommendations.append(f"Class imbalance detected. Consider balancing classes: {class_counts}")
    
    if args.check_images and "statistics" in validation_report:
        stats = validation_report["statistics"]
        if "dimensions" in stats:
            dim_stats = stats["dimensions"]
            if dim_stats["width"]["std"] > 100 or dim_stats["height"]["std"] > 100:
                recommendations.append("High variance in image dimensions. Consider resizing images.")
    
    if not recommendations:
        recommendations.append("Dataset looks good! No major issues detected.")
    
    validation_report["recommendations"] = recommendations
    
    for rec in recommendations:
        print(f"  - {rec}")
    
    # 5. Save validation report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print(f"\n=== Validation Complete ===")
    print(f"Report saved to: {output_path}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  - Structure: {structure_analysis['structure_type']}")
    print(f"  - Total Images: {structure_analysis['image_files']}")
    if args.check_images and "image_validation" in validation_report:
        iv = validation_report["image_validation"]
        print(f"  - Valid Images: {iv['valid_images']}/{iv['total_checked']} checked")
    print(f"  - Recommendations: {len(recommendations)}")


if __name__ == "__main__":
    main()