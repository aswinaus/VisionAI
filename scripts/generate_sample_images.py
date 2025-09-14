"""
Generate sample images for testing the Vision AI Showcase.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_sample_images(output_dir: str = "data/samples"):
    """
    Create sample images for testing vision AI models.
    
    Args:
        output_dir: Directory to save sample images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Image size
    width, height = 224, 224
    
    # Sample 1: Solid color images
    colors = [
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)), 
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
        ("purple", (255, 0, 255)),
        ("cyan", (0, 255, 255)),
    ]
    
    for color_name, color_rgb in colors:
        img = Image.new('RGB', (width, height), color_rgb)
        
        # Add text label
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a default font
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), color_name.upper(), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Add black text with white outline
        outline_range = 2
        for dx in range(-outline_range, outline_range + 1):
            for dy in range(-outline_range, outline_range + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), color_name.upper(), font=font, fill="white")
        draw.text((x, y), color_name.upper(), font=font, fill="black")
        
        img.save(output_path / f"sample_{color_name}.jpg", quality=95)
        print(f"Created: sample_{color_name}.jpg")
    
    # Sample 2: Geometric patterns
    patterns = [
        ("circles", create_circle_pattern),
        ("squares", create_square_pattern),
        ("stripes", create_stripe_pattern),
        ("gradient", create_gradient_pattern),
    ]
    
    for pattern_name, pattern_func in patterns:
        img = pattern_func(width, height)
        img.save(output_path / f"pattern_{pattern_name}.png")
        print(f"Created: pattern_{pattern_name}.png")
    
    # Sample 3: Noise images
    noise_types = ["random", "gaussian"]
    for noise_type in noise_types:
        if noise_type == "random":
            # Random noise
            noise_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        else:
            # Gaussian noise
            noise_data = np.random.normal(128, 50, (height, width, 3))
            noise_data = np.clip(noise_data, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(noise_data, 'RGB')
        img.save(output_path / f"noise_{noise_type}.png")
        print(f"Created: noise_{noise_type}.png")
    
    # Sample 4: Test image with mixed content
    img = create_test_composite(width, height)
    img.save(output_path / "test_composite.jpg", quality=95)
    print("Created: test_composite.jpg")
    
    print(f"\nGenerated {len(colors) + len(patterns) + len(noise_types) + 1} sample images in {output_path}")
    

def create_circle_pattern(width: int, height: int) -> Image.Image:
    """Create an image with circular patterns."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw concentric circles
    center_x, center_y = width // 2, height // 2
    for i in range(1, 6):
        radius = i * 20
        color = tuple(np.random.randint(0, 256, 3))
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                     outline=color, width=3)
    
    return img


def create_square_pattern(width: int, height: int) -> Image.Image:
    """Create an image with square patterns."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw grid of squares
    square_size = 40
    for x in range(0, width, square_size):
        for y in range(0, height, square_size):
            color = tuple(np.random.randint(0, 256, 3))
            draw.rectangle([x, y, x + square_size - 5, y + square_size - 5], 
                          fill=color)
    
    return img


def create_stripe_pattern(width: int, height: int) -> Image.Image:
    """Create an image with stripe patterns."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw horizontal stripes
    stripe_height = 20
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, y in enumerate(range(0, height, stripe_height)):
        color = colors[i % len(colors)]
        draw.rectangle([0, y, width, y + stripe_height], fill=color)
    
    return img


def create_gradient_pattern(width: int, height: int) -> Image.Image:
    """Create an image with gradient pattern."""
    # Create horizontal gradient
    data = np.zeros((height, width, 3), dtype=np.uint8)
    
    for x in range(width):
        intensity = int(255 * x / width)
        data[:, x, 0] = intensity  # Red channel
        data[:, x, 1] = 255 - intensity  # Green channel  
        data[:, x, 2] = intensity // 2  # Blue channel
    
    return Image.fromarray(data, 'RGB')


def create_test_composite(width: int, height: int) -> Image.Image:
    """Create a composite test image with various elements."""
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Quarter 1: Red circle
    draw.ellipse([10, 10, width//2 - 10, height//2 - 10], fill=(255, 100, 100))
    
    # Quarter 2: Blue rectangle
    draw.rectangle([width//2 + 10, 10, width - 10, height//2 - 10], fill=(100, 100, 255))
    
    # Quarter 3: Green triangle (using polygon)
    points = [
        (10, height//2 + 10),
        (width//2 - 10, height//2 + 10),
        (width//4, height - 10)
    ]
    draw.polygon(points, fill=(100, 255, 100))
    
    # Quarter 4: Pattern
    for i in range(0, width//2, 10):
        for j in range(height//2, height, 10):
            if (i + j) % 20 == 0:
                draw.rectangle([width//2 + i, j, width//2 + i + 8, j + 8], 
                              fill=(255, 255, 100))
    
    return img


if __name__ == "__main__":
    create_sample_images()