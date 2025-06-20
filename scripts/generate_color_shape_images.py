import logging
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.colors as mcolors
import math
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_circle(draw, center, radius, color):
    """Draw a filled circle."""
    draw.ellipse([center[0] - radius, center[1] - radius,
                  center[0] + radius, center[1] + radius],
                 fill=color)

def draw_square(draw, center, size, color):
    """Draw a filled square."""
    half_size = size // 2
    draw.rectangle([center[0] - half_size, center[1] - half_size,
                   center[0] + half_size, center[1] + half_size],
                  fill=color)

def draw_triangle(draw, center, size, color):
    """Draw a filled equilateral triangle."""
    height = size * math.sqrt(3) / 2
    points = [
        (center[0], center[1] - height/2),  # top
        (center[0] - size/2, center[1] + height/2),  # bottom left
        (center[0] + size/2, center[1] + height/2),  # bottom right
    ]
    draw.polygon(points, fill=color)

def draw_star(draw, center, size, color):
    """Draw a filled 5-pointed star."""
    outer_radius = size / 2
    inner_radius = outer_radius * 0.4
    points = []
    for i in range(10):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = math.pi / 2 + i * math.pi / 5
        x = center[0] + radius * math.cos(angle)
        y = center[1] - radius * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=color)

def create_shape_image(color, shape_name, image_size=224, shape_size=100):
    """Create an image with a shape of the specified color."""
    # Create white background
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Calculate center and size
    center = (image_size // 2, image_size // 2)
    
    # Draw the shape
    if shape_name == 'circle':
        draw_circle(draw, center, shape_size // 2, color)
    elif shape_name == 'square':
        draw_square(draw, center, shape_size, color)
    elif shape_name == 'triangle':
        draw_triangle(draw, center, shape_size, color)
    elif shape_name == 'star':
        draw_star(draw, center, shape_size, color)
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Generate shape images with colors from the original dataset.")
    parser.add_argument("--output-dir", type=str, default="molmo_data/shape_images",
                        help="Directory to save shape images (default: molmo_data/shape_images)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all XKCD colors
    xkcd_colors = mcolors.XKCD_COLORS  # dict: {"xkcd:color name": "#hex"}
    
    # List of shapes to generate
    shapes = ['circle', 'square', 'triangle', 'star']
    
    # Dictionary to store color sequences
    color_sequences = {}
    
    log.info("Generating shape images...")
    image_count = 0
    
    for full_name, hex_value in xkcd_colors.items():
        color_name = full_name.replace("xkcd:", "")
        rgb = hex_to_rgb(hex_value)
        
        for shape in shapes:
            # Create the shape image
            img = create_shape_image(rgb, shape)
            
            # Save the image
            filename = f"{image_count:03d}.png"
            img.save(output_dir / filename)
            
            # Store the color and shape information
            color_sequences[filename] = {
                "color": color_name,
                "shape": shape
            }
            
            image_count += 1
            if image_count % 100 == 0:
                log.info(f"Generated {image_count} images")

    # Save the color and shape information to a JSON file
    json_path = output_dir / "shape_info.json"
    with open(json_path, 'w') as f:
        json.dump(color_sequences, f, indent=2)
    
    log.info(f"✅ Generated {image_count} shape images")
    log.info(f"✅ Saved shape information to {json_path}")

if __name__ == "__main__":
    main() 