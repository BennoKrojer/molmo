import logging
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors
import random
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_mosaic_image(colors, grid_size=28, cell_size=224):
    """Create a mosaic image with a grid of colors.
    
    Args:
        colors: List of RGB tuples
        grid_size: Size of the grid (e.g., 12 for 12x12)
        cell_size: Size of each cell in pixels
    """
    # Create a new image with white background
    total_size = grid_size * cell_size
    img = Image.new('RGB', (total_size, total_size), 'white')
    
    # Fill in each cell with its corresponding color
    for i in range(grid_size):
        for j in range(grid_size):
            color_idx = i * grid_size + j
            color = colors[color_idx]
            
            # Calculate cell boundaries
            left = j * cell_size
            upper = i * cell_size
            right = left + cell_size
            lower = upper + cell_size
            
            # Create a cell image and paste it
            cell = Image.new('RGB', (cell_size, cell_size), color)
            img.paste(cell, (left, upper))
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Generate mosaic images with a 12x12 grid of colors.")
    parser.add_argument("--output-dir", type=str, default="molmo_data/color_mosaic_images_gridsize-24",
                        help="Directory to save color mosaic images (default: molmo_data/color_mosaic_images)")
    parser.add_argument("--num-images", type=int, default=10000,
                        help="Number of mosaic images to generate (default: 10000)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Replace this list with your chosen 10 XKCD color names
    selected_colors = [
        "xkcd:red",
        "xkcd:green",
        "xkcd:blue",
        "xkcd:yellow",
        "xkcd:purple",
        "xkcd:orange",
        "xkcd:pink",
        "xkcd:brown"
    ]

    # Convert XKCD color names to RGB
    rgb_colors = [hex_to_rgb(mcolors.XKCD_COLORS[color]) for color in selected_colors]
    
    # Dictionary to store color sequences for each image
    color_sequences = {}

    log.info(f"Generating {args.num_images} mosaic images...")
    
    for i in range(args.num_images):
        # Generate random sequence of colors (144 colors for 12x12 grid)
        random_colors = random.choices(rgb_colors, k=24*24)
        
        # Create the mosaic image
        mosaic = create_mosaic_image(random_colors, grid_size=24)
        
        # Save the mosaic image
        filename = f"{i:03d}.png"
        mosaic.save(output_dir / filename)
        
        # Store the color sequence (using original XKCD names)
        color_names = [selected_colors[rgb_colors.index(color)].split(':')[1] for color in random_colors]
        color_sequences[filename] = color_names
        
        if (i + 1) % 10 == 0:
            log.info(f"Generated {i + 1} images")

    # Save the color sequences to a JSON file
    json_path = output_dir / "color_sequences.json"
    with open(json_path, 'w') as f:
        json.dump(color_sequences, f, indent=2)
    
    log.info(f"✅ Generated {args.num_images} mosaic images")
    log.info(f"✅ Saved color sequences to {json_path}")

if __name__ == "__main__":
    main()
