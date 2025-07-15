"""Interactive Web App for Viewing Nearest Neighbors - Mosaic Color Data

This script creates an interactive Altair visualization where you can hover over 
image patches to see the top5 nearest neighbor words for each visual token.
Specifically designed for color mosaic data format.

Usage:
    python interactive_nearest_neighbors_viewer_mosaic_data.py --results-file path/to/results.json
    python interactive_nearest_neighbors_viewer_mosaic_data.py --results-file path/to/results.json --image-idx 5
    python interactive_nearest_neighbors_viewer_mosaic_data.py --results-file path/to/results.json --split train --image-idx 5
"""

import json
import argparse
import math
import base64
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import io

log = logging.getLogger(__name__)

def escape_for_html_attribute(text: str) -> str:
    """Properly escape text for HTML attributes."""
    if not text:
        return ""
    # First escape HTML entities
    escaped = html.escape(text, quote=True)
    # Then escape for JavaScript (in case it's used in JS)
    escaped = escaped.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    return escaped

def patch_idx_to_row_col(patch_idx: int, patches_per_chunk: int) -> Tuple[int, int]:
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col

def image_to_base64(image_path: str, preprocessor=None) -> str:
    """Convert image file to base64 string for embedding in HTML, with optional preprocessing."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply the same preprocessing as the model if preprocessor is provided
            if preprocessor is not None:
                try:
                    # Convert PIL image to numpy array for preprocessing
                    from olmo.data.model_preprocessor import load_image
                    image_array = load_image(img)
                    
                    # Apply the same preprocessing as the model
                    processed_image, img_mask = preprocessor.mm_preprocessor.resize_image(
                        image_array, 
                        (512, 512), 
                        is_training=False,
                        rng=np.random
                    )
                    
                    # Convert back to PIL image (processed_image is in [0,1] range)
                    processed_image = (processed_image * 255).astype(np.uint8)
                    img = Image.fromarray(processed_image)
                    print(f"Applied preprocessing to image: {image_path}")
                except Exception as e:
                    print(f"Could not preprocess image {image_path}: {e}, using original")
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        log.warning(f"Could not convert image {image_path} to base64: {e}")
        return ""

def create_interactive_visualization(
    image_data: Dict, 
    image_path: Optional[str] = None,
    split_name: str = "unknown",
    output_path: Optional[str] = None
) -> alt.Chart:
    """Create an interactive Altair visualization for a single image."""
    
    # Extract basic info
    image_idx = image_data.get("image_idx", 0)
    
    # Handle color grid format which uses true_color_sequence instead of ground_truth_caption
    color_sequence = image_data.get("true_color_sequence", [])
    caption = f"Color sequence: {' '.join(color_sequence)}" if color_sequence else "No color sequence available"
    
    # Get chunks and patches
    chunks = image_data.get("chunks", [])
    if not chunks:
        log.warning("No chunks found in image data")
        return None
    
    # Determine patches_per_chunk from the data
    patches_per_chunk = len(chunks[0].get("patches", []))
    grid_size = int(math.sqrt(patches_per_chunk))
    
    print(f"Processing image {image_idx} with {patches_per_chunk} patches ({grid_size}x{grid_size} grid)")
    
    # Prepare data for Altair
    chart_data = []
    
    for chunk_idx, chunk in enumerate(chunks):
        for patch in chunk.get("patches", []):
            patch_idx = patch.get("patch_idx", -1)
            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
            
            # Check interpretability using ground_truth_match (for color matching)
            is_interpretable = patch.get("ground_truth_match", False)
            corresponding_color = patch.get("corresponding_color", "")
            
            # Get nearest neighbors
            nearest_neighbors = patch.get("nearest_neighbors", [])
            nn_text = ""
            if nearest_neighbors:
                nn_list = []
                for i, nn in enumerate(nearest_neighbors[:5]):  # Top 5
                    token = nn.get("token", "")
                    similarity = nn.get("similarity", 0.0)
                    nn_list.append(f"{i+1}. '{token}' ({similarity:.3f})")
                nn_text = "\n".join(nn_list)
            
            # Get matches if available
            matches = patch.get("matches", [])
            matches_text = ""
            if matches:
                match_list = []
                for match in matches:
                    token = match.get("token", "")
                    ground_truth_color = match.get("ground_truth_color", "")
                    match_type = match.get("match_type", "")
                    match_list.append(f"'{token}' → '{ground_truth_color}' ({match_type})")
                matches_text = "; ".join(match_list)
            
            # Determine color based on interpretability (simpler for color grids)
            if is_interpretable:
                color = "#00AA00"  # Green
                opacity = 0.6
                category = "Color Match"
            else:
                color = "#AA0000"  # Red
                opacity = 0.3
                category = "No Match"
            
            # Create tooltip text with color-specific information
            tooltip_parts = [
                f"Patch ({row},{col}) - Index {patch_idx}",
                f"Category: {category}"
            ]
            
            if corresponding_color:
                tooltip_parts.append(f"Expected Color: {corresponding_color}")
            
            tooltip_parts.append(f"\nTop 5 Nearest Neighbors:\n{nn_text}")
            
            if matches_text:
                tooltip_parts.append(f"\nColor Matches:\n{matches_text}")
            
            tooltip_text = "\n".join(tooltip_parts)
            
            chart_data.append({
                "patch_idx": patch_idx,
                "row": row,
                "col": col,
                "x": col,
                "y": grid_size - row - 1,  # Flip Y to match image coordinates
                "category": category,
                "interpretable": is_interpretable,
                "color": color,
                "opacity": opacity,
                "nearest_neighbors": nn_text,
                "matches": matches_text,
                "corresponding_color": corresponding_color,
                "tooltip_text": tooltip_text
            })
    
    # Create DataFrame
    df = pd.DataFrame(chart_data)
    
    # Configure Altair
    alt.data_transformers.disable_max_rows()
    
    # Create the base chart
    base_chart = alt.Chart(df).add_selection(
        alt.selection_single(on='mouseover', empty='all')
    )
    
    # Create the grid rectangles
    grid_chart = base_chart.mark_rect(
        stroke='black',
        strokeWidth=1,
        cornerRadius=2
    ).encode(
        x=alt.X('x:O', 
                title='Column',
                axis=alt.Axis(orient='top')),
        y=alt.Y('y:O', 
                title='Row',
                axis=alt.Axis(orient='left')),
        color=alt.Color('color:N', 
                       scale=None,
                       legend=alt.Legend(title="Category",
                                       values=['#00AA00', '#AA0000'],
                                       labelExpr="datum.value == '#00AA00' ? 'Color Match' : 'No Match'")),
        opacity=alt.Opacity('opacity:Q', scale=None, legend=None),
        tooltip=[
            alt.Tooltip('patch_idx:O', title='Patch Index'),
            alt.Tooltip('row:O', title='Row'),
            alt.Tooltip('col:O', title='Column'),
            alt.Tooltip('category:O', title='Category'),
            alt.Tooltip('corresponding_color:N', title='Expected Color'),
            alt.Tooltip('nearest_neighbors:N', title='Top 5 Nearest Neighbors'),
            alt.Tooltip('matches:N', title='Color Matches')
        ],
        size=alt.value(400)  # Make rectangles larger
    ).properties(
        width=grid_size * 40,  # Adjust size based on grid
        height=grid_size * 40,
        title=f"Interactive Nearest Neighbors Viewer - {split_name.title()} Image {image_idx}"
    )
    
    # Add text labels for patch indices (optional, can be toggled)
    text_chart = base_chart.mark_text(
        fontSize=8,
        color='white',
        fontWeight='bold'
    ).encode(
        x=alt.X('x:O'),
        y=alt.Y('y:O'),
        text=alt.Text('patch_idx:O'),
        opacity=alt.value(0.7)
    )
    
    # Combine charts
    final_chart = (grid_chart + text_chart).resolve_scale(
        color='independent'
    ).properties(
        title=alt.TitleParams(
            text=[
                f"Interactive Nearest Neighbors Viewer - {split_name.title()} Image {image_idx}",
                f"Color Sequence: {caption}",
                "Hover over cells to see top 5 nearest neighbor words | Green = Color Match, Red = No Match"
            ],
            fontSize=14,
            anchor='start'
        )
    )
    
    # Save if output path provided
    if output_path:
        final_chart.save(output_path)
        print(f"Interactive visualization saved to {output_path}")
    
    return final_chart

def create_html_with_image_overlay(
    image_data: Dict,
    images_dir: Optional[Path] = None,
    split_name: str = "unknown",
    output_path: Optional[str] = None,
    preprocessor=None
) -> bool:
    """Create an HTML page with actual image background and interactive grid overlay."""
    
    # Extract basic info
    image_idx = image_data.get("image_idx", 0)
    
    # Handle color grid format
    color_sequence = image_data.get("true_color_sequence", [])
    caption = f"Color sequence: {' '.join(color_sequence)}" if color_sequence else "No color sequence available"
    
    # Get chunks and patches
    chunks = image_data.get("chunks", [])
    if not chunks:
        log.warning("No chunks found in image data")
        return False
    
    # Determine patches_per_chunk from the data
    patches_per_chunk = len(chunks[0].get("patches", []))
    grid_size = int(math.sqrt(patches_per_chunk))
    
    print(f"Processing image {image_idx} with {patches_per_chunk} patches ({grid_size}x{grid_size} grid)")
    
    # Try to find the image file
    image_base64 = None
    if images_dir:
        # Look for image files
        image_filename = image_data.get("image_filename")
        if image_filename:
            image_path = images_dir / image_filename
            if image_path.exists():
                image_base64 = image_to_base64(str(image_path), preprocessor)
                print(f"Found image: {image_path}")
            else:
                print(f"Image file not found: {image_path}")
    
    if not image_base64:
        print("No image found, falling back to grid-only visualization")
        return False
    
    # Calculate grid dimensions
    display_size = 600  # Size for display
    cell_size = display_size / grid_size
    
    # Prepare patch data
    patches_data = []
    total_patches_processed = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_patches = chunk.get("patches", [])
        print(f"  Chunk {chunk_idx}: {len(chunk_patches)} patches")
        
        for patch in chunk_patches:
            patch_idx = patch.get("patch_idx", -1)
            
            # Use patch_row and patch_col directly from data if available
            if "patch_row" in patch and "patch_col" in patch:
                row = patch["patch_row"]
                col = patch["patch_col"]
            else:
                # Fall back to calculation
                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
            
            # Calculate pixel coordinates for this patch
            x = col * cell_size
            y = row * cell_size
            
            # Check interpretability using ground_truth_match
            is_interpretable = patch.get("ground_truth_match", False)
            corresponding_color = patch.get("corresponding_color", "")
            
            # Get nearest neighbors
            nearest_neighbors = patch.get("nearest_neighbors", [])
            nn_text = ""
            if nearest_neighbors:
                nn_list = []
                for i, nn in enumerate(nearest_neighbors[:5]):  # Top 5
                    token = nn.get("token", "")
                    similarity = nn.get("similarity", 0.0)
                    nn_list.append(f"{i+1}. '{token}' ({similarity:.3f})")
                nn_text = "\n".join(nn_list)
            
            # Get matches if available
            matches = patch.get("matches", [])
            matches_list = []
            if matches:
                for match in matches:
                    token = match.get("token", "")
                    ground_truth_color = match.get("ground_truth_color", "")
                    match_type = match.get("match_type", "")
                    
                    # Escape properly
                    safe_token = escape_for_html_attribute(token)
                    safe_color = escape_for_html_attribute(ground_truth_color)
                    safe_match_type = escape_for_html_attribute(match_type)
                    matches_list.append(f"'{safe_token}' → '{safe_color}' ({safe_match_type})")
            
            # Create safe strings for HTML attributes
            nn_attr = escape_for_html_attribute(" | ".join([f"{i+1}. '{nn.get('token', '')}' ({nn.get('similarity', 0.0):.3f})" for i, nn in enumerate(nearest_neighbors[:5])]))
            matches_attr = escape_for_html_attribute(" | ".join(matches_list))
            corresponding_color_attr = escape_for_html_attribute(corresponding_color)
            
            # Determine color based on interpretability
            if is_interpretable:
                color = "#00AA00"  # Green
                opacity = 0.6
                category = "Color Match"
            else:
                color = "#AA0000"  # Red
                opacity = 0.3
                category = "No Match"
            
            patches_data.append({
                "patch_idx": patch_idx,
                "row": row,
                "col": col,
                "x": x,
                "y": y,
                "width": cell_size,
                "height": cell_size,
                "category": category,
                "interpretable": is_interpretable,
                "color": color,
                "opacity": opacity,
                "nearest_neighbors": nn_attr,
                "matches": matches_attr,
                "corresponding_color": corresponding_color_attr
            })
            total_patches_processed += 1
    
    print(f"  Total patches processed: {total_patches_processed}")
    print(f"  Expected patches: {patches_per_chunk} ({grid_size}x{grid_size})")
    
    if total_patches_processed != patches_per_chunk:
        print(f"  WARNING: Processed {total_patches_processed} patches but expected {patches_per_chunk}")
    
    # Debug: Show range of coordinates
    if patches_data:
        rows = [p["row"] for p in patches_data]
        cols = [p["col"] for p in patches_data]
        print(f"  Row range: {min(rows)} to {max(rows)}")
        print(f"  Col range: {min(cols)} to {max(cols)}")
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Nearest Neighbors - {split_name.title()} Image {image_idx}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .title {{
            color: #333;
            margin-bottom: 10px;
        }}
        .caption {{
            color: #666;
            margin-bottom: 20px;
            font-style: italic;
        }}
        .image-container {{
            position: relative;
            display: inline-block;
        }}
        .background-image {{
            width: {display_size}px;
            height: {display_size}px;
            object-fit: cover;
        }}
        .overlay-svg {{
            position: absolute;
            top: 0;
            left: 0;
            width: {display_size}px;
            height: {display_size}px;
            pointer-events: none;
        }}
        .grid-cell {{
            pointer-events: all;
            cursor: pointer;
        }}
        .tooltip {{
            position: absolute;
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px;
            border-radius: 5px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            font-size: 12px;
            line-height: 1.4;
            display: none;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .toggle-button {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 0;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .toggle-button:hover {{
            background-color: #45a049;
        }}
        .toggle-button.off {{
            background-color: #f44336;
        }}
        .toggle-button.off:hover {{
            background-color: #da190b;
        }}
        .controls {{
            margin-bottom: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">Interactive Nearest Neighbors Viewer - {split_name.title()} Image {image_idx}</h1>
        <p class="caption">{caption}</p>
        
        <div class="controls">
            <button id="toggleGrid" class="toggle-button">Hide Grid</button>
        </div>
        
        <div class="image-container">
            <img src="{image_base64}" alt="Image {image_idx}" class="background-image">
            <svg class="overlay-svg" id="gridOverlay">
"""
    
    # Add grid cells to SVG
    for patch in patches_data:
        html_content += f"""
                <rect class="grid-cell" 
                      x="{patch['x']}" y="{patch['y']}" 
                      width="{patch['width']}" height="{patch['height']}"
                      fill="{patch['color']}" 
                      opacity="{patch['opacity']}"
                      stroke="black" 
                      stroke-width="1"
                      data-patch-idx="{patch['patch_idx']}"
                      data-row="{patch['row']}"
                      data-col="{patch['col']}"
                      data-category="{patch['category']}"
                      data-interpretable="{patch['interpretable']}"
                      data-corresponding-color="{patch['corresponding_color']}"
                      data-nn="{patch['nearest_neighbors']}"
                      data-matches="{patch['matches']}"/>
"""
    
    html_content += f"""
            </svg>
        </div>
        
        <div class="tooltip" id="tooltip"></div>
        
        <div class="legend">
            <h3>Legend:</h3>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #00AA00;"></span>
                <span>Color Match (nearest neighbor matches expected color)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #AA0000;"></span>
                <span>No Match</span>
            </div>
        </div>
        
        <p><strong>How to use:</strong> Hover over the colored grid cells to see the top 5 nearest neighbor words for each image patch. Green cells have nearest neighbors that match the expected color for that position, red cells do not.</p>
    </div>

    <script>
        const tooltip = document.getElementById('tooltip');
        const gridCells = document.querySelectorAll('.grid-cell');
        const toggleButton = document.getElementById('toggleGrid');
        const gridOverlay = document.getElementById('gridOverlay');
        
        let gridVisible = true;
        
        // Toggle grid visibility
        toggleButton.addEventListener('click', () => {{
            gridVisible = !gridVisible;
            if (gridVisible) {{
                gridOverlay.style.display = 'block';
                toggleButton.textContent = 'Hide Grid';
                toggleButton.classList.remove('off');
            }} else {{
                gridOverlay.style.display = 'none';
                toggleButton.textContent = 'Show Grid';
                toggleButton.classList.add('off');
            }}
        }});
        
        gridCells.forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const patchIdx = e.target.getAttribute('data-patch-idx');
                const row = e.target.getAttribute('data-row');
                const col = e.target.getAttribute('data-col');
                const category = e.target.getAttribute('data-category');
                const correspondingColor = e.target.getAttribute('data-corresponding-color');
                const nearestNeighbors = e.target.getAttribute('data-nn');
                const matches = e.target.getAttribute('data-matches');
                
                let tooltipContent = `
                    <strong>Patch (${{row}}, ${{col}}) - Index ${{patchIdx}}</strong><br>
                    <strong>Category:</strong> ${{category}}<br>
                `;
                
                if (correspondingColor && correspondingColor.trim()) {{
                    tooltipContent += `<strong>Expected Color:</strong> ${{correspondingColor}}<br>`;
                }}
                
                tooltipContent += `<br><strong>Top 5 Nearest Neighbors:</strong><br>
                    ${{nearestNeighbors.split(' | ').join('<br>')}}
                `;
                
                if (matches && matches.trim()) {{
                    tooltipContent += `<br><br><strong>Color Matches:</strong><br>${{matches.split(' | ').join('<br>')}}`;
                }}
                
                tooltip.innerHTML = tooltipContent;
                tooltip.style.display = 'block';
            }});
            
            cell.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.pageX + 10) + 'px';
                tooltip.style.top = (e.pageY + 10) + 'px';
            }});
            
            cell.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
        }});
    </script>
</body>
</html>
"""
    
    # Save HTML file
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Interactive HTML visualization with image background saved to {output_path}")
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Create interactive visualization of nearest neighbors for color mosaic data")
    
    # Default parameters matching the analysis script
    checkpoint_path = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2/step12000-unsharded"
    ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
    default_results_file = f"analysis_results/nearest_neighbors/{ckpt_name}/nearest_neighbors_analysis_color_names_mosaic_24x24.json"
    
    parser.add_argument("--results-file", type=str, default=default_results_file,
                       help=f"Path to the JSON results file from general_and_nearest_neighbors_mosaic-data.py (default: {default_results_file})")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                       help="Which split to visualize (default: train)")
    parser.add_argument("--image-idx", type=int, default=None,
                       help="Specific image index to visualize (if not provided, creates visualizations for all images)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for HTML files (defaults to same directory as results file)")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process (default: 10)")
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file {results_file} does not exist")
        print(f"Expected path: {results_file}")
        print("\nMake sure you've run the analysis first:")
        print("python general_and_nearest_neighbors_mosaic-data.py")
        return
    
    print(f"Loading results from {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create preprocessor for image processing (same as original script)
    preprocessor = None
    try:
        checkpoint_path = results.get("checkpoint", "")
        if checkpoint_path:
            print(f"Creating preprocessor from checkpoint: {checkpoint_path}")
            from olmo.config import ModelConfig
            from olmo.data import build_mm_preprocessor
            from olmo.model import Molmo
            from olmo.util import resource_path
            
            if "hf:" in checkpoint_path:
                model = Molmo.from_checkpoint(checkpoint_path, device="cpu")
                model_config = model.config
            else:
                model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
            model_config.system_prompt_kind = "style"
            preprocessor = build_mm_preprocessor(
                model_config,
                for_inference=True,
                shuffle_messages=False,
                is_training=False,
                require_image_features=True
            )
            print("Successfully created preprocessor")
        else:
            print("No checkpoint path found in results, will use original images")
    except Exception as e:
        print(f"Could not create preprocessor: {e}, will use original images")
        preprocessor = None
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_file.parent / "interactive_visualizations_mosaic"
    output_dir.mkdir(exist_ok=True)
    
    # Try to find images directory
    images_dir = results_file.parent / "images"
    if not images_dir.exists():
        print(f"Warning: Images directory not found at {images_dir}")
        print("Will create visualizations without image background")
        images_dir = None
    else:
        print(f"Found images directory: {images_dir}")
    
    # Get split data
    split_data = results.get("splits", {}).get(args.split, {})
    if not split_data:
        print(f"Error: No data found for split '{args.split}'")
        return
    
    images = split_data.get("images", [])
    if not images:
        print(f"Error: No images found in split '{args.split}'")
        return
    
    print(f"Found {len(images)} images in {args.split} split")
    
    # Determine which images to process
    if args.image_idx is not None:
        if args.image_idx >= len(images):
            print(f"Error: Image index {args.image_idx} not found (max: {len(images) - 1})")
            return
        images_to_process = [images[args.image_idx]]
        indices_to_process = [args.image_idx]
    else:
        images_to_process = images[:args.max_images]
        indices_to_process = list(range(min(args.max_images, len(images))))
    
    print(f"Processing {len(images_to_process)} images...")
    
    # Create visualizations
    for idx, image_data in zip(indices_to_process, images_to_process):
        try:
            print(f"Creating visualization for image {idx}...")
            
            # Determine output filename
            output_filename = f"interactive_nn_viewer_mosaic_{args.split}_image_{idx:04d}.html"
            output_path = output_dir / output_filename
            
            # Try to create HTML with image background first
            success = False
            if images_dir:
                success = create_html_with_image_overlay(
                    image_data,
                    images_dir=images_dir,
                    split_name=args.split,
                    output_path=str(output_path),
                    preprocessor=preprocessor
                )
            
            # Fall back to Altair grid if HTML creation failed
            if not success:
                print(f"Falling back to simple grid visualization for image {idx}")
                chart = create_interactive_visualization(
                    image_data,
                    split_name=args.split,
                    output_path=str(output_path)
                )
                if chart is not None:
                    print(f"✓ Saved: {output_path}")
                else:
                    print(f"✗ Failed to create visualization for image {idx}")
            else:
                print(f"✓ Saved: {output_path}")
                
        except Exception as e:
            print(f"✗ Error processing image {idx}: {e}")
            continue
    
    print(f"\nDone! Interactive visualizations saved to {output_dir}")
    print(f"Open the HTML files in a web browser to explore the nearest neighbors interactively.")
    
    # Create an index file listing all visualizations
    index_file = output_dir / f"index_mosaic_{args.split}.html"
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Nearest Neighbors Viewer - Mosaic {args.split.title()} Split</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin: 10px 0; }}
        a {{ text-decoration: none; color: #0066cc; padding: 10px; border: 1px solid #ddd; display: block; border-radius: 5px; }}
        a:hover {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Interactive Nearest Neighbors Viewer - Mosaic Color Data {args.split.title()} Split</h1>
    <p>Click on any link below to open the interactive visualization for that color grid image.</p>
    <ul>
"""
    
    for idx in indices_to_process:
        filename = f"interactive_nn_viewer_mosaic_{args.split}_image_{idx:04d}.html"
        html_content += f'        <li><a href="{filename}">Image {idx:04d}</a></li>\n'
    
    html_content += """
    </ul>
    <p><strong>How to use:</strong> Hover over the colored grid cells to see the top 5 nearest neighbor words for each image patch. Green cells have nearest neighbors that match the expected color for that position, red cells do not.</p>
</body>
</html>
"""
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Index file created: {index_file}")

if __name__ == "__main__":
    main() 