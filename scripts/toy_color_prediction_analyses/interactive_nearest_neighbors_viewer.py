"""Interactive Web App for Viewing Nearest Neighbors - Unified Version

This script creates an interactive Altair visualization where you can hover over 
image patches to see the top5 nearest neighbor words for each visual token.
Supports both PixMo-Cap and Mosaic datasets via the --dataset argument.

Usage:
    python interactive_nearest_neighbors_viewer.py --dataset pixmo_cap --results-file path/to/results.json
    python interactive_nearest_neighbors_viewer.py --dataset mosaic --results-file path/to/results.json --image-idx 5
    python interactive_nearest_neighbors_viewer.py --dataset pixmo_cap --results-file path/to/results.json --split train --image-idx 5
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
    dataset_type: str = "pixmo_cap",
    image_path: Optional[str] = None,
    split_name: str = "unknown",
    output_path: Optional[str] = None
) -> alt.Chart:
    """Create an interactive Altair visualization for a single image."""
    
    # Extract basic info
    image_idx = image_data.get("image_idx", 0)
    
    # Handle different dataset formats
    if dataset_type == "pixmo_cap":
        caption = image_data.get("ground_truth_caption", "No caption available")
    else:  # mosaic
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
            
            # Handle different dataset types
            if dataset_type == "pixmo_cap":
                # Check interpretability using three-category system
                is_interpretable = patch.get("caption_match", False)
                is_visual_task = patch.get("visual_task_match", False)
                
                if matches:
                    match_list = []
                    for match in matches:
                        token = match.get("token", "")
                        matched_word = match.get("matched_word", "")
                        match_list.append(f"'{token}' → '{matched_word}'")
                    matches_text = "; ".join(match_list)
                
                # Determine color based on interpretability (three-category system)
                if is_interpretable:
                    color = "#00AA00"  # Green
                    opacity = 0.5
                    category = "Interpretable"
                elif is_visual_task:
                    color = "#0066CC"  # Blue
                    opacity = 0.4
                    category = "Visual/Task"
                else:
                    color = "#AA0000"  # Red
                    opacity = 0.2
                    category = "Non-interpretable"
                
                # Create tooltip text
                tooltip_parts = [
                    f"Patch ({row},{col})",
                    f"Category: {category}",
                    f"\nTop 5 Nearest Neighbors:\n{nn_text}"
                ]
                if matches_text:
                    tooltip_parts.append(f"\nCaption Matches:\n{matches_text}")
                
            else:  # mosaic
                # Check interpretability using ground_truth_match (for color matching)
                is_interpretable = patch.get("ground_truth_match", False)
                corresponding_color = patch.get("corresponding_color", "")
                
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
    
    # Create legend values based on dataset type
    if dataset_type == "pixmo_cap":
        legend_values = ['#00AA00', '#0066CC', '#AA0000']
        legend_labels = ['Interpretable', 'Visual/Task', 'Non-interpretable']
    else:  # mosaic
        legend_values = ['#00AA00', '#AA0000']
        legend_labels = ['Color Match', 'No Match']
    
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
                                       values=legend_values)),
        opacity=alt.value(0.7),
        tooltip=alt.Tooltip('tooltip_text:N', title="Patch Info")
    ).properties(
        width=400,
        height=400,
        title=f"Image {image_idx}: Nearest Neighbors Visualization"
    )
    
    # Create text labels for patch indices (optional, can be toggled)
    text_chart = base_chart.mark_text(
        align='center',
        baseline='middle',
        fontSize=8,
        color='white',
        fontWeight='bold'
    ).encode(
        x=alt.X('x:O'),
        y=alt.Y('y:O'),
        text=alt.Text('patch_idx:O'),
        opacity=alt.value(0.8)
    )
    
    # Combine charts
    final_chart = (grid_chart + text_chart).resolve_scale(
        color='independent'
    )
    
    # Save if output path provided
    if output_path:
        try:
            final_chart.save(output_path)
            print(f"Saved visualization to {output_path}")
        except Exception as e:
            print(f"Could not save chart to {output_path}: {e}")
    
    return final_chart

def create_html_with_image_overlay(
    image_data: Dict,
    dataset_type: str = "pixmo_cap",
    images_dir: Optional[Path] = None,
    split_name: str = "unknown",
    output_path: Optional[str] = None,
    preprocessor=None
) -> bool:
    """Create an HTML file with image background and interactive overlay."""
    
    try:
        # Extract basic info
        image_idx = image_data.get("image_idx", 0)
        
        # Handle different dataset formats for ground truth
        if dataset_type == "pixmo_cap":
            ground_truth = image_data.get("ground_truth_caption", "No caption available")
            ground_truth_label = "Caption"
        else:  # mosaic
            color_sequence = image_data.get("true_color_sequence", [])
            ground_truth = f"{' '.join(color_sequence)}" if color_sequence else "No color sequence available"
            ground_truth_label = "Color Sequence"
        
        # Get chunks and patches
        chunks = image_data.get("chunks", [])
        if not chunks:
            log.warning("No chunks found in image data")
            return False
        
        # Determine patches_per_chunk and grid_size
        patches_per_chunk = len(chunks[0].get("patches", []))
        grid_size = int(math.sqrt(patches_per_chunk))
        
        # Try to find the image file
        image_base64 = ""
        image_filename = image_data.get("image_filename")
        if images_dir and image_filename:
            image_path = images_dir / image_filename
            if image_path.exists():
                image_base64 = image_to_base64(str(image_path), preprocessor)
            else:
                print(f"Image file not found: {image_path}")
        
        # Prepare patch data
        patch_data = []
        for chunk_idx, chunk in enumerate(chunks):
            for patch in chunk.get("patches", []):
                patch_idx = patch.get("patch_idx", -1)
                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                
                # Get nearest neighbors
                nearest_neighbors = patch.get("nearest_neighbors", [])
                nn_list = []
                for i, nn in enumerate(nearest_neighbors[:5]):
                    token = nn.get("token", "")
                    similarity = nn.get("similarity", 0.0)
                    nn_list.append(f"{i+1}. '{escape_for_html_attribute(token)}' ({similarity:.3f})")
                
                # Get matches
                matches = patch.get("matches", [])
                match_list = []
                if matches:
                    if dataset_type == "pixmo_cap":
                        for match in matches:
                            token = match.get("token", "")
                            matched_word = match.get("matched_word", "")
                            match_list.append(f"'{escape_for_html_attribute(token)}' → '{escape_for_html_attribute(matched_word)}'")
                    else:  # mosaic
                        for match in matches:
                            token = match.get("token", "")
                            ground_truth_color = match.get("ground_truth_color", "")
                            match_type = match.get("match_type", "")
                            match_list.append(f"'{escape_for_html_attribute(token)}' → '{escape_for_html_attribute(ground_truth_color)}' ({escape_for_html_attribute(match_type)})")
                
                # Determine interpretability and color
                if dataset_type == "pixmo_cap":
                    is_interpretable = patch.get("caption_match", False)
                    is_visual_task = patch.get("visual_task_match", False)
                    
                    if is_interpretable:
                        color = "#00AA00"
                        category = "Interpretable"
                    elif is_visual_task:
                        color = "#0066CC"
                        category = "Visual/Task"
                    else:
                        color = "#AA0000"
                        category = "Non-interpretable"
                else:  # mosaic
                    is_interpretable = patch.get("ground_truth_match", False)
                    
                    if is_interpretable:
                        color = "#00AA00"
                        category = "Color Match"
                    else:
                        color = "#AA0000"
                        category = "No Match"
                
                patch_data.append({
                    "patch_idx": patch_idx,
                    "row": row,
                    "col": col,
                    "color": color,
                    "category": category,
                    "nearest_neighbors": nn_list,
                    "matches": match_list,
                    "corresponding_color": patch.get("corresponding_color", "") if dataset_type == "mosaic" else ""
                })
        
        # Create HTML content with embedded JavaScript
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Nearest Neighbors - {dataset_type.title()} Image {image_idx}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .image-container {{ position: relative; display: inline-block; margin: 20px auto; }}
        .base-image {{ max-width: 512px; max-height: 512px; border: 2px solid #333; }}
        .patch-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
        .patch {{ position: absolute; border: 1px solid rgba(0,0,0,0.3); cursor: pointer; pointer-events: all; transition: opacity 0.2s; }}
        .patch:hover {{ opacity: 0.8 !important; border: 2px solid #fff; }}
        .info-panel {{ margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }}
        .legend {{ display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border: 1px solid #000; }}
        .tooltip {{ position: absolute; background-color: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 5px; font-size: 12px; max-width: 300px; z-index: 1000; pointer-events: none; white-space: pre-line; display: none; }}
        .ground-truth {{ background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #17a2b8; }}
        .controls {{ margin-bottom: 20px; text-align: center; }}
        .toggle-button {{ background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 10px 0; cursor: pointer; border-radius: 5px; transition: background-color 0.3s; }}
        .toggle-button:hover {{ background-color: #45a049; }}
        .toggle-button.off {{ background-color: #f44336; }}
        .toggle-button.off:hover {{ background-color: #da190b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Interactive Nearest Neighbors - {dataset_type.title()} Dataset</h1>
            <h2>Image {image_idx} ({split_name} split)</h2>
        </div>
        <div class="ground-truth">
            <strong>{ground_truth_label}:</strong> {escape_for_html_attribute(ground_truth)}
        </div>
        <div class="legend">'''
        
        # Add legend items based on dataset type
        if dataset_type == "pixmo_cap":
            html_content += '''
            <div class="legend-item"><div class="legend-color" style="background-color: #00AA00;"></div><span>Interpretable (matches caption)</span></div>
            <div class="legend-item"><div class="legend-color" style="background-color: #0066CC;"></div><span>Visual/Task related</span></div>
            <div class="legend-item"><div class="legend-color" style="background-color: #AA0000;"></div><span>Non-interpretable</span></div>'''
        else:  # mosaic
            html_content += '''
            <div class="legend-item"><div class="legend-color" style="background-color: #00AA00;"></div><span>Color Match</span></div>
            <div class="legend-item"><div class="legend-color" style="background-color: #AA0000;"></div><span>No Match</span></div>'''
        
        html_content += f'''
        </div>
        <div class="controls">
            <button id="toggleGrid" class="toggle-button">Hide Grid</button>
        </div>
        <div class="image-container">'''
        
        if image_base64:
            html_content += f'<img src="{image_base64}" alt="Image {image_idx}" class="base-image" id="baseImage">'
        else:
            html_content += f'<div class="base-image" id="baseImage" style="width: 512px; height: 512px; background-color: #ddd; display: flex; align-items: center; justify-content: center;">Image not available</div>'
        
        html_content += f'''
            <div class="patch-overlay" id="patchOverlay"></div>
        </div>
        <div class="info-panel">
            <p><strong>Instructions:</strong> Hover over the colored patches to see the top 5 nearest neighbor words for each image region.</p>
            <p><strong>Grid:</strong> {grid_size}×{grid_size} patches ({patches_per_chunk} total)</p>
        </div>
        <div class="tooltip" id="tooltip"></div>
    </div>
    <script>
        const patchData = {json.dumps(patch_data, indent=2)};
        const gridSize = {grid_size};
        
        function createPatches() {{
            const baseImage = document.getElementById('baseImage');
            const overlay = document.getElementById('patchOverlay');
            const tooltip = document.getElementById('tooltip');
            
            const rect = baseImage.getBoundingClientRect();
            const imageWidth = rect.width;
            const imageHeight = rect.height;
            
            patchData.forEach(patch => {{
                const patchDiv = document.createElement('div');
                patchDiv.className = 'patch';
                
                const patchWidth = imageWidth / gridSize;
                const patchHeight = imageHeight / gridSize;
                const left = patch.col * patchWidth;
                const top = patch.row * patchHeight;
                
                patchDiv.style.left = left + 'px';
                patchDiv.style.top = top + 'px';
                patchDiv.style.width = patchWidth + 'px';
                patchDiv.style.height = patchHeight + 'px';
                patchDiv.style.backgroundColor = patch.color;
                patchDiv.style.opacity = '0.4';
                
                let tooltipContent = `Patch (${{patch.row}},${{patch.col}}) - Index ${{patch.patch_idx}}\\n`;
                tooltipContent += `Category: ${{patch.category}}\\n`;
                
                if (patch.corresponding_color) {{
                    tooltipContent += `Expected Color: ${{patch.corresponding_color}}\\n`;
                }}
                
                tooltipContent += '\\nTop 5 Nearest Neighbors:\\n';
                patch.nearest_neighbors.forEach(nn => {{ tooltipContent += nn + '\\n'; }});
                
                if (patch.matches.length > 0) {{
                    tooltipContent += '\\nMatches:\\n';
                    patch.matches.forEach(match => {{ tooltipContent += match + '\\n'; }});
                }}
                
                patchDiv.addEventListener('mouseenter', (e) => {{
                    tooltip.textContent = tooltipContent;
                    tooltip.style.display = 'block';
                }});
                
                patchDiv.addEventListener('mousemove', (e) => {{
                    tooltip.style.left = (e.pageX + 10) + 'px';
                    tooltip.style.top = (e.pageY + 10) + 'px';
                }});
                
                patchDiv.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});
                
                overlay.appendChild(patchDiv);
            }});
        }}
        
        const baseImage = document.getElementById('baseImage');
        if (baseImage.tagName === 'IMG') {{
            baseImage.addEventListener('load', createPatches);
            if (baseImage.complete) {{ createPatches(); }}
        }} else {{ createPatches(); }}
        
        window.addEventListener('resize', () => {{
            const overlay = document.getElementById('patchOverlay');
            overlay.innerHTML = '';
            setTimeout(createPatches, 100);
        }});
        
        // Toggle grid functionality
        const toggleButton = document.getElementById('toggleGrid');
        const patchOverlay = document.getElementById('patchOverlay');
        let gridVisible = true;
        
        toggleButton.addEventListener('click', () => {{
            gridVisible = !gridVisible;
            if (gridVisible) {{
                patchOverlay.style.display = 'block';
                toggleButton.textContent = 'Hide Grid';
                toggleButton.classList.remove('off');
            }} else {{
                patchOverlay.style.display = 'none';
                toggleButton.textContent = 'Show Grid';
                toggleButton.classList.add('off');
            }}
        }});
    </script>
</body>
</html>'''
        
        # Save HTML file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✓ Saved HTML visualization: {output_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error creating HTML visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create interactive visualization of nearest neighbors for PixMo-Cap or Mosaic data")
    
    parser.add_argument("--dataset", type=str, choices=["pixmo_cap", "mosaic"], default="pixmo_cap",
                       help="Which dataset type to visualize (default: pixmo_cap)")
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to the JSON results file from the generation script")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                       help="Which split to visualize (default: train)")
    parser.add_argument("--image-idx", type=int, default=None,
                       help="Specific image index to visualize (if not provided, creates visualizations for all images)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for HTML files (defaults to same directory as results file)")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process (default: 10)")
    
    args = parser.parse_args()
    
    # Set default results file based on dataset if not provided
    if args.results_file is None:
        checkpoint_path = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2/step12000-unsharded"
        ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
        
        if args.dataset == "pixmo_cap":
            args.results_file = f"analysis_results/nearest_neighbors/{ckpt_name}/nearest_neighbors_analysis_pixmo_cap.json"
        else:  # mosaic
            args.results_file = f"analysis_results/nearest_neighbors/{ckpt_name}/nearest_neighbors_analysis_color_names_mosaic_24x24.json"
    
    print(f"Dataset: {args.dataset}")
    print(f"Results file: {args.results_file}")
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file {results_file} does not exist")
        print(f"Expected path: {results_file}")
        print(f"\nMake sure you've run the analysis first:")
        print(f"python general_and_nearest_neighbors_pixmo_cap.py --dataset {args.dataset}")
        return
    
    print(f"Loading results from {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Verify dataset type matches
    results_dataset_type = results.get("dataset_type", "pixmo_cap")
    if results_dataset_type != args.dataset:
        print(f"Warning: Results file is for {results_dataset_type} but you specified {args.dataset}")
        print(f"Using dataset type from results file: {results_dataset_type}")
        args.dataset = results_dataset_type
    
    # Create preprocessor for image processing
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
            
            model_config.system_prompt_kind = "style" if args.dataset == "mosaic" else "none"
            
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
        output_dir = results_file.parent / f"interactive_visualizations_{args.dataset}"
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
            
            output_filename = f"interactive_nn_viewer_{args.dataset}_{args.split}_image_{idx:04d}.html"
            output_path = output_dir / output_filename
            
            # Try to create HTML with image background first
            success = False
            if images_dir:
                success = create_html_with_image_overlay(
                    image_data,
                    dataset_type=args.dataset,
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
                    dataset_type=args.dataset,
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
    index_file = output_dir / f"index_{args.dataset}_{args.split}.html"
    
    # Dataset-specific titles and descriptions
    if args.dataset == "pixmo_cap":
        dataset_title = "PixMo-Cap"
        description = "Hover over the colored patches to see the top 5 nearest neighbor words for each image patch. Green patches have nearest neighbors that match words in the caption, blue patches are visual/task related, and red patches are non-interpretable."
    else:  # mosaic
        dataset_title = "Mosaic Color Data"
        description = "Hover over the colored patches to see the top 5 nearest neighbor words for each image patch. Green patches have nearest neighbors that match the expected color for that position, red patches do not."
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Interactive Nearest Neighbors Viewer - {dataset_title} {args.split.title()} Split</title>
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
    <h1>Interactive Nearest Neighbors Viewer - {dataset_title} {args.split.title()} Split</h1>
    <p>Click on any link below to open the interactive visualization for that image.</p>
    <ul>'''
    
    for idx in indices_to_process:
        filename = f"interactive_nn_viewer_{args.dataset}_{args.split}_image_{idx:04d}.html"
        html_content += f'        <li><a href="{filename}">Image {idx:04d}</a></li>\n'
    
    html_content += f'''
    </ul>
    <p><strong>How to use:</strong> {description}</p>
</body>
</html>'''
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Index file created: {index_file}")

if __name__ == "__main__":
    main() 