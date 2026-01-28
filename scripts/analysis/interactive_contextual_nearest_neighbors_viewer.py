"""Interactive Web App for Viewing Contextual Nearest Neighbors

This script creates an interactive visualization where you can hover over 
image patches to see the top-5 nearest contextual text embeddings (real words in real sentences)
for each visual token.

Usage:
    python interactive_contextual_nearest_neighbors_viewer.py --results-file path/to/results.json --image-idx 5
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
from PIL import Image
import io

from olmo.data.pixmo_datasets import PixMoCap

log = logging.getLogger(__name__)

def escape_for_html(text: str) -> str:
    """Properly escape text for HTML."""
    if not text:
        return ""
    return html.escape(text, quote=True)

def patch_idx_to_row_col(patch_idx: int, patches_per_chunk: int) -> Tuple[int, int]:
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col

def pil_image_to_base64(img: Image.Image, preprocessor=None) -> str:
    """Convert PIL Image to base64 string for embedding in HTML, with optional preprocessing."""
    try:
        if img is None:
            return ""
        
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
                print(f"Applied preprocessing to image")
            except Exception as e:
                print(f"Could not preprocess image: {e}, using original")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        log.warning(f"Could not convert image to base64: {e}")
        return ""

def highlight_token_in_caption(caption: str, token_str: str, position: int) -> str:
    """
    Highlight the token in the caption with bold/color.
    
    Args:
        caption: Full caption text
        token_str: The token string to highlight
        position: Token position in the caption
    
    Returns:
        HTML string with highlighted token
    """
    # Escape the caption for HTML
    escaped_caption = escape_for_html(caption)
    escaped_token = escape_for_html(token_str)
    
    # Try to find and highlight the token
    # Simple approach: just wrap it in a span with styling
    highlighted = escaped_caption.replace(
        escaped_token, 
        f'<span style="background-color: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 3px;">{escaped_token}</span>',
        1  # Only replace first occurrence
    )
    
    return highlighted

def create_html_visualization(
    image_data: Dict,
    pil_image: Optional[Image.Image] = None,
    split_name: str = "unknown",
    output_path: Optional[str] = None,
    preprocessor=None
) -> bool:
    """Create an HTML file with image background and interactive overlay for contextual embeddings."""
    
    # Extract basic info
    image_idx = image_data.get("image_idx", 0)
    ground_truth = image_data.get("ground_truth_caption", "No caption available")
    
    # Get chunks and patches
    chunks = image_data.get("chunks", [])
    if not chunks:
        log.warning("No chunks found in image data")
        return False
    
    # Determine patches_per_chunk and grid_size
    patches_per_chunk = len(chunks[0].get("patches", []))
    grid_size = int(math.sqrt(patches_per_chunk))
    
    # Convert PIL image to base64
    image_base64 = ""
    if pil_image is not None:
        image_base64 = pil_image_to_base64(pil_image, preprocessor)
    
    # Prepare patch data
    patch_data = []
    for chunk_idx, chunk in enumerate(chunks):
        for patch in chunk.get("patches", []):
            patch_idx = patch.get("patch_idx", -1)
            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
            
            # Get nearest contextual neighbors
            nearest_contextual = patch.get("nearest_contextual_neighbors", [])
            
            # Format each neighbor with highlighted caption
            neighbors_html = []
            for i, neighbor in enumerate(nearest_contextual[:5]):
                token_str = neighbor.get("token_str", "")
                caption = neighbor.get("caption", "")
                position = neighbor.get("position", 0)
                similarity = neighbor.get("similarity", 0.0)
                
                # Highlight the token in the caption
                highlighted_caption = highlight_token_in_caption(caption, token_str, position)
                
                # Truncate caption if too long
                max_len = 120
                if len(caption) > max_len:
                    # Try to keep the token in view
                    escaped_caption_short = escape_for_html(caption[:max_len]) + "..."
                    if escape_for_html(token_str) in escaped_caption_short:
                        highlighted_caption = escaped_caption_short.replace(
                            escape_for_html(token_str),
                            f'<span style="background-color: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 3px;">{escape_for_html(token_str)}</span>',
                            1
                        )
                    else:
                        highlighted_caption = escaped_caption_short
                
                neighbor_html = f'''<div style="margin-bottom: 8px; padding: 6px; background-color: #f8f9fa; border-left: 3px solid #007bff; border-radius: 3px;">
                    <div style="font-weight: bold; color: #007bff; margin-bottom: 3px;">
                        {i+1}. Similarity: {similarity:.3f} (pos: {position})
                    </div>
                    <div style="font-size: 13px;">
                        {highlighted_caption}
                    </div>
                </div>'''
                neighbors_html.append(neighbor_html)
            
            # Default color (can be changed based on analysis if needed)
            color = "#6c757d"  # Neutral gray
            
            patch_data.append({
                "patch_idx": patch_idx,
                "row": row,
                "col": col,
                "color": color,
                "neighbors_html": neighbors_html
            })
    
    # Create HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Contextual Nearest Neighbors - Image {image_idx}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f0f2f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; border-bottom: 3px solid #007bff; padding-bottom: 20px; }}
        h1 {{ color: #2c3e50; margin: 0; font-size: 28px; }}
        h2 {{ color: #34495e; margin: 10px 0 0 0; font-size: 20px; font-weight: normal; }}
        .main-layout {{ display: flex; gap: 30px; }}
        .image-section {{ flex: 1; max-width: 600px; }}
        .info-section {{ flex: 1; min-width: 400px; }}
        .image-container {{ position: relative; display: inline-block; width: 100%; }}
        .base-image {{ width: 100%; max-width: 512px; border: 3px solid #34495e; border-radius: 8px; }}
        .patch-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
        .patch {{ position: absolute; border: 1px solid rgba(255,255,255,0.4); cursor: pointer; pointer-events: all; transition: all 0.2s; }}
        .patch:hover {{ border: 3px solid #fff; box-shadow: 0 0 10px rgba(255,255,255,0.8); z-index: 10; }}
        .patch.active {{ border: 3px solid #ffc107; box-shadow: 0 0 15px rgba(255,193,7,0.9); z-index: 10; }}
        .ground-truth {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .ground-truth strong {{ font-size: 16px; }}
        .controls {{ margin-bottom: 20px; display: flex; gap: 10px; justify-content: center; }}
        .toggle-button {{ background-color: #28a745; border: none; color: white; padding: 12px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 6px; transition: all 0.3s; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .toggle-button:hover {{ background-color: #218838; box-shadow: 0 4px 8px rgba(0,0,0,0.15); transform: translateY(-2px); }}
        .toggle-button.off {{ background-color: #dc3545; }}
        .toggle-button.off:hover {{ background-color: #c82333; }}
        .neighbors-panel {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 2px solid #dee2e6; min-height: 200px; }}
        .neighbors-panel h3 {{ margin-top: 0; color: #495057; font-size: 18px; }}
        .empty-state {{ text-align: center; color: #6c757d; padding: 40px 20px; }}
        .empty-state-icon {{ font-size: 48px; margin-bottom: 10px; }}
        .instructions {{ background-color: #e7f3ff; padding: 15px; border-radius: 6px; border-left: 4px solid #0066cc; margin-top: 20px; }}
        .instructions p {{ margin: 5px 0; color: #004080; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Interactive Contextual Nearest Neighbors</h1>
            <h2>Image {image_idx} ({split_name} split)</h2>
        </div>
        
        <div class="ground-truth">
            <strong>Ground Truth Caption:</strong><br>
            <span style="font-size: 15px; margin-top: 8px; display: block;">{escape_for_html(ground_truth)}</span>
        </div>
        
        <div class="main-layout">
            <div class="image-section">
                <div class="controls">
                    <button id="toggleGrid" class="toggle-button">Hide Grid</button>
                </div>
                <div class="image-container">'''
    
    if image_base64:
        html_content += f'<img src="{image_base64}" alt="Image {image_idx}" class="base-image" id="baseImage">'
    else:
        html_content += f'<div class="base-image" id="baseImage" style="width: 512px; height: 512px; background-color: #ddd; display: flex; align-items: center; justify-content: center; border-radius: 8px;">Image not available</div>'
    
    html_content += f'''
                    <div class="patch-overlay" id="patchOverlay"></div>
                </div>
                <div class="instructions">
                    <p><strong>Instructions:</strong> Click on any colored patch to see its top 5 nearest contextual embeddings (words in real captions) on the right.</p>
                    <p><strong>Grid:</strong> {grid_size}×{grid_size} patches ({patches_per_chunk} total)</p>
                </div>
            </div>
            
            <div class="info-section">
                <div class="neighbors-panel" id="neighborsPanel">
                    <div class="empty-state">
                        <div class="empty-state-icon">👆</div>
                        <p>Click on a patch to see its nearest contextual embeddings</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const patchData = {json.dumps(patch_data, indent=2)};
        const gridSize = {grid_size};
        let activePatchDiv = null;
        
        function createPatches() {{
            const baseImage = document.getElementById('baseImage');
            const overlay = document.getElementById('patchOverlay');
            
            const rect = baseImage.getBoundingClientRect();
            const imageWidth = rect.width;
            const imageHeight = rect.height;
            
            patchData.forEach(patch => {{
                const patchDiv = document.createElement('div');
                patchDiv.className = 'patch';
                patchDiv.dataset.patchIdx = patch.patch_idx;
                
                const patchWidth = imageWidth / gridSize;
                const patchHeight = imageHeight / gridSize;
                const left = patch.col * patchWidth;
                const top = patch.row * patchHeight;
                
                patchDiv.style.left = left + 'px';
                patchDiv.style.top = top + 'px';
                patchDiv.style.width = patchWidth + 'px';
                patchDiv.style.height = patchHeight + 'px';
                patchDiv.style.backgroundColor = patch.color;
                patchDiv.style.opacity = '0.3';
                
                patchDiv.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    
                    // Remove active class from previous patch
                    if (activePatchDiv) {{
                        activePatchDiv.classList.remove('active');
                    }}
                    
                    // Add active class to clicked patch
                    patchDiv.classList.add('active');
                    activePatchDiv = patchDiv;
                    
                    // Update neighbors panel
                    const neighborsPanel = document.getElementById('neighborsPanel');
                    let content = `<h3>Patch (${{patch.row}}, ${{patch.col}}) - Index ${{patch.patch_idx}}</h3>`;
                    content += `<div style="margin-top: 15px;">`;
                    
                    if (patch.neighbors_html.length > 0) {{
                        patch.neighbors_html.forEach(neighborHtml => {{
                            content += neighborHtml;
                        }});
                    }} else {{
                        content += '<p style="color: #6c757d;">No nearest neighbors found for this patch.</p>';
                    }}
                    
                    content += `</div>`;
                    neighborsPanel.innerHTML = content;
                }});
                
                overlay.appendChild(patchDiv);
            }});
        }}
        
        // Clear active patch when clicking outside
        document.addEventListener('click', (e) => {{
            if (!e.target.classList.contains('patch')) {{
                if (activePatchDiv) {{
                    activePatchDiv.classList.remove('active');
                    activePatchDiv = null;
                }}
            }}
        }});
        
        const baseImage = document.getElementById('baseImage');
        if (baseImage.tagName === 'IMG') {{
            baseImage.addEventListener('load', createPatches);
            if (baseImage.complete) {{ createPatches(); }}
        }} else {{
            createPatches();
        }}
        
        window.addEventListener('resize', () => {{
            const overlay = document.getElementById('patchOverlay');
            overlay.innerHTML = '';
            activePatchDiv = null;
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

def main():
    parser = argparse.ArgumentParser(description="Create interactive visualization of contextual nearest neighbors")
    
    parser.add_argument("--results-file", type=str, required=True,
                       help="Path to the JSON results file from contextual_nearest_neighbors.py")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Which split to visualize (default: validation)")
    parser.add_argument("--image-idx", type=int, default=None,
                       help="Specific image index to visualize (if not provided, creates visualizations for all images)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for HTML files (defaults to same directory as results file)")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process (default: 10)")
    
    args = parser.parse_args()
    
    print(f"Results file: {args.results_file}")
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file {results_file} does not exist")
        return
    
    print(f"Loading results from {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
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
            
            model_config.system_prompt_kind = "none"
            
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
        identifier = str(results_file.parent.name) + "_" + results_file.name
        output_dir = results_file.parent / f"interactive_visualizations_contextual_{identifier}"
    output_dir.mkdir(exist_ok=True)
    
    # Load PixMoCap dataset
    print(f"Loading PixMoCap dataset ({args.split} split)...")
    dataset = PixMoCap(split=args.split, mode="captions")
    print(f"Dataset loaded")
    
    # Get results data
    images = results.get("results", [])
    if not images:
        print(f"Error: No images found in results")
        return
    
    print(f"Found {len(images)} images")
    
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
        print(f"Creating visualization for image {idx}...")
        
        # Get image from dataset
        image_idx = image_data.get("image_idx", idx)
        print(f"  Loading image {image_idx} from dataset...")
        example_data = dataset.get(image_idx, np.random)
        image_data_raw = example_data.get("image")
        
        # Handle both PIL Image and file path string
        if isinstance(image_data_raw, str):
            # It's a file path, load it
            print(f"  Image is a file path: {image_data_raw}")
            pil_image = Image.open(image_data_raw)
        elif isinstance(image_data_raw, Image.Image):
            # Already a PIL Image
            print(f"  Image is already PIL Image")
            pil_image = image_data_raw
        else:
            print(f"  Unknown image type: {type(image_data_raw)}")
            pil_image = None
        
        if pil_image:
            print(f"  Successfully loaded image: mode={pil_image.mode}, size={pil_image.size}")
        
        output_filename = f"interactive_contextual_nn_image_{idx:04d}.html"
        output_path = output_dir / output_filename
        
        success = create_html_visualization(
            image_data,
            pil_image=pil_image,
            split_name=args.split,
            output_path=str(output_path),
            preprocessor=preprocessor
        )
        
        if success:
            print(f"✓ Saved: {output_path}")
        else:
            print(f"✗ Failed to create visualization for image {idx}")
    
    print(f"\nDone! Interactive visualizations saved to {output_dir}")
    print(f"Open the HTML files in a web browser to explore the contextual nearest neighbors interactively.")
    
    # Create an index file
    index_file = output_dir / "index.html"
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Interactive Contextual Nearest Neighbors Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f2f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #007bff; padding-bottom: 15px; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin: 10px 0; }}
        a {{ text-decoration: none; color: #007bff; padding: 12px 15px; border: 1px solid #dee2e6; display: block; border-radius: 6px; transition: all 0.3s; }}
        a:hover {{ background-color: #e7f3ff; border-color: #007bff; transform: translateX(5px); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Contextual Nearest Neighbors Viewer</h1>
        <p>Click on any link below to open the interactive visualization for that image.</p>
        <p><strong>How to use:</strong> Click on colored patches to see the top 5 contextual text embeddings (words in real captions) that are most similar to each visual patch. The matched words are highlighted in yellow.</p>
        <ul>'''
    
    for idx in indices_to_process:
        filename = f"interactive_contextual_nn_image_{idx:04d}.html"
        html_content += f'        <li><a href="{filename}">Image {idx:04d}</a></li>\n'
    
    html_content += '''
        </ul>
    </div>
</body>
</html>'''
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Index file created: {index_file}")

if __name__ == "__main__":
    main()

