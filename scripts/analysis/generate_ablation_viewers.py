#!/usr/bin/env python3
"""
Generate Ablation Viewers - MINIMAL ISOLATED SCRIPT

This script generates HTML viewers for ablation models.
It uses the EXACT SAME logic as the main create_unified_viewer.py
but is isolated so it can't break the main viewer.

Usage:
    # Generate all ablations
    python generate_ablation_viewers.py --output-dir analysis_results/unified_viewer_lite

    # Generate specific ablation
    python generate_ablation_viewers.py --output-dir analysis_results/unified_viewer_lite --only seed10

    # Generate just 1 image for testing
    python generate_ablation_viewers.py --output-dir analysis_results/unified_viewer_lite --only seed10 --num-images 1
"""

import json
import argparse
import sys
import math
import base64
import html
import io
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

# Import shared utility functions from viewer_lib
sys.path.insert(0, str(Path(__file__).parent))
from viewer_lib import (
    pil_image_to_base64,
    escape_for_html,
    create_preprocessor,
)

# Import HTML template from create_unified_viewer
from create_unified_viewer import create_unified_html_content

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load models config
CONFIG_PATH = Path(__file__).parent / "viewer_models.json"

def patch_idx_to_row_col(patch_idx: int, patches_per_chunk: int) -> Tuple[int, int]:
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def get_ablation_data_paths(ablation_config: Dict, base_dir: Path) -> Dict:
    """Get data paths for an ablation, handling special cases like Qwen2-VL."""
    
    if "data_paths" in ablation_config:
        # Explicit data paths provided
        return ablation_config["data_paths"]
    else:
        # Standard ablation - use checkpoint name
        checkpoint = ablation_config["checkpoint"]
        return {
            "nn": checkpoint,
            "logitlens": checkpoint,
            "contextual": checkpoint,
        }


def find_analysis_files(base_dir: Path, analysis_type: str, data_path: str) -> Dict[int, Path]:
    """Find analysis JSON files for a given type and path."""
    results = {}
    
    if analysis_type == "nn":
        search_dir = base_dir / "nearest_neighbors" / data_path
        patterns = ["nearest_neighbors_analysis_*_layer*.json", "nearest_neighbors_layer*_topk*.json"]
    elif analysis_type == "logitlens":
        search_dir = base_dir / "logit_lens" / data_path
        patterns = ["logit_lens_layer*_topk*.json"]
    elif analysis_type == "contextual":
        search_dir = base_dir / "contextual_nearest_neighbors" / data_path
        patterns = ["contextual_neighbors_visual*_allLayers.json"]
    else:
        return results
    
    if not search_dir.exists():
        return results
    
    for pattern in patterns:
        for json_file in search_dir.glob(pattern):
            try:
                if analysis_type == "contextual":
                    # Parse _visualN_ pattern
                    parts = json_file.stem.split("_")
                    for part in parts:
                        if part.startswith("visual") and len(part) > len("visual"):
                            layer = int(part.replace("visual", ""))
                            results[layer] = json_file
                            break
                elif "_layer" in json_file.stem:
                    # Parse _layerN_ pattern
                    layer_part = json_file.stem.split("_layer")[1]
                    layer = int(layer_part.split("_")[0])
                    results[layer] = json_file
            except (ValueError, IndexError):
                pass
    
    return results


def load_ablation_data(data_paths: Dict, base_dir: Path, num_images: int, split: str = "validation") -> Dict:
    """
    Load all analysis data for an ablation.
    
    CRITICAL: Different analysis types have different JSON structures!
    - NN: data["splits"][split]["images"] - list of images with chunks/patches
    - LogitLens: data["results"] - list of images with chunks/patches  
    - Contextual: data["results"] - list of images with patches
    
    This function VALIDATES that data was actually loaded and warns loudly if not.
    """
    
    all_data = {
        "nn": {},
        "logitlens": {},
        "contextual": {},
    }
    
    validation_errors = []
    
    # Load NN data - supports TWO formats:
    # Format A (main models/ablations): splits/validation/images with chunks/patches/nearest_neighbors
    # Format B (Qwen2-VL): results with patches/top_neighbors
    nn_files = find_analysis_files(base_dir, "nn", data_paths.get("nn", ""))
    log.info(f"    Loading {len(nn_files)} NN files...")
    t0 = time.time()
    for layer, json_path in nn_files.items():
        try:
            with open(json_path) as f:
                data = json.load(f)
                
                # Try Format A first (splits/validation/images)
                images = data.get("splits", {}).get(split, {}).get("images", [])
                data_format = "A"
                
                # Fallback to Format B (results directly)
                if not images:
                    images = data.get("results", [])
                    data_format = "B"
                
                if not images:
                    validation_errors.append(f"NN layer {layer}: No data in splits/{split}/images or results")
                    continue
                
                # Validate structure based on format
                if data_format == "A":
                    if "chunks" not in images[0]:
                        validation_errors.append(f"NN layer {layer}: Format A but no 'chunks' key")
                        continue
                elif data_format == "B":
                    if "patches" not in images[0]:
                        validation_errors.append(f"NN layer {layer}: Format B but no 'patches' key")
                        continue
                
                all_data["nn"][layer] = images[:num_images]
                # Store format info for later processing
                all_data["nn"][f"_format_{layer}"] = data_format
                log.info(f"      Layer {layer}: loaded {len(images[:num_images])} images (format {data_format})")
                
        except Exception as e:
            validation_errors.append(f"NN layer {layer}: Failed to load {json_path}: {e}")
    log.info(f"    ⏱️  NN: {time.time() - t0:.2f}s")
    
    # Load LogitLens data - supports TWO formats:
    # Format A: results with chunks/patches  
    # Format B (Qwen2-VL): results with patches directly
    logit_files = find_analysis_files(base_dir, "logitlens", data_paths.get("logitlens", ""))
    log.info(f"    Loading {len(logit_files)} LogitLens files...")
    t0 = time.time()
    for layer, json_path in logit_files.items():
        try:
            with open(json_path) as f:
                data = json.load(f)
                results = data.get("results", [])
                
                if not results:
                    validation_errors.append(f"LogitLens layer {layer}: File exists but no 'results' found")
                    continue
                
                # Determine format: chunks (A) or patches directly (B)
                if "chunks" in results[0]:
                    data_format = "A"
                elif "patches" in results[0]:
                    data_format = "B"
                else:
                    validation_errors.append(f"LogitLens layer {layer}: No 'chunks' or 'patches' key")
                    continue
                    
                all_data["logitlens"][layer] = results[:num_images]
                all_data["logitlens"][f"_format_{layer}"] = data_format
                log.info(f"      Layer {layer}: loaded {len(results[:num_images])} images (format {data_format})")
                
        except Exception as e:
            validation_errors.append(f"LogitLens layer {layer}: Failed to load {json_path}: {e}")
    log.info(f"    ⏱️  LogitLens: {time.time() - t0:.2f}s")
    
    # Load Contextual data - supports TWO formats:
    # Format A: results with chunks/patches  
    # Format B (Qwen2-VL): results with patches directly
    ctx_files = find_analysis_files(base_dir, "contextual", data_paths.get("contextual", ""))
    log.info(f"    Loading {len(ctx_files)} Contextual files...")
    t0 = time.time()
    for layer, json_path in ctx_files.items():
        try:
            with open(json_path) as f:
                data = json.load(f)
                results = data.get("results", [])
                
                if not results:
                    validation_errors.append(f"Contextual layer {layer}: File exists but no 'results' found")
                    continue
                
                # Determine format: chunks (A) or patches directly (B)
                if "chunks" in results[0]:
                    data_format = "A"
                elif "patches" in results[0]:
                    data_format = "B"
                else:
                    validation_errors.append(f"Contextual layer {layer}: No 'chunks' or 'patches' key")
                    continue
                    
                all_data["contextual"][layer] = results[:num_images]
                all_data["contextual"][f"_format_{layer}"] = data_format
                log.info(f"      Layer {layer}: loaded {len(results[:num_images])} images (format {data_format})")
                
        except Exception as e:
            validation_errors.append(f"Contextual layer {layer}: Failed to load {json_path}: {e}")
    log.info(f"    ⏱️  Contextual: {time.time() - t0:.2f}s")
    
    # Report all validation errors
    if validation_errors:
        log.error("  ❌ DATA LOADING VALIDATION ERRORS:")
        for err in validation_errors:
            log.error(f"    - {err}")
    
    return all_data


def validate_viewer_output(html_path: Path, expected_nn_layers: int, expected_logit_layers: int, expected_ctx_layers: int) -> bool:
    """
    BULLETPROOF VALIDATION: Check that generated HTML actually contains data.
    This catches the bug where data loading fails silently.
    """
    if not html_path.exists():
        log.error(f"  ❌ VALIDATION FAILED: {html_path} does not exist")
        return False
    
    with open(html_path, 'r') as f:
        content = f.read()
    
    # Check file size - empty data viewers are ~4KB, full ones are ~50KB+
    size_kb = len(content) / 1024
    if size_kb < 20:
        log.error(f"  ❌ VALIDATION FAILED: {html_path.name} is only {size_kb:.1f}KB - likely missing data!")
        return False
    
    errors = []
    
    # Check for nearest_neighbors data (should have actual tokens with similarity scores)
    if expected_nn_layers > 0:
        if '"nearest_neighbors": []' in content:
            errors.append("NN data has empty nearest_neighbors arrays!")
        if '"similarity":' not in content:
            errors.append("NN data missing similarity scores!")
    
    # Check for logitlens data (should have top_tokens with logit scores)
    if expected_logit_layers > 0:
        if '"top_tokens": []' in content:
            errors.append("LogitLens data has empty top_tokens arrays!")
    
    # Check for contextual data (should have contextual_neighbors)
    if expected_ctx_layers > 0:
        if '"contextual_neighbors": []' in content:
            errors.append("Contextual data has empty contextual_neighbors arrays!")
    
    if errors:
        log.error(f"  ❌ VALIDATION FAILED for {html_path.name}:")
        for err in errors:
            log.error(f"    - {err}")
        return False
    
    log.info(f"  ✅ Validation passed: {html_path.name} ({size_kb:.1f}KB)")
    return True


def create_ablation_model_index(output_dir: Path, ablation_config: Dict, 
                                 num_images: int, available_layers: Dict) -> None:
    """Create index.html for an ablation model."""
    
    display_name = ablation_config["name"]
    checkpoint = ablation_config["checkpoint"]
    
    model_dir = output_dir / "ablations" / checkpoint
    model_dir.mkdir(parents=True, exist_ok=True)
    
    nn_layers = available_layers.get("nn", [])
    logit_layers = available_layers.get("logitlens", [])
    ctx_layers = available_layers.get("contextual", [])
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{display_name} - Analysis Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px;
            background-color: #f0f2f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #9c27b0;
        }}
        h1 {{ color: #2c3e50; margin: 0 0 10px 0; }}
        .breadcrumb {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
            color: #666;
        }}
        .breadcrumb a {{ color: #9c27b0; text-decoration: none; }}
        .breadcrumb a:hover {{ text-decoration: underline; }}
        .model-info {{
            background-color: #f3e5f5;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #9c27b0;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
            text-align: center;
        }}
        .stat-label {{ font-size: 12px; color: #666; margin-bottom: 5px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #9c27b0; }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }}
        .image-card {{
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .image-card:hover {{
            background-color: #f3e5f5;
            border-color: #9c27b0;
            transform: translateY(-4px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .image-card a {{ text-decoration: none; color: #9c27b0; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="breadcrumb">
            <a href="../../index.html">← Back to Model Grid</a>
        </div>
        
        <div class="header">
            <h1>🔬 {display_name}</h1>
            <p style="color: #666;">Ablation Study</p>
        </div>
        
        <div class="model-info">
            <h3>Available Analyses</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Static NN</div>
                    <div class="stat-value">{len(nn_layers)}</div>
                    <div class="stat-label">layers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">LogitLens</div>
                    <div class="stat-value">{len(logit_layers)}</div>
                    <div class="stat-label">layers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Contextual NN</div>
                    <div class="stat-value">{len(ctx_layers)}</div>
                    <div class="stat-label">layers</div>
                </div>
            </div>
        </div>
        
        <h2>Select an Image to Explore</h2>
        <div class="image-grid">'''
    
    for img_idx in range(num_images):
        html_content += f'''
            <div class="image-card">
                <a href="image_{img_idx:04d}.html">Image {img_idx:04d}</a>
            </div>'''
    
    html_content += '''
        </div>
    </div>
</body>
</html>'''
    
    with open(model_dir / "index.html", 'w') as f:
        f.write(html_content)
    
    log.info(f"  Created model index: {model_dir / 'index.html'}")


def create_image_viewer(output_dir: Path, ablation_config: Dict,
                        image_idx: int, all_data: Dict,
                        dataset, split: str, preprocessor=None) -> bool:
    """Create image viewer for one image - uses SAME logic as main viewer."""

    checkpoint = ablation_config["checkpoint"]
    display_name = ablation_config["name"]
    model_dir = output_dir / "ablations" / checkpoint
    
    # Get data for this image (filter out _format_* metadata keys)
    image_data = {
        "nn": {},
        "logitlens": {},
        "contextual": {},
    }
    
    for analysis_type in ["nn", "logitlens", "contextual"]:
        for layer, images_list in all_data.get(analysis_type, {}).items():
            if str(layer).startswith("_format"):
                continue  # Skip format metadata
            if isinstance(images_list, list) and image_idx < len(images_list):
                image_data[analysis_type][layer] = images_list[image_idx]
    
    # Get available layers
    nn_layers = sorted(image_data["nn"].keys())
    logit_layers = sorted(image_data["logitlens"].keys())
    ctx_layers = sorted(image_data["contextual"].keys())
    
    if not nn_layers and not logit_layers and not ctx_layers:
        return False
    
    # Get image from dataset
    pil_image = None
    ground_truth = "Caption not available"
    image_base64 = ""
    
    try:
        example = dataset.get(image_idx, np.random)
        image_data_raw = example.get("image")
        
        if isinstance(image_data_raw, str):
            pil_image = Image.open(image_data_raw)
        elif isinstance(image_data_raw, Image.Image):
            pil_image = image_data_raw
        
        if pil_image:
            # Apply model-specific preprocessing (ViT=black padding, SigLIP/DINOv2=resize)
            # Uses same preprocessor as main viewer for consistency
            image_base64 = pil_image_to_base64(pil_image, preprocessor)
        
        ground_truth = example.get("caption", "No caption available")
    except Exception as e:
        log.warning(f"Could not load image {image_idx}: {e}")
    
    # Determine grid dimensions - handles both Format A (chunks) and Format B (patches directly)
    # For Qwen2-VL, grid is non-square and varies per image - use max row/col from data
    # IMPORTANT: Different analysis types may have different grids (e.g., NN=28x28, Contextual=16x16)
    # We need to find the MAXIMUM grid across all types to display all patches correctly
    patches_per_chunk = 576  # default
    max_row = 0
    max_col = 0
    
    # Check ALL analysis types to find max grid (don't break early!)
    for analysis_type in ["nn", "logitlens", "contextual"]:  # NN first - usually has most patches
        for layer, layer_data in image_data.get(analysis_type, {}).items():
            if str(layer).startswith("_format"):
                continue  # Skip format metadata
            
            # Get all patches
            all_patches = []
            if "chunks" in layer_data:
                for chunk in layer_data.get("chunks", []):
                    all_patches.extend(chunk.get("patches", []))
            elif "patches" in layer_data:
                all_patches = layer_data.get("patches", [])
            
            if all_patches:
                patches_per_chunk = max(patches_per_chunk, len(all_patches))
                # Find actual grid dimensions from patch row/col values
                for patch in all_patches:
                    max_row = max(max_row, patch.get("patch_row", 0))
                    max_col = max(max_col, patch.get("patch_col", 0))
                break  # Only need first layer per analysis type
    
    # Grid size is max(row, col) + 1 (since they're 0-indexed)
    # For square grids (main models), fall back to sqrt
    if max_row > 0 or max_col > 0:
        grid_rows = max_row + 1
        grid_cols = max_col + 1
        grid_size = max(grid_rows, grid_cols)  # Use larger dimension for positioning
    else:
        grid_size = int(math.sqrt(patches_per_chunk))
        grid_rows = grid_size
        grid_cols = grid_size
    
    # Build unified patch data structure
    unified_patch_data = {
        "nn": {},
        "logitlens": {},
        "contextual_vg": {},  # Must match JS template key (contextual_vg)
        "contextual_cc": {},  # Must match JS template key (contextual_cc)
    }
    
    # Process NN data - handles both Format A (chunks/patches) and Format B (patches directly)
    for layer, layer_data in image_data.get("nn", {}).items():
        if str(layer).startswith("_format"):
            continue  # Skip format metadata
        unified_patch_data["nn"][layer] = {}
        
        # Get patches based on format
        if "chunks" in layer_data:
            # Format A: chunks/patches structure
            all_patches = []
            for chunk in layer_data.get("chunks", []):
                all_patches.extend(chunk.get("patches", []))
            nn_key = "nearest_neighbors"
        else:
            # Format B: patches directly (Qwen2-VL style)
            all_patches = layer_data.get("patches", [])
            nn_key = "top_neighbors"  # Qwen2-VL uses different key
        
        for patch in all_patches:
            patch_idx = patch.get("patch_idx", -1)
            # Use row/col from patch data if available (Qwen2-VL has non-square grids)
            row = patch.get("patch_row", patch_idx // grid_size)
            col = patch.get("patch_col", patch_idx % grid_size)
            
            nn_list = []
            neighbors = patch.get(nn_key, []) or patch.get("nearest_neighbors", [])
            for i, nn in enumerate(neighbors[:5]):
                nn_list.append({
                    "rank": i + 1,
                    "token": escape_for_html(nn.get("token", "")),
                    "similarity": nn.get("similarity", 0.0)
                })
            
            unified_patch_data["nn"][layer][patch_idx] = {
                "row": row, "col": col,
                "neighbors": nn_list  # Must be "neighbors" to match JS template!
            }
    
    # Process LogitLens data - handles both Format A (chunks/patches) and Format B (patches directly)
    for layer, layer_data in image_data.get("logitlens", {}).items():
        if str(layer).startswith("_format"):
            continue  # Skip format metadata
        unified_patch_data["logitlens"][layer] = {}
        
        # Get patches based on format
        if "chunks" in layer_data:
            # Format A: chunks/patches structure
            all_patches = []
            for chunk in layer_data.get("chunks", []):
                all_patches.extend(chunk.get("patches", []))
        else:
            # Format B: patches directly (Qwen2-VL style)
            all_patches = layer_data.get("patches", [])
        
        for patch in all_patches:
            patch_idx = patch.get("patch_idx", -1)
            # Use row/col from patch data if available (Qwen2-VL has non-square grids)
            row = patch.get("patch_row", patch_idx // grid_size)
            col = patch.get("patch_col", patch_idx % grid_size)
            
            logit_list = []
            # LogitLens uses "top_predictions"
            for i, pred in enumerate(patch.get("top_predictions", [])[:5]):
                logit_list.append({
                    "rank": i + 1,
                    "token": escape_for_html(pred.get("token", "")),
                    "logit": pred.get("logit", 0.0)
                })
            
            unified_patch_data["logitlens"][layer][patch_idx] = {
                "row": row, "col": col,
                "predictions": logit_list  # Must be "predictions" to match JS template!
            }
    
    # Process Contextual data - handles both Format A (chunks/patches) and Format B (patches directly)
    for layer, layer_data in image_data.get("contextual", {}).items():
        if str(layer).startswith("_format"):
            continue  # Skip format metadata
        unified_patch_data["contextual_vg"][layer] = {}  # VG corpus for ablations
        
        # Get patches based on format
        if "chunks" in layer_data:
            # Format A: chunks/patches structure
            all_patches = []
            for chunk in layer_data.get("chunks", []):
                all_patches.extend(chunk.get("patches", []))
        else:
            # Format B: patches directly (Qwen2-VL style)
            all_patches = layer_data.get("patches", [])
        
        for patch in all_patches:
            patch_idx = patch.get("patch_idx", -1)
            # Use row/col from patch data if available (Qwen2-VL has non-square grids)
            row = patch.get("patch_row", patch_idx // grid_size)
            col = patch.get("patch_col", patch_idx % grid_size)
            
            ctx_list = []
            for i, neighbor in enumerate(patch.get("nearest_contextual_neighbors", [])[:5]):
                token_str = escape_for_html(neighbor.get("token_str", ""))
                caption = escape_for_html(neighbor.get("caption", ""))
                
                highlighted_caption = caption.replace(
                    token_str,
                    f'<span class="highlight">{token_str}</span>',
                    1
                )
                
                ctx_list.append({
                    "rank": i + 1,
                    "token": token_str,
                    "caption": highlighted_caption,
                    "similarity": neighbor.get("similarity", 0.0),
                    "contextual_layer": neighbor.get("contextual_layer", None)
                })
            
            unified_patch_data["contextual_vg"][layer][patch_idx] = {
                "row": row, "col": col,
                "contextual_neighbors": ctx_list
            }
    
    # Determine default layer
    all_layers = ctx_layers or nn_layers or logit_layers
    default_layer = all_layers[len(all_layers) // 2] if all_layers else 0
    
    # Generate HTML using the main viewer's function
    # Signature: (image_idx, image_base64, ground_truth, checkpoint_name, llm, ve,
    #             nn_layers, logit_layers, ctx_cc_layers, ctx_vg_layers,
    #             unified_patch_data, grid_size, patches_per_chunk, interpretability_map)
    # 
    # Get LLM and VE from ablation config or default to olmo-7b + vit-l-14-336
    llm = ablation_config.get("llm", "olmo-7b")
    ve = ablation_config.get("vision_encoder", "vit-l-14-336")
    
    try:
        html_content = create_unified_html_content(
            image_idx, image_base64, ground_truth,
            checkpoint,  # checkpoint_name
            llm, ve,  # Use actual llm/ve for lookup
            nn_layers, logit_layers, [], ctx_layers,  # ctx_cc=empty, ctx_vg=ctx_layers (VG corpus)
            unified_patch_data, grid_size, patches_per_chunk,
            {},  # interpretability_map empty
            grid_rows=grid_rows, grid_cols=grid_cols  # For non-square grids (Qwen2-VL)
        )
    except Exception as e:
        import traceback
        log.warning(f"Failed to create HTML for image {image_idx}: {e}")
        traceback.print_exc()
        return False
    
    # Fix breadcrumb path (ablations are 2 levels deep)
    html_content = html_content.replace(
        'href="../index.html"',
        'href="../../index.html"'
    )
    
    # Save
    output_file = model_dir / f"image_{image_idx:04d}.html"
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate ablation viewers")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                       help="Path to viewer_models.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory (existing unified viewer)")
    parser.add_argument("--only", type=str,
                       help="Only generate this ablation ID")
    parser.add_argument("--num-images", type=int, default=10,
                       help="Number of images to generate")
    parser.add_argument("--split", default="validation",
                       help="Dataset split")
    parser.add_argument("--force", action="store_true",
                       help="Force regenerate even if exists")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate data loading, don't generate viewers")
    parser.add_argument("--strict", action="store_true",
                       help="Fail immediately on any validation error")
    
    args = parser.parse_args()
    
    # Track overall success
    all_validation_passed = True
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    ablations = config.get("ablations", [])
    
    if args.only:
        ablations = [a for a in ablations if a["id"] == args.only or args.only in a.get("checkpoint", "")]
        if not ablations:
            print(f"ERROR: No ablation found matching '{args.only}'")
            sys.exit(1)
    
    log.info(f"Will generate viewers for {len(ablations)} ablation(s)")
    
    # Load dataset
    log.info("Loading PixMoCap dataset...")
    from olmo.data.pixmo_datasets import PixMoCap
    dataset = PixMoCap(split=args.split, mode="captions")
    log.info(f"Dataset loaded: {len(dataset)} examples")
    
    base_dir = Path("analysis_results")
    
    for ablation in ablations:
        abl_name = ablation["name"]
        abl_id = ablation["id"]
        checkpoint = ablation["checkpoint"]
        
        log.info(f"\n{'='*60}")
        log.info(f"Processing: {abl_name}")
        log.info(f"{'='*60}")
        
        # Check if already exists
        model_dir = args.output_dir / "ablations" / checkpoint
        if model_dir.exists() and not args.force:
            existing = list(model_dir.glob("image_*.html"))
            if len(existing) >= args.num_images:
                log.info(f"  ⏭️  Skipping (already has {len(existing)} images)")
                continue
        
        # Get data paths
        data_paths = get_ablation_data_paths(ablation, base_dir)
        
        # Load data
        log.info("  Loading analysis data...")
        all_data = load_ablation_data(data_paths, base_dir, args.num_images, args.split)
        
        # Get available layers (excluding _format_* metadata keys)
        available_layers = {
            "nn": sorted([k for k in all_data["nn"].keys() if not str(k).startswith("_format")]),
            "logitlens": sorted([k for k in all_data["logitlens"].keys() if not str(k).startswith("_format")]),
            "contextual": sorted([k for k in all_data["contextual"].keys() if not str(k).startswith("_format")]),
        }
        
        total_layers = sum(len(v) for v in available_layers.values())
        if total_layers == 0:
            log.warning(f"  ❌ No data found, skipping")
            continue
        
        log.info(f"  Found: NN={len(available_layers['nn'])}, LogitLens={len(available_layers['logitlens'])}, Contextual={len(available_layers['contextual'])}")
        
        # VALIDATION: Check that we have actual patch data, not just empty lists
        data_validation_ok = True
        for analysis_type, layers in all_data.items():
            for layer, images in layers.items():
                if str(layer).startswith("_format"):
                    continue  # Skip format metadata
                if not isinstance(images, list) or not images:
                    continue
                    
                # Check first image has actual content
                first_img = images[0]
                
                # Get patches based on format (A: chunks/patches, B: patches directly)
                if "chunks" in first_img:
                    chunks = first_img.get("chunks", [])
                    patches = chunks[0].get("patches", []) if chunks else []
                else:
                    patches = first_img.get("patches", [])
                
                if patches:
                    first_patch = patches[0]
                    if analysis_type == "nn":
                        # Try both key names (nearest_neighbors for Format A, top_neighbors for Format B)
                        neighbors = first_patch.get("nearest_neighbors", []) or first_patch.get("top_neighbors", [])
                        if not neighbors:
                            log.error(f"  ❌ VALIDATION: {analysis_type} layer {layer} has patches but empty neighbors!")
                            data_validation_ok = False
                    elif analysis_type == "logitlens":
                        tokens = first_patch.get("top_predictions", [])
                        if not tokens:
                            log.error(f"  ❌ VALIDATION: {analysis_type} layer {layer} has patches but empty top_predictions!")
                            data_validation_ok = False
                    elif analysis_type == "contextual":
                        neighbors = first_patch.get("nearest_contextual_neighbors", [])
                        if not neighbors:
                            log.error(f"  ❌ VALIDATION: {analysis_type} layer {layer} has patches but empty nearest_contextual_neighbors!")
                            data_validation_ok = False
        
        if not data_validation_ok:
            all_validation_passed = False
            if args.strict:
                log.error("  ❌ STRICT MODE: Failing due to validation errors")
                sys.exit(1)
        
        if args.validate_only:
            if data_validation_ok:
                log.info(f"  ✅ Data validation passed for {abl_name}")
            continue
        
        # Create model index
        create_ablation_model_index(args.output_dir, ablation, args.num_images, available_layers)

        # Create preprocessor for model-specific image preprocessing
        # ViT models use black padding, SigLIP/DINOv2 use resize
        preprocessor = create_preprocessor(checkpoint)

        # Create image viewers
        log.info(f"  Creating image viewers...")
        success = 0
        t0 = time.time()

        for img_idx in range(args.num_images):
            if create_image_viewer(args.output_dir, ablation, img_idx, all_data, dataset, args.split, preprocessor):
                success += 1
            
            if (img_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                rate = (img_idx + 1) / elapsed
                log.info(f"    Progress: {img_idx + 1}/{args.num_images} ({rate:.1f} img/s)")
        
        log.info(f"  ✅ Created {success}/{args.num_images} image viewers")
        
        # VALIDATE OUTPUT: Check first generated file has actual data
        first_html = args.output_dir / "ablations" / checkpoint / "image_0000.html"
        if first_html.exists():
            validate_viewer_output(
                first_html,
                len(available_layers['nn']),
                len(available_layers['logitlens']),
                len(available_layers['contextual'])
            )
    
    log.info("\n" + "="*60)
    if args.validate_only:
        if all_validation_passed:
            log.info("✅ ALL VALIDATIONS PASSED")
        else:
            log.error("❌ SOME VALIDATIONS FAILED - check errors above")
            sys.exit(1)
    else:
        log.info("DONE!")
    log.info("="*60)


if __name__ == "__main__":
    main()

