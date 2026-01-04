#!/usr/bin/env python3
"""Create Unified Interactive Viewer for All Analysis Results

This script scans all analysis results and creates a unified web interface with:
1. Main index: 2D grid of LLMs × Vision Encoders
2. Model pages: List of images for each model
3. Unified image viewer: Dropdowns to switch between layers and analysis types

Usage:
    python create_unified_viewer.py --output-dir analysis_results/unified_viewer --num-images 300
"""

import json
import argparse
import math
import base64
import html
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging

import numpy as np
from PIL import Image
import io

from olmo.data.pixmo_datasets import PixMoCap

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define model combinations
LLMS = ["llama3-8b", "olmo-7b", "qwen2-7b"]
VISION_ENCODERS = ["vit-l-14-336", "dinov2-large-336", "siglip"]

# LLM display names
LLM_DISPLAY_NAMES = {
    "llama3-8b": "LLaMA3-8B",
    "olmo-7b": "OLMo-7B",
    "qwen2-7b": "Qwen2-7B"
}

# Vision encoder display names
VISION_ENCODER_DISPLAY_NAMES = {
    "vit-l-14-336": "ViT-L/14-336",
    "dinov2-large-336": "DINOv2-L-336",
    "siglip": "SigLIP",
    "openvision2-l-14-336": "OpenVision2-L/14-336"
}

def escape_for_html(text: str) -> str:
    """Properly escape text for HTML."""
    if not text:
        return ""
    return html.escape(text, quote=True)

def get_checkpoint_name(llm: str, vision_encoder: str) -> str:
    """Get checkpoint name for a given LLM and vision encoder combination."""
    if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
        return f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_seed10"
    return f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}"

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
            except Exception as e:
                log.warning(f"Could not preprocess image: {e}, using original")
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        log.warning(f"Could not convert image to base64: {e}")
        return ""

def patch_idx_to_row_col(patch_idx: int, patches_per_chunk: int) -> Tuple[int, int]:
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col

def create_preprocessor(checkpoint_name: str):
    """Create preprocessor for a given checkpoint."""
    try:
        checkpoint_path = f"molmo_data/checkpoints/{checkpoint_name}/step12000-unsharded"
        
        log.info(f"    Creating preprocessor from checkpoint: {checkpoint_path}")
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
        log.info("    ✓ Preprocessor created successfully")
        return preprocessor
    except Exception as e:
        log.warning(f"    Could not create preprocessor: {e}, will use original images")
        return None

def scan_analysis_results(checkpoint_name: str, lite_suffix: str = "") -> Dict[str, Dict]:
    """Scan and collect all analysis results for a given checkpoint.
    
    Args:
        checkpoint_name: Name of the checkpoint
        lite_suffix: Optional suffix for lite directories (e.g., "_lite10")
    """
    results = {
        "nn": {},  # layer -> json path
        "logitlens": {},  # layer -> json path  
        "contextual_cc": {},  # layer -> json path (Conceptual Captions)
        "contextual_vg": {},  # layer -> json path (Visual Genome)
        "interpretability": None  # path to interpretability heuristic JSON
    }
    
    base_dir = Path("analysis_results")
    
    # Scan nearest neighbors
    nn_dir = base_dir / "nearest_neighbors" / f"{checkpoint_name}_step12000-unsharded{lite_suffix}"
    if nn_dir.exists():
        for json_file in nn_dir.glob("nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer*.json"):
            # Extract layer number from filename
            layer_str = json_file.stem.split("_layer")[-1]
            if layer_str.isdigit():
                layer = int(layer_str)
                results["nn"][layer] = json_file
    
    # Scan logit lens (note: directory is "logit_lens" not "logitlens")
    logit_dir = base_dir / "logit_lens" / f"{checkpoint_name}_step12000-unsharded{lite_suffix}"
    if logit_dir.exists():
        for json_file in logit_dir.glob("logit_lens_layer*_topk5_multi-gpu.json"):
            # Extract layer number
            try:
                layer_str = json_file.stem.split("layer")[1].split("_")[0]
                layer = int(layer_str)
                results["logitlens"][layer] = json_file
            except:
                pass
    
    # NOTE: Conceptual Captions (CC) corpus was removed - now only using Visual Genome (VG)
    # The contextual_cc dict is kept empty for backward compatibility with the template
    
    # Scan contextual nearest neighbors (Visual Genome corpus)
    # Note: folder is now just "contextual_nearest_neighbors" (CC data was removed)
    # New format: contextual_neighbors_visual*_allLayers.json (one file per visual layer, contains all contextual layers)
    contextual_vg_dir = base_dir / "contextual_nearest_neighbors" / f"{checkpoint_name}_step12000-unsharded{lite_suffix}"
    if contextual_vg_dir.exists():
        # Try new allLayers format first
        allLayers_files = list(contextual_vg_dir.glob("contextual_neighbors_visual*_allLayers.json"))
        if allLayers_files:
            for json_file in allLayers_files:
                try:
                    # Extract visual layer from filename
                    parts = json_file.stem.split("_")
                    for part in parts:
                        if part.startswith("visual") and len(part) > len("visual"):
                            visual_layer = int(part.replace("visual", ""))
                            # Use visual layer as key, file contains all contextual layers
                            results["contextual_vg"][visual_layer] = json_file
                            break
                except Exception as e:
                    log.warning(f"Could not parse contextual VG file {json_file.name}: {e}")
        else:
            # Fallback to old format: contextual_neighbors_visual*_contextual*_multi-gpu.json
            for json_file in contextual_vg_dir.glob("contextual_neighbors_visual*_contextual*_multi-gpu.json"):
                try:
                    parts = json_file.stem.split("_")
                    visual_layer = None
                    contextual_layer = None
                    for part in parts:
                        if part.startswith("visual") and len(part) > len("visual"):
                            visual_layer = int(part.replace("visual", ""))
                        elif part.startswith("contextual") and len(part) > len("contextual"):
                            contextual_layer = int(part.replace("contextual", ""))
                    if visual_layer is not None and contextual_layer is not None:
                        results["contextual_vg"][contextual_layer] = json_file
                except Exception as e:
                    log.warning(f"Could not parse contextual VG file {json_file.name}: {e}")
    
    # Scan interpretability heuristic (single file per model, no lite suffix for now)
    interpretability_dir = base_dir / "interpretability_heuristic" / f"{checkpoint_name}_step12000-unsharded"
    if interpretability_dir.exists():
        # Look for interpretability_heuristic_nn0_contextual8_threshold1.5.json
        for json_file in interpretability_dir.glob("interpretability_heuristic_*.json"):
            results["interpretability"] = json_file
            break  # Only take the first one
    
    return results

def create_main_index(output_dir: Path, model_availability: Dict) -> None:
    """Create the main index page with LLM × Vision Encoder grid."""
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Model Analysis Viewer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            font-size: 36px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
            font-size: 18px;
        }
        .grid-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            padding: 20px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            font-size: 16px;
        }
        th:first-child {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        td:first-child {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        .model-cell {
            position: relative;
            cursor: pointer;
            background-color: #ffffff;
            transition: all 0.3s ease;
            min-height: 80px;
        }
        .model-cell:hover {
            background-color: #e3f2fd;
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .model-cell.available {
            background-color: #e8f5e9;
        }
        .model-cell.available:hover {
            background-color: #c8e6c9;
        }
        .model-link {
            text-decoration: none;
            color: #1976d2;
            font-weight: 500;
            display: block;
            padding: 10px;
        }
        .model-link:hover {
            color: #0d47a1;
        }
        .unavailable {
            color: #bdbdbd;
            font-style: italic;
        }
        .stats {
            font-size: 12px;
            color: #666;
            margin-top: 8px;
        }
        .legend {
            margin-top: 30px;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .legend-color {
            width: 30px;
            height: 30px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .info {
            background-color: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin-bottom: 30px;
        }
        .info h3 {
            margin-top: 0;
            color: #1976d2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Unified Model Analysis Viewer</h1>
        <div class="subtitle">Interactive visualization of vision-language model internals</div>
        
        <div class="info">
            <h3>About This Viewer</h3>
            <p>This unified interface provides interactive access to three types of analyses across multiple model combinations:</p>
            <ul>
                <li><strong>Embedding Matrix:</strong> Top-5 nearest vocabulary tokens for each visual patch</li>
                <li><strong>Logit Lens:</strong> Predicted tokens by applying LM head to intermediate layers</li>
                <li><strong>LN-Lens:</strong> Nearest contextual embeddings (words in real captions)</li>
            </ul>
            <p>Click on any available model cell below to explore its analysis results.</p>
        </div>
        
        <div class="grid-container">
            <table>
                <thead>
                    <tr>
                        <th>LLM \ Vision Encoder</th>'''
    
    # Add vision encoder headers
    for ve in VISION_ENCODERS:
        html_content += f'<th>{VISION_ENCODER_DISPLAY_NAMES[ve]}</th>'
    
    html_content += '''
                    </tr>
                </thead>
                <tbody>'''
    
    # Add rows for each LLM
    for llm in LLMS:
        html_content += f'''
                    <tr>
                        <td>{LLM_DISPLAY_NAMES[llm]}</td>'''
        
        for ve in VISION_ENCODERS:
            checkpoint_name = get_checkpoint_name(llm, ve)
            model_key = (llm, ve)
            
            if model_key in model_availability and model_availability[model_key]["available"]:
                stats = model_availability[model_key]
                nn_count = len(stats["nn_layers"])
                logit_count = len(stats["logit_layers"])
                ctx_cc_count = len(stats["contextual_cc_layers"])
                ctx_vg_count = len(stats["contextual_vg_layers"])
                ctx_count = max(ctx_cc_count, ctx_vg_count)  # Show the max of either
                
                html_content += f'''
                        <td class="model-cell available">
                            <a href="{checkpoint_name}/index.html" class="model-link">
                                <div>View Results</div>
                                <div class="stats">
                                    NN: {nn_count} layers<br>
                                    Logit: {logit_count} layers<br>
                                    Ctx: {ctx_count} layers
                                </div>
                            </a>
                        </td>'''
            else:
                html_content += '''
                        <td class="model-cell">
                            <span class="unavailable">No results</span>
                        </td>'''
        
        html_content += '''
                    </tr>'''
    
    html_content += '''
                </tbody>
            </table>
                    </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #e8f5e9;"></div>
                <span>Available (click to view)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffffff;"></div>
                <span>No results available</span>
            </div>
            </div>
    </div>
</body>
</html>'''
    
    # Save main index
    index_file = output_dir / "index.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    log.info(f"Created main index: {index_file}")

def create_model_index(output_dir: Path, checkpoint_name: str, llm: str, ve: str, 
                       num_images: int, analysis_results: Dict) -> None:
    """Create index page for a specific model listing all images."""
    
    model_dir = output_dir / checkpoint_name
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Count available layers for each analysis type
    nn_layers = sorted(analysis_results["nn"].keys())
    logit_layers = sorted(analysis_results["logitlens"].keys())
    ctx_cc_layers = sorted(analysis_results["contextual_cc"].keys())
    ctx_vg_layers = sorted(analysis_results["contextual_vg"].keys())
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{LLM_DISPLAY_NAMES[llm]} + {VISION_ENCODER_DISPLAY_NAMES[ve]} - Analysis Results</title>
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
            border-bottom: 3px solid #007bff;
        }}
        h1 {{
            color: #2c3e50;
            margin: 0 0 10px 0;
        }}
        .breadcrumb {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
            color: #666;
        }}
        .breadcrumb a {{
            color: #007bff;
            text-decoration: none;
        }}
        .breadcrumb a:hover {{
            text-decoration: underline;
        }}
        .model-info {{
            background-color: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
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
            background-color: #e3f2fd;
            border-color: #007bff;
            transform: translateY(-4px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .image-card a {{
            text-decoration: none;
            color: #007bff;
            font-weight: 500;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="breadcrumb">
            <a href="../index.html">← Back to Model Grid</a>
        </div>
        
        <div class="header">
            <h1>{LLM_DISPLAY_NAMES[llm]} + {VISION_ENCODER_DISPLAY_NAMES[ve]}</h1>
            <p style="color: #666;">Interactive Analysis Results</p>
        </div>
        
        <div class="model-info">
            <h3>Available Analyses</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Embedding Matrix</div>
                    <div class="stat-value">{len(nn_layers)}</div>
                    <div class="stat-label">Layers: {", ".join(map(str, nn_layers)) if nn_layers else "None"}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Logit Lens</div>
                    <div class="stat-value">{len(logit_layers)}</div>
                    <div class="stat-label">Layers: {", ".join(map(str, logit_layers)) if logit_layers else "None"}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">LN-Lens</div>
                    <div class="stat-value">{len(ctx_vg_layers)}</div>
                    <div class="stat-label">Layers: {", ".join(map(str, ctx_vg_layers)) if ctx_vg_layers else "None"}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">LN-Lens (CC)</div>
                    <div class="stat-value">{len(ctx_cc_layers)}</div>
                    <div class="stat-label">Layers: {", ".join(map(str, ctx_cc_layers)) if ctx_cc_layers else "None"}</div>
                </div>
            </div>
        </div>
        
        <h2>Select an Image to Explore</h2>
        <div class="image-grid">'''
    
    # Add links for each image
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
    
    # Save model index
    index_file = model_dir / "index.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    log.info(f"Created model index: {index_file}")

def load_all_analysis_data(analysis_results: Dict, split: str, num_images: int) -> Dict:
    """Load ALL analysis data from JSON files at once (much faster than loading per image)."""
    log.info(f"  📦 Loading all analysis data at once (this is much faster)...")
    start = time.time()
    
    # Structure: {analysis_type: {layer: [image0_data, image1_data, ...]}}
    all_data = {
        "nn": {},
        "logitlens": {},
        "contextual_cc": {},
        "contextual_vg": {},
        "interpretability": []  # Single list of image data
    }
    
    # Load nearest neighbors data
    log.info(f"    Loading {len(analysis_results['nn'])} NN files...")
    t0 = time.time()
    for layer, json_path in analysis_results["nn"].items():
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)
                images = full_data.get("splits", {}).get(split, {}).get("images", [])
                all_data["nn"][layer] = images[:num_images]
        except Exception as e:
            log.warning(f"Could not load NN data for layer {layer}: {e}")
    log.info(f"    ⏱️  NN: {time.time() - t0:.2f}s")
    
    # Load logit lens data
    log.info(f"    Loading {len(analysis_results['logitlens'])} Logit Lens files...")
    t0 = time.time()
    for layer, json_path in analysis_results["logitlens"].items():
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)
                results = full_data.get("results", [])
                all_data["logitlens"][layer] = results[:num_images]
        except Exception as e:
            log.warning(f"Could not load logit lens data for layer {layer}: {e}")
    log.info(f"    ⏱️  Logit Lens: {time.time() - t0:.2f}s")
    
    # Load contextual NN data - CC (Conceptual Captions)
    log.info(f"    Loading {len(analysis_results['contextual_cc'])} LN-Lens (CC) files...")
    t0 = time.time()
    for layer, json_path in analysis_results["contextual_cc"].items():
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)
                results = full_data.get("results", [])
                all_data["contextual_cc"][layer] = results[:num_images]
        except Exception as e:
            log.warning(f"Could not load contextual NN (CC) data for layer {layer}: {e}")
    log.info(f"    ⏱️  LN-Lens (CC): {time.time() - t0:.2f}s")
    
    # Load contextual NN data - VG (Visual Genome)
    # Note: allLayers format has one file per visual layer, containing all contextual layers
    # We use visual_layer=0 for the viewer (input layer representations)
    log.info(f"    Loading {len(analysis_results['contextual_vg'])} LN-Lens files...")
    t0 = time.time()
    
    # Find visual_layer=0 file (preferred) or use first available
    target_visual_layer = 0
    if target_visual_layer not in analysis_results["contextual_vg"]:
        if analysis_results["contextual_vg"]:
            target_visual_layer = min(analysis_results["contextual_vg"].keys())
            log.info(f"    Using visual_layer={target_visual_layer} (visual_layer=0 not found)")
    
    if target_visual_layer in analysis_results["contextual_vg"]:
        json_path = analysis_results["contextual_vg"][target_visual_layer]
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)
                results = full_data.get("results", [])[:num_images]
                contextual_layers = full_data.get("contextual_layers_used", [])
                
                # Store results under each contextual layer key
                # The neighbors in each patch are tagged with their contextual_layer
                for ctx_layer in contextual_layers:
                    all_data["contextual_vg"][ctx_layer] = results
                
                log.info(f"    Loaded visual_layer={target_visual_layer} with contextual_layers={contextual_layers}")
        except Exception as e:
            log.warning(f"Could not load contextual NN (VG) data: {e}")
    
    log.info(f"    ⏱️  LN-Lens: {time.time() - t0:.2f}s")
    
    # Load interpretability data
    if analysis_results.get("interpretability"):
        log.info(f"    Loading interpretability heuristic...")
        t0 = time.time()
        try:
            with open(analysis_results["interpretability"], 'r') as f:
                full_data = json.load(f)
                results = full_data.get("results", [])
                all_data["interpretability"] = results[:num_images]
                log.info(f"    ⏱️  Interpretability: {time.time() - t0:.2f}s")
        except Exception as e:
            log.warning(f"Could not load interpretability data: {e}")
    else:
        log.info(f"    No interpretability data available for this model")
    
    log.info(f"  ✅ Total load time: {time.time() - start:.2f}s")
    
    return all_data

def get_image_data_from_cache(all_data_cache: Dict, image_idx: int) -> Dict:
    """Extract data for a single image from the pre-loaded cache."""
    image_data = {
        "nn": {},
        "logitlens": {},
        "contextual_cc": {},
        "contextual_vg": {},
        "interpretability": None
    }
    
    for analysis_type in ["nn", "logitlens", "contextual_cc", "contextual_vg"]:
        for layer, images_list in all_data_cache[analysis_type].items():
            if image_idx < len(images_list):
                image_data[analysis_type][layer] = images_list[image_idx]
    
    # Get interpretability data for this image
    if all_data_cache.get("interpretability") and image_idx < len(all_data_cache["interpretability"]):
        image_data["interpretability"] = all_data_cache["interpretability"][image_idx]
    
    return image_data

def create_unified_image_viewer(output_dir: Path, checkpoint_name: str, llm: str, ve: str,
                                image_idx: int, all_data_cache: Dict, dataset, split: str,
                                preprocessor=None) -> bool:
    """Create unified HTML viewer for a single image with dynamic layer/analysis switching."""
    
    model_dir = output_dir / checkpoint_name
    
    # Extract data for this image from the cache
    all_data = get_image_data_from_cache(all_data_cache, image_idx)
    
    # Get available layers for each analysis type
    nn_layers = sorted(all_data["nn"].keys())
    logit_layers = sorted(all_data["logitlens"].keys())
    ctx_cc_layers = sorted(all_data["contextual_cc"].keys())
    ctx_vg_layers = sorted(all_data["contextual_vg"].keys())
    
    if not nn_layers and not logit_layers and not ctx_cc_layers and not ctx_vg_layers:
        if image_idx % 50 == 0:
            log.warning(f"  No data found for image {image_idx}, skipping")
        return False
    
    # Get image from dataset
    try:
        example_data = dataset.get(image_idx, np.random)
        image_data_raw = example_data.get("image")
        
        if isinstance(image_data_raw, str):
            pil_image = Image.open(image_data_raw)
        elif isinstance(image_data_raw, Image.Image):
            pil_image = image_data_raw
        else:
            pil_image = None
        
        # Apply model-specific preprocessing
        image_base64 = pil_image_to_base64(pil_image, preprocessor) if pil_image else ""
        
        # Get ground truth caption
        ground_truth = example_data.get("caption", "No caption available")
    except Exception as e:
        if image_idx % 50 == 0:
            log.warning(f"  Could not load image {image_idx}: {e}")
        image_base64 = ""
        ground_truth = "Caption not available"
    
    # Get grid size from first available data
    grid_size = 16  # default
    patches_per_chunk = 256  # default
    
    for analysis_type in ["nn", "logitlens", "contextual_vg", "contextual_cc"]:
        if all_data[analysis_type]:
            first_layer_data = list(all_data[analysis_type].values())[0]
            chunks = first_layer_data.get("chunks", [])
            if chunks:
                patches_per_chunk = len(chunks[0].get("patches", []))
                grid_size = int(math.sqrt(patches_per_chunk))
                break
    
    # Prepare unified patch data structure
    # Format: {analysis_type: {layer: {patch_idx: patch_data}}}
    unified_patch_data = {
        "nn": {},
        "logitlens": {},
        "contextual_vg": {},
        "contextual_cc": {}
    }
    
    # Extract interpretability data (mapping of patch_idx -> interpretability info)
    interpretability_map = {}  # patch_idx -> {interpretable: 0/1, ratio: float, ...}
    if all_data.get("interpretability"):
        chunks = all_data["interpretability"].get("chunks", [])
        for chunk in chunks:
            for patch in chunk.get("patches", []):
                patch_idx = patch.get("patch_idx", -1)
                interpretability_map[patch_idx] = {
                    "interpretable": patch.get("interpretable", 0),
                    "ratio": patch.get("ratio", 0.0),
                    "nn_top_similarity": patch.get("nn_top_similarity", 0.0),
                    "contextual_top_similarity": patch.get("contextual_top_similarity", 0.0)
                }
    
    # Process NN data
    for layer, image_data in all_data["nn"].items():
        unified_patch_data["nn"][layer] = {}
        chunks = image_data.get("chunks", [])
        for chunk in chunks:
            for patch in chunk.get("patches", []):
                patch_idx = patch.get("patch_idx", -1)
                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                
                nearest_neighbors = patch.get("nearest_neighbors", [])
                nn_list = []
                for i, nn in enumerate(nearest_neighbors[:5]):
                    token = escape_for_html(nn.get("token", ""))
                    similarity = nn.get("similarity", 0.0)
                    nn_list.append({
                        "rank": i + 1,
                        "token": token,
                        "similarity": similarity
                    })
                
                unified_patch_data["nn"][layer][patch_idx] = {
                    "row": row,
                    "col": col,
                    "neighbors": nn_list
                }
    
    # Process Logit Lens data
    for layer, image_data in all_data["logitlens"].items():
        unified_patch_data["logitlens"][layer] = {}
        chunks = image_data.get("chunks", [])
        for chunk in chunks:
            for patch in chunk.get("patches", []):
                patch_idx = patch.get("patch_idx", -1)
                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                
                top_predictions = patch.get("top_predictions", [])
                pred_list = []
                for i, pred in enumerate(top_predictions[:5]):
                    token = escape_for_html(pred.get("token", ""))
                    logit = pred.get("logit", 0.0)
                    token_id = pred.get("token_id", 0)
                    pred_list.append({
                        "rank": i + 1,
                        "token": token,
                        "logit": logit,
                        "token_id": token_id
                    })
                
                unified_patch_data["logitlens"][layer][patch_idx] = {
                    "row": row,
                    "col": col,
                    "predictions": pred_list
                }
    
    # Process LN-Lens data - VG (Visual Genome corpus)
    # Note: allLayers format has neighbors from all contextual layers mixed together,
    # each neighbor is tagged with its contextual_layer
    # The layer dropdown is for VISUAL layers (we use visual_layer=0), not contextual layers
    # We show ALL top-5 contextual neighbors with badges indicating which contextual layer they came from
    for layer, image_data in all_data["contextual_vg"].items():
        unified_patch_data["contextual_vg"][layer] = {}
        chunks = image_data.get("chunks", [])
        for chunk in chunks:
            for patch in chunk.get("patches", []):
                patch_idx = patch.get("patch_idx", -1)
                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)

                # Get ALL top-5 contextual neighbors (don't filter by contextual_layer!)
                # Each neighbor already has contextual_layer in its data for badge display
                all_neighbors = patch.get("nearest_contextual_neighbors", [])
                ctx_list = []
                for i, neighbor in enumerate(all_neighbors[:5]):
                    token_str = escape_for_html(neighbor.get("token_str", ""))
                    caption = escape_for_html(neighbor.get("caption", ""))
                    position = neighbor.get("position", 0)
                    similarity = neighbor.get("similarity", 0.0)
                    contextual_layer = neighbor.get("contextual_layer", -1)  # Get from neighbor data!

                    # Highlight token in caption
                    highlighted_caption = caption.replace(
                        token_str,
                        f'<span class="highlight">{token_str}</span>',
                        1
                    )

                    # Get additional analysis data
                    lowest_sim = neighbor.get("lowest_similarity_same_token", None)
                    inter_nn_sims = neighbor.get("similarity_to_other_nns", None)

                    ctx_entry = {
                        "rank": i + 1,
                        "token": token_str,
                        "caption": highlighted_caption,
                        "position": position,
                        "similarity": similarity,
                        "contextual_layer": contextual_layer  # Use layer from neighbor data for badge
                    }
                    
                    # Add optional fields if they exist
                    if lowest_sim is not None:
                        # Escape HTML in lowest similarity caption
                        lowest_caption = escape_for_html(lowest_sim.get("caption", ""))
                        ctx_entry["lowest_similarity_same_token"] = {
                            "token_str": escape_for_html(lowest_sim.get("token_str", "")),
                            "caption": lowest_caption,
                            "position": lowest_sim.get("position", 0),
                            "similarity": lowest_sim.get("similarity", 0.0),
                            "num_instances": lowest_sim.get("num_instances", 0)
                        }
                    
                    if inter_nn_sims is not None:
                        ctx_entry["similarity_to_other_nns"] = inter_nn_sims
                    
                    ctx_list.append(ctx_entry)
                
                unified_patch_data["contextual_vg"][layer][patch_idx] = {
                    "row": row,
                    "col": col,
                    "contextual_neighbors": ctx_list
                }
    
    # Process LN-Lens data - CC (Conceptual Captions corpus, legacy)
    for layer, image_data in all_data["contextual_cc"].items():
        unified_patch_data["contextual_cc"][layer] = {}
        chunks = image_data.get("chunks", [])
        for chunk in chunks:
            for patch in chunk.get("patches", []):
                patch_idx = patch.get("patch_idx", -1)
                row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                
                nearest_contextual = patch.get("nearest_contextual_neighbors", [])
                ctx_list = []
                for i, neighbor in enumerate(nearest_contextual[:5]):
                    token_str = escape_for_html(neighbor.get("token_str", ""))
                    caption = escape_for_html(neighbor.get("caption", ""))
                    position = neighbor.get("position", 0)
                    similarity = neighbor.get("similarity", 0.0)
                    
                    # Highlight token in caption
                    highlighted_caption = caption.replace(
                        token_str,
                        f'<span class="highlight">{token_str}</span>',
                        1
                    )
                    
                    # Get additional analysis data
                    lowest_sim = neighbor.get("lowest_similarity_same_token", None)
                    inter_nn_sims = neighbor.get("similarity_to_other_nns", None)
                    
                    ctx_entry = {
                        "rank": i + 1,
                        "token": token_str,
                        "caption": highlighted_caption,
                        "position": position,
                        "similarity": similarity
                    }
                    
                    # Add optional fields if they exist
                    if lowest_sim is not None:
                        # Escape HTML in lowest similarity caption
                        lowest_caption = escape_for_html(lowest_sim.get("caption", ""))
                        ctx_entry["lowest_similarity_same_token"] = {
                            "token_str": escape_for_html(lowest_sim.get("token_str", "")),
                            "caption": lowest_caption,
                            "position": lowest_sim.get("position", 0),
                            "similarity": lowest_sim.get("similarity", 0.0),
                            "num_instances": lowest_sim.get("num_instances", 0)
                        }
                    
                    if inter_nn_sims is not None:
                        ctx_entry["similarity_to_other_nns"] = inter_nn_sims
                    
                    ctx_list.append(ctx_entry)
                
                unified_patch_data["contextual_cc"][layer][patch_idx] = {
                    "row": row,
                    "col": col,
                    "contextual_neighbors": ctx_list
                }
    
    # Now create the HTML - this will be a continuation in next message due to length
    html_content = create_unified_html_content(
        image_idx, image_base64, ground_truth, checkpoint_name, llm, ve,
        nn_layers, logit_layers, ctx_cc_layers, ctx_vg_layers, unified_patch_data, grid_size, patches_per_chunk,
        interpretability_map
    )
    
    # Save HTML file
    output_file = model_dir / f"image_{image_idx:04d}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return True

def create_unified_html_content(image_idx: int, image_base64: str, ground_truth: str,
                                checkpoint_name: str, llm: str, ve: str,
                                nn_layers: List, logit_layers: List, ctx_cc_layers: List, ctx_vg_layers: List,
                                unified_patch_data: Dict, grid_size: int, patches_per_chunk: int,
                                interpretability_map: Dict,
                                grid_rows: int = None, grid_cols: int = None) -> str:
    """Generate the HTML content for the unified viewer.
    
    Args:
        grid_rows, grid_cols: Optional. For non-square grids (e.g., Qwen2-VL), specify 
                              the actual row/col dimensions. If not provided, assumes square grid.
    """
    # For non-square grids, use actual dimensions; otherwise assume square
    if grid_rows is None:
        grid_rows = grid_size
    if grid_cols is None:
        grid_cols = grid_size
    
    # Find all unique layers across all analysis types
    all_layers = sorted(set(nn_layers + logit_layers + ctx_cc_layers + ctx_vg_layers))
    default_layer = all_layers[len(all_layers) // 2] if all_layers else 0
    
    # Start HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image {image_idx:04d} - {LLM_DISPLAY_NAMES[llm]} + {VISION_ENCODER_DISPLAY_NAMES[ve]}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .breadcrumb {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
            color: #666;
        }}
        .breadcrumb a {{
            color: #007bff;
            text-decoration: none;
            margin: 0 5px;
        }}
        .breadcrumb a:hover {{
            text-decoration: underline;
        }}
        .ground-truth {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .ground-truth strong {{
            font-size: 16px;
        }}
        .controls {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .control-label {{
            font-weight: 600;
            color: #495057;
            font-size: 14px;
        }}
        select {{
            padding: 10px 15px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 14px;
            background-color: white;
            cursor: pointer;
            min-width: 200px;
            transition: border-color 0.3s;
        }}
        select:hover {{
            border-color: #007bff;
        }}
        select:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
        }}
        .main-layout {{
            display: flex;
            gap: 20px;
        }}
        .image-section {{
            flex: 0 0 512px;
        }}
        .info-section {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .analysis-columns {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            flex: 1;
        }}
        .analysis-column {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #dee2e6;
            display: flex;
            flex-direction: column;
        }}
        .analysis-column h3 {{
            margin: 0 0 15px 0;
            color: #495057;
            font-size: 16px;
            padding-bottom: 10px;
            border-bottom: 2px solid #dee2e6;
            text-align: center;
        }}
        .analysis-column.nn-column h3 {{ color: #007bff; border-color: #007bff; }}
        .analysis-column.logit-column h3 {{ color: #28a745; border-color: #28a745; }}
        .analysis-column.ctx-column h3 {{ color: #dc3545; border-color: #dc3545; }}
        .analysis-results {{
            flex: 1;
            overflow-y: auto;
            max-height: 500px;
        }}
        .image-container {{
            position: relative;
            display: inline-block;
            width: 100%;
        }}
        .base-image {{
            width: 100%;
            max-width: 512px;
            border: 3px solid #34495e;
            border-radius: 8px;
        }}
        .patch-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        .patch {{
            position: absolute;
            border: 1px solid rgba(255,255,255,0.2);
            cursor: pointer;
            pointer-events: all;
            transition: all 0.2s;
            background-color: transparent;
        }}
        .patch.maybe-interpretable {{
            background-color: rgba(144, 238, 144, 0.3);
            border: 1px solid rgba(144, 238, 144, 0.5);
        }}
        .patch.likely-interpretable {{
            background-color: rgba(40, 167, 69, 0.35);
            border: 1px solid rgba(40, 167, 69, 0.6);
        }}
        .patch:hover {{
            border: 2px solid #fff;
            box-shadow: 0 0 10px rgba(255,255,255,0.8);
            z-index: 10;
        }}
        .patch.active {{
            border: 3px solid #ffc107;
            box-shadow: 0 0 15px rgba(255,193,7,0.9);
            z-index: 10;
        }}
        .no-data {{
            text-align: center;
            color: #adb5bd;
            padding: 20px;
            font-style: italic;
            font-size: 13px;
        }}
        .empty-state {{
            text-align: center;
            color: #6c757d;
            padding: 40px 20px;
        }}
        .empty-state-icon {{
            font-size: 48px;
            margin-bottom: 10px;
        }}
        .result-item {{
            margin-bottom: 8px;
            padding: 8px;
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
            font-size: 13px;
        }}
        .nn-column .result-item {{ border-left: 3px solid #007bff; }}
        .logit-column .result-item {{ border-left: 3px solid #28a745; }}
        .ctx-column .result-item {{ border-left: 3px solid #dc3545; }}
        .result-header {{
            font-weight: 600;
            margin-bottom: 4px;
            font-size: 12px;
        }}
        .nn-column .result-header {{ color: #007bff; }}
        .logit-column .result-header {{ color: #28a745; }}
        .ctx-column .result-header {{ color: #dc3545; }}
        .result-content {{
            font-size: 13px;
            color: #495057;
        }}
        .highlight {{
            background-color: #ffeb3b;
            font-weight: bold;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .details-toggle {{
            cursor: pointer;
            user-select: none;
            font-weight: 500;
        }}
        .details-toggle:hover {{
            text-decoration: underline !important;
        }}
        .logit-value {{
            font-weight: bold;
        }}
        .logit-high {{ color: #28a745; }}
        .logit-med {{ color: #17a2b8; }}
        .logit-low {{ color: #ffc107; }}
        .logit-neg {{ color: #dc3545; }}
        .instructions {{
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #0066cc;
            margin-top: 20px;
        }}
        .instructions p {{
            margin: 5px 0;
            color: #004080;
        }}
        .interpretability-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            margin-left: 10px;
        }}
        .interpretability-badge.likely-interpretable {{
            background-color: #28a745;
            color: white;
        }}
        .interpretability-badge.maybe-interpretable {{
            background-color: #90EE90;
            color: #333;
        }}
        .interpretability-badge.not-interpretable {{
            background-color: #6c757d;
            color: white;
        }}
        .interpretability-details {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }}
        .toggle-button {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 0 10px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .toggle-button:hover {{
            background-color: #45a049;
        }}
        .toggle-button.off {{
            background-color: #6c757d;
        }}
        .toggle-button.off:hover {{
            background-color: #5a6268;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="breadcrumb">
            <a href="../index.html">← Model Grid</a>
            <span>|</span>
            <a href="index.html">← {LLM_DISPLAY_NAMES[llm]} + {VISION_ENCODER_DISPLAY_NAMES[ve]}</a>
        </div>
        
        <div class="header">
            <h1>Image {image_idx:04d}</h1>
            <p style="color: #666; margin: 0;">{LLM_DISPLAY_NAMES[llm]} + {VISION_ENCODER_DISPLAY_NAMES[ve]}</p>
        </div>
        
        <div class="ground-truth">
            <strong>Ground Truth Caption:</strong><br>
            <span style="font-size: 15px; margin-top: 8px; display: block;">{escape_for_html(ground_truth)}</span>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label class="control-label" for="layer">Layer:</label>
                <select id="layer">'''
    
    # Add layer options
    for layer in all_layers:
        selected = ' selected' if layer == default_layer else ''
        html += f'<option value="{layer}"{selected}>Layer {layer}</option>'
    
    html += f'''
                </select>
            </div>
            <div style="color: #666; font-size: 13px; padding: 10px;">
                Showing all three analyses side-by-side for the selected layer
            </div>
            <div>
                <button id="toggleGrid" class="toggle-button">Hide Grid</button>
            </div>
        </div>
        
        <div class="main-layout">
            <div class="image-section">
                <div class="image-container">'''
    
    if image_base64:
        html += f'<img src="{image_base64}" alt="Image {image_idx}" class="base-image" id="baseImage">'
    else:
        html += f'<div class="base-image" id="baseImage" style="width: 512px; height: 512px; background-color: #ddd; display: flex; align-items: center; justify-content: center; border-radius: 8px;">Image not available</div>'
    
    html += f'''
                    <div class="patch-overlay" id="patchOverlay"></div>
                </div>
                <div class="instructions">
                    <p><strong>Instructions:</strong> Click on any patch to see all three analyses side-by-side.</p>
                    <p><strong>Grid:</strong> {grid_size}×{grid_size} patches ({patches_per_chunk} total)</p>
                    <p><strong>Light green (1.5-1.8×):</strong> Maybe interpretable | <strong>Dark green (≥1.8×):</strong> Likely interpretable</p>
                    <p><strong>Tip:</strong> Use the "Hide Grid" button to remove overlays. Use layer dropdown to switch layers. Patch selection persists!</p>
                </div>
            </div>
            
            <div class="info-section">
                <div id="patchInfo" style="text-align: center; padding: 10px; background: #e7f3ff; border-radius: 6px; margin-bottom: 10px;">
                    <strong>Click on a patch to see results</strong>
                </div>
                <div class="analysis-columns">
                    <div class="analysis-column nn-column">
                        <h3>🔍 Embedding Matrix</h3>
                        <div class="analysis-results" id="nnResults">
                            <div class="empty-state">Select a patch</div>
                        </div>
                    </div>
                    <div class="analysis-column logit-column">
                        <h3>💭 Logit Lens</h3>
                        <div class="analysis-results" id="logitResults">
                            <div class="empty-state">Select a patch</div>
                        </div>
                    </div>
                    <div class="analysis-column ctx-column">
                        <h3>📝 LN-Lens</h3>
                        <div class="analysis-results" id="contextualResults">
                            <div class="empty-state">Select a patch</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Embed all data
        const allData = {json.dumps(unified_patch_data, indent=2)};
        const interpretabilityData = {json.dumps(interpretability_map, indent=2)};
        const gridSize = {grid_size};
        const gridRows = {grid_rows};
        const gridCols = {grid_cols};
        const availableLayers = {{
            "nn": {json.dumps(nn_layers)},
            "logitlens": {json.dumps(logit_layers)},
            "contextual_cc": {json.dumps(ctx_cc_layers)},
            "contextual_vg": {json.dumps(ctx_vg_layers)}
        }};
        
        let currentLayer = {default_layer};
        let activePatchDiv = null;
        let activePatchIdx = null;
        let patches = {{}};
        
        // Create patch overlays - use any available analysis type to get patch positions
        function createPatches() {{
            const baseImage = document.getElementById('baseImage');
            const overlay = document.getElementById('patchOverlay');
            overlay.innerHTML = '';
            patches = {{}};
            
            const rect = baseImage.getBoundingClientRect();
            const imageWidth = rect.width;
            const imageHeight = rect.height;
            
            // Find available data for current layer (try NN first, then logit, then contextual VG, then CC)
            // Note: JSON keys are strings, so convert currentLayer to string
            const layerKey = String(currentLayer);
            let referenceData = null;
            if (allData.nn[layerKey]) {{
                referenceData = allData.nn[layerKey];
            }} else if (allData.logitlens[layerKey]) {{
                referenceData = allData.logitlens[layerKey];
            }} else if (allData.contextual_vg[layerKey]) {{
                referenceData = allData.contextual_vg[layerKey];
            }} else if (allData.contextual_cc[layerKey]) {{
                referenceData = allData.contextual_cc[layerKey];
            }}
            
            if (!referenceData) {{
                console.log('No data for layer', currentLayer);
                return;
            }}
            
            // Create patches based on reference data
            const patchDivs = [];
            Object.entries(referenceData).forEach(([patchIdx, patchData]) => {{
                const patchDiv = document.createElement('div');
                patchDiv.className = 'patch';
                patchDiv.dataset.patchIdx = patchIdx;
                
                // Add interpretable class based on ratio threshold
                if (interpretabilityData[patchIdx]) {{
                    const ratio = interpretabilityData[patchIdx].ratio;
                    if (ratio >= 1.8) {{
                        patchDiv.classList.add('likely-interpretable');
                    }} else if (ratio >= 1.5) {{
                        patchDiv.classList.add('maybe-interpretable');
                    }}
                }}
                
                // Use gridCols for width, gridRows for height (handles non-square grids like Qwen2-VL)
                const patchWidth = imageWidth / gridCols;
                const patchHeight = imageHeight / gridRows;
                const left = patchData.col * patchWidth;
                const top = patchData.row * patchHeight;
                
                patchDiv.style.left = left + 'px';
                patchDiv.style.top = top + 'px';
                patchDiv.style.width = patchWidth + 'px';
                patchDiv.style.height = patchHeight + 'px';
                
                patchDiv.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    if (activePatchDiv) {{
                        activePatchDiv.classList.remove('active');
                    }}
                    patchDiv.classList.add('active');
                    activePatchDiv = patchDiv;
                    activePatchIdx = patchIdx;
                    updateResults(patchIdx);
                }});
                
                overlay.appendChild(patchDiv);
                patches[patchIdx] = patchDiv;
                patchDivs.push({{idx: patchIdx, div: patchDiv}});
            }});
            
            // Restore previous selection if it exists
            if (activePatchIdx !== null && patches[activePatchIdx]) {{
                const patchDiv = patches[activePatchIdx];
                patchDiv.classList.add('active');
                activePatchDiv = patchDiv;
                updateResults(activePatchIdx);
            }}
        }}
        
        // Update all three results panels simultaneously
        function updateResults(patchIdx) {{
            // Get data for this patch across all analysis types
            // Note: JSON keys are strings, so convert currentLayer to string
            const layerKey = String(currentLayer);
            const nnData = allData.nn[layerKey] && allData.nn[layerKey][patchIdx];
            const logitData = allData.logitlens[layerKey] && allData.logitlens[layerKey][patchIdx];
            const ctxVgData = allData.contextual_vg[layerKey] && allData.contextual_vg[layerKey][patchIdx];
            const ctxCcData = allData.contextual_cc[layerKey] && allData.contextual_cc[layerKey][patchIdx];
            const interpData = interpretabilityData[patchIdx];
            
            // Update patch info at top
            const patchInfo = document.getElementById('patchInfo');
            if (nnData || logitData || ctxVgData || ctxCcData) {{
                const row = (nnData || logitData || ctxVgData || ctxCcData).row;
                const col = (nnData || logitData || ctxVgData || ctxCcData).col;
                
                // Build interpretability badge
                let interpBadgeHTML = '';
                if (interpData) {{
                    const ratio = interpData.ratio;
                    let badgeClass, badgeText;
                    
                    if (ratio >= 1.8) {{
                        badgeClass = 'likely-interpretable';
                        badgeText = '✓✓ Likely Interpretable';
                    }} else if (ratio >= 1.5) {{
                        badgeClass = 'maybe-interpretable';
                        badgeText = '✓ Maybe Interpretable';
                    }} else {{
                        badgeClass = 'not-interpretable';
                        badgeText = '✗ Not Interpretable';
                    }}
                    
                    interpBadgeHTML = `<span class="interpretability-badge ${{badgeClass}}">${{badgeText}}</span>`;
                    
                    // Add ratio details
                    const ratioDetails = `<div class="interpretability-details">Ratio: ${{ratio.toFixed(2)}} (contextual: ${{interpData.contextual_top_similarity.toFixed(3)}} / NN: ${{interpData.nn_top_similarity.toFixed(3)}})</div>`;
                    interpBadgeHTML += ratioDetails;
                }}
                
                patchInfo.innerHTML = `<strong>Patch (${{row}}, ${{col}}) - Index ${{patchIdx}} | Layer ${{currentLayer}}</strong>${{interpBadgeHTML}}`;
            }}
            
            // Update NN column
            const nnResults = document.getElementById('nnResults');
            if (nnData && nnData.neighbors && nnData.neighbors.length > 0) {{
                let html = '';
                nnData.neighbors.forEach(nn => {{
                    html += `<div class="result-item">
                        <div class="result-header">${{nn.rank}}. Sim: ${{nn.similarity.toFixed(3)}}</div>
                        <div class="result-content">"${{nn.token}}"</div>
                    </div>`;
                }});
                nnResults.innerHTML = html;
            }} else {{
                nnResults.innerHTML = '<div class="no-data">No data for layer ${{currentLayer}}</div>';
            }}
            
            // Update Logit Lens column
            const logitResults = document.getElementById('logitResults');
            if (logitData && logitData.predictions && logitData.predictions.length > 0) {{
                let html = '';
                logitData.predictions.forEach(pred => {{
                    const logit = pred.logit;
                    let logitClass = 'logit-neg';
                    if (logit > 10) logitClass = 'logit-high';
                    else if (logit > 5) logitClass = 'logit-med';
                    else if (logit > 0) logitClass = 'logit-low';
                    
                    html += `<div class="result-item">
                        <div class="result-header">${{pred.rank}}. Logit: <span class="logit-value ${{logitClass}}">${{logit.toFixed(2)}}</span></div>
                        <div class="result-content" style="font-family: monospace; background-color: #e9ecef; padding: 4px; border-radius: 3px; margin-top: 3px;">
                            "${{pred.token}}"
                        </div>
                        <div style="font-size: 10px; color: #6c757d; margin-top: 2px;">ID: ${{pred.token_id}}</div>
                    </div>`;
                }});
                logitResults.innerHTML = html;
            }} else {{
                logitResults.innerHTML = '<div class="no-data">No data for layer ${{currentLayer}}</div>';
            }}
            
            // Update LN-Lens column - show VG first, then CC (legacy)
            const ctxResults = document.getElementById('contextualResults');
            let html = '';
            
            // Add Visual Genome results first
            if (ctxVgData && ctxVgData.contextual_neighbors && ctxVgData.contextual_neighbors.length > 0) {{
                html += '<div style="background: #e8f5e9; padding: 8px; margin-bottom: 10px; border-radius: 4px; font-weight: 600; text-align: center; color: #2e7d32;">📸 Visual Genome Phrases</div>';
                ctxVgData.contextual_neighbors.forEach((ctx, idx) => {{
                    const detailsId = `ctx-vg-details-${{patchIdx}}-${{idx}}`;
                    // Show which contextual layer this neighbor came from
                    const layerInfo = ctx.contextual_layer !== null && ctx.contextual_layer !== undefined 
                        ? `<span style="background: #6c757d; color: white; padding: 1px 4px; border-radius: 3px; font-size: 10px; margin-left: 4px;">L${{ctx.contextual_layer}}</span>` 
                        : '';
                    
                    html += `<div class="result-item">
                        <div class="result-header">${{ctx.rank}}. Sim: ${{ctx.similarity.toFixed(3)}}${{layerInfo}}</div>
                        <div class="result-content" style="margin-top: 3px; font-size: 12px;">
                            ${{ctx.caption}}
                            </div>`;
                    
                    // Add expandable details section if additional data is available
                    const hasLowest = ctx.lowest_similarity_same_token !== undefined && ctx.lowest_similarity_same_token !== null;
                    const hasInterNN = ctx.similarity_to_other_nns !== undefined && ctx.similarity_to_other_nns !== null;
                    
                    if (hasLowest || hasInterNN) {{
                        html += `
                        <div style="margin-top: 5px;">
                            <a href="#" class="details-toggle" id="toggle-${{detailsId}}" onclick="event.preventDefault(); const el = document.getElementById('${{detailsId}}'); const toggle = document.getElementById('toggle-${{detailsId}}'); if (el.style.display === 'none') {{ el.style.display = 'block'; toggle.innerHTML = '▾ Hide details'; }} else {{ el.style.display = 'none'; toggle.innerHTML = '▸ Show details'; }}" style="font-size: 11px; color: #dc3545; text-decoration: none;">
                                ▸ Show details
                            </a>
                        </div>
                        <div id="${{detailsId}}" style="display: none; margin-top: 8px; padding: 8px; background: #fff3cd; border-radius: 4px; font-size: 11px; border-left: 2px solid #ffc107;">`;
                        
                        // Show inter-NN similarities (only for 1st neighbor)
                        if (hasInterNN && idx === 0) {{
                            html += `<div style="margin-bottom: 6px;">
                                <strong style="color: #856404;">🔗 Similarity to other top neighbors:</strong><br>
                                <span style="color: #666; font-size: 10px;">(How similar is this 1st NN to the 2nd, 3rd, etc.)</span><br>`;
                            
                            const interSims = Object.entries(ctx.similarity_to_other_nns)
                                .sort((a, b) => a[0].localeCompare(b[0]));
                            
                            interSims.forEach(([key, value]) => {{
                                const nnNum = key.replace('vs_', '');
                                html += `<span style="margin-right: 8px;">vs ${{nnNum}}: <strong>${{value.toFixed(3)}}</strong></span>`;
                            }});
                            html += `</div>`;
                        }}
                        
                        // Show lowest similarity for same token
                        if (hasLowest) {{
                            const lowest = ctx.lowest_similarity_same_token;
                            html += `<div>
                                <strong style="color: #856404;">📉 Same token, different context:</strong><br>
                                <span style="color: #666; font-size: 10px;">(Lowest similarity for same token in a different phrase - shows context matters!)</span><br>
                                <span style="margin-top: 2px; display: block;">Sim: <strong>${{lowest.similarity.toFixed(3)}}</strong> | ${{lowest.num_instances}} total instances</span>
                                <div style="margin-top: 3px; font-style: italic; color: #555;">"${{lowest.caption}}"</div>
                                <span style="font-size: 10px; color: #777;">Position: ${{lowest.position}}</span>
                            </div>`;
                        }}
                        
                        html += `</div>`;
                    }}
                    
                    html += `</div>`;
                }});
            }}
            
            // Add Conceptual Captions results second
            if (ctxCcData && ctxCcData.contextual_neighbors && ctxCcData.contextual_neighbors.length > 0) {{
                if (html) html += '<div style="margin: 15px 0; border-top: 2px solid #dee2e6;"></div>';
                html += '<div style="background: #e3f2fd; padding: 8px; margin-bottom: 10px; border-radius: 4px; font-weight: 600; text-align: center; color: #1565c0;">💬 Conceptual Captions</div>';
                ctxCcData.contextual_neighbors.forEach((ctx, idx) => {{
                    const detailsId = `ctx-cc-details-${{patchIdx}}-${{idx}}`;
                    // Show which contextual layer this neighbor came from
                    const layerInfo = ctx.contextual_layer !== null && ctx.contextual_layer !== undefined 
                        ? `<span style="background: #6c757d; color: white; padding: 1px 4px; border-radius: 3px; font-size: 10px; margin-left: 4px;">L${{ctx.contextual_layer}}</span>` 
                        : '';
                    
                    html += `<div class="result-item">
                        <div class="result-header">${{ctx.rank}}. Sim: ${{ctx.similarity.toFixed(3)}}${{layerInfo}}</div>
                        <div class="result-content" style="margin-top: 3px; font-size: 12px;">
                            ${{ctx.caption}}
                        </div>`;
                    
                    // Add expandable details section if additional data is available
                    const hasLowest = ctx.lowest_similarity_same_token !== undefined && ctx.lowest_similarity_same_token !== null;
                    const hasInterNN = ctx.similarity_to_other_nns !== undefined && ctx.similarity_to_other_nns !== null;
                    
                    if (hasLowest || hasInterNN) {{
                        html += `
                        <div style="margin-top: 5px;">
                            <a href="#" class="details-toggle" id="toggle-${{detailsId}}" onclick="event.preventDefault(); const el = document.getElementById('${{detailsId}}'); const toggle = document.getElementById('toggle-${{detailsId}}'); if (el.style.display === 'none') {{ el.style.display = 'block'; toggle.innerHTML = '▾ Hide details'; }} else {{ el.style.display = 'none'; toggle.innerHTML = '▸ Show details'; }}" style="font-size: 11px; color: #dc3545; text-decoration: none;">
                                ▸ Show details
                            </a>
                        </div>
                        <div id="${{detailsId}}" style="display: none; margin-top: 8px; padding: 8px; background: #fff3cd; border-radius: 4px; font-size: 11px; border-left: 2px solid #ffc107;">`;
                        
                        // Show inter-NN similarities (only for 1st neighbor)
                        if (hasInterNN && idx === 0) {{
                            html += `<div style="margin-bottom: 6px;">
                                <strong style="color: #856404;">🔗 Similarity to other top neighbors:</strong><br>
                                <span style="color: #666; font-size: 10px;">(How similar is this 1st NN to the 2nd, 3rd, etc.)</span><br>`;
                            
                            const interSims = Object.entries(ctx.similarity_to_other_nns)
                                .sort((a, b) => a[0].localeCompare(b[0]));
                            
                            interSims.forEach(([key, value]) => {{
                                const nnNum = key.replace('vs_', '');
                                html += `<span style="margin-right: 8px;">vs ${{nnNum}}: <strong>${{value.toFixed(3)}}</strong></span>`;
                            }});
                            html += `</div>`;
                        }}
                        
                        // Show lowest similarity for same token
                        if (hasLowest) {{
                            const lowest = ctx.lowest_similarity_same_token;
                            html += `<div>
                                <strong style="color: #856404;">📉 Same token, different context:</strong><br>
                                <span style="color: #666; font-size: 10px;">(Lowest similarity for same token in a different caption - shows context matters!)</span><br>
                                <span style="margin-top: 2px; display: block;">Sim: <strong>${{lowest.similarity.toFixed(3)}}</strong> | ${{lowest.num_instances}} total instances</span>
                                <div style="margin-top: 3px; font-style: italic; color: #555;">"${{lowest.caption}}"</div>
                                <span style="font-size: 10px; color: #777;">Position: ${{lowest.position}}</span>
                            </div>`;
                        }}
                        
                        html += `</div>`;
                    }}
                    
                    html += `</div>`;
                }});
            }}
            
            // If no data at all
            if (!html) {{
                html = '<div class="no-data">No data for layer ${{currentLayer}}</div>';
            }}
            
            ctxResults.innerHTML = html;
        }}
        
        // Event listener for layer changes
        document.getElementById('layer').addEventListener('change', (e) => {{
            currentLayer = parseInt(e.target.value);
            createPatches();  // Will restore patch selection if possible
        }});
        
        // Clear active patch when clicking outside (but not on controls)
        document.addEventListener('click', (e) => {{
            // Don't clear if clicking on patch, dropdown, or any control element
            if (!e.target.classList.contains('patch') && 
                e.target.id !== 'layer' && 
                !e.target.closest('.controls') &&
                !e.target.closest('.analysis-columns')) {{
                if (activePatchDiv) {{
                    activePatchDiv.classList.remove('active');
                    activePatchDiv = null;
                    activePatchIdx = null;
                    // Reset all three results panels
                    document.getElementById('patchInfo').innerHTML = '<strong>Click on a patch to see results</strong>';
                    document.getElementById('nnResults').innerHTML = '<div class="empty-state">Select a patch</div>';
                    document.getElementById('logitResults').innerHTML = '<div class="empty-state">Select a patch</div>';
                    document.getElementById('contextualResults').innerHTML = '<div class="empty-state">Select a patch</div>';
                }}
            }}
        }});
        
        // Initialize
        const baseImage = document.getElementById('baseImage');
        if (baseImage.tagName === 'IMG') {{
            baseImage.addEventListener('load', createPatches);
            if (baseImage.complete) {{ createPatches(); }}
        }} else {{
            createPatches();
        }}
        
        window.addEventListener('resize', () => {{
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
    
    return html

def main():
    parser = argparse.ArgumentParser(description="Create unified interactive viewer for all analysis results")
    parser.add_argument("--output-dir", type=str, default="analysis_results/unified_viewer",
                       help="Output directory for the unified viewer")
    parser.add_argument("--num-images", type=int, default=300,
                       help="Number of images to process (default: 300)")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Which split to visualize (default: validation)")
    parser.add_argument("--lite-suffix", type=str, default="",
                       help="Suffix for lite JSON directories (e.g., '_lite10'). Use this for fast iteration!")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log.info("Scanning analysis results...")
    
    # Load dataset once
    log.info(f"Loading PixMoCap dataset ({args.split} split)...")
    try:
        dataset = PixMoCap(split=args.split, mode="captions")
        log.info("✓ Dataset loaded successfully")
    except Exception as e:
        log.error(f"Could not load dataset: {e}")
        return
    
    # Scan all model combinations
    model_availability = {}
    for llm in LLMS:
        for ve in VISION_ENCODERS:
            checkpoint_name = get_checkpoint_name(llm, ve)
            log.info(f"\nScanning {checkpoint_name}...")
            
            analysis_results = scan_analysis_results(checkpoint_name, args.lite_suffix)
            
            has_results = (len(analysis_results["nn"]) > 0 or 
                          len(analysis_results["logitlens"]) > 0 or 
                          len(analysis_results["contextual_cc"]) > 0 or
                          len(analysis_results["contextual_vg"]) > 0)
            
            model_availability[(llm, ve)] = {
                "available": has_results,
                "checkpoint_name": checkpoint_name,
                "nn_layers": sorted(analysis_results["nn"].keys()),
                "logit_layers": sorted(analysis_results["logitlens"].keys()),
                "contextual_cc_layers": sorted(analysis_results["contextual_cc"].keys()),
                "contextual_vg_layers": sorted(analysis_results["contextual_vg"].keys()),
                "analysis_results": analysis_results
            }
            
            if has_results:
                log.info(f"  Found: NN={len(analysis_results['nn'])}, " 
                        f"Logit={len(analysis_results['logitlens'])}, "
                        f"Contextual_CC={len(analysis_results['contextual_cc'])}, "
                        f"Contextual_VG={len(analysis_results['contextual_vg'])}")
                
                # Create model index page
                create_model_index(output_dir, checkpoint_name, llm, ve, 
                                 args.num_images, analysis_results)
                
                # Create model-specific preprocessor
                preprocessor = create_preprocessor(checkpoint_name)
                
                # Load all analysis data at once (MUCH faster than loading per image!)
                all_data_cache = load_all_analysis_data(analysis_results, args.split, args.num_images)
                
                # Create unified image viewers
                log.info(f"  Creating unified image viewers (now fast since data is cached!)...")
                success_count = 0
                t_start = time.time()
                for img_idx in range(args.num_images):
                    if img_idx % 50 == 0 and img_idx > 0:
                        elapsed = time.time() - t_start
                        rate = img_idx / elapsed
                        remaining = (args.num_images - img_idx) / rate if rate > 0 else 0
                        log.info(f"    Progress: {img_idx}/{args.num_images} ({img_idx/args.num_images*100:.1f}%) "
                                f"- {rate:.1f} images/sec - ETA: {remaining:.0f}s")
                    
                    try:
                        if create_unified_image_viewer(output_dir, checkpoint_name, llm, ve,
                                                      img_idx, all_data_cache, dataset, args.split,
                                                      preprocessor):
                            success_count += 1
                    except Exception as e:
                        if img_idx % 50 == 0:
                            log.warning(f"    Error creating viewer for image {img_idx}: {e}")
                
                total_time = time.time() - t_start
                log.info(f"  ✓ Created {success_count}/{args.num_images} image viewers in {total_time:.1f}s "
                        f"({success_count/total_time:.1f} images/sec)")
    
    # Create main index
    log.info("\nCreating main index...")
    create_main_index(output_dir, model_availability)
    
    log.info(f"\n✅ Unified viewer created successfully!")
    log.info(f"📂 Output directory: {output_dir}")
    log.info(f"🌐 Open {output_dir / 'index.html'} in a web browser to start exploring!")

if __name__ == "__main__":
    main()

