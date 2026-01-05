#!/usr/bin/env python3
"""
Shared library for viewer generation.

This module contains common functions used by both create_unified_viewer.py
and generate_ablation_viewers.py to ensure consistency and avoid duplication.

Key functions:
- File discovery: find_analysis_files()
- Data loading: load_analysis_data_for_type()
- Image processing: pil_image_to_base64(), create_preprocessor()
- Patch processing: patch_idx_to_row_col(), extract_patches_from_data()
"""

import base64
import html
import io
import json
import math
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def escape_for_html(text: str) -> str:
    """Properly escape text for HTML."""
    if not text:
        return ""
    return html.escape(text, quote=True)


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


# Alias for consistency
calculate_patch_position = patch_idx_to_row_col


def create_preprocessor(checkpoint_name: str):
    """Create preprocessor for a given checkpoint.

    Args:
        checkpoint_name: Either just the model name (e.g., "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336")
                         or the full checkpoint path including step (e.g., "...seed10_step12000-unsharded")

    Raises:
        RuntimeError: If preprocessor cannot be created (config.yaml not found or other error)
    """
    from olmo.config import ModelConfig
    from olmo.data import build_mm_preprocessor
    from olmo.model import Molmo
    from olmo.util import resource_path

    # Ablations have _step12000-unsharded suffix in viewer_models.json but not in directory names
    # Ablations are in molmo_data/checkpoints/ablations/{name}/ (without suffix)
    if "_step12000-unsharded" in checkpoint_name:
        # Strip the suffix to get base name
        base_name = checkpoint_name.replace("_step12000-unsharded", "")
        # Try ablations path first
        ablation_path = f"molmo_data/checkpoints/ablations/{base_name}"
        if Path(ablation_path).exists():
            checkpoint_path = ablation_path
            log.info(f"    Creating preprocessor from ablation checkpoint: {checkpoint_path}")
        else:
            # Not in ablations, use full name
            checkpoint_path = f"molmo_data/checkpoints/{checkpoint_name}"
            log.info(f"    Creating preprocessor from checkpoint: {checkpoint_path}")
    elif checkpoint_name.endswith("step12000-unsharded"):
        checkpoint_path = f"molmo_data/checkpoints/{checkpoint_name}"
        log.info(f"    Creating preprocessor from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = f"molmo_data/checkpoints/{checkpoint_name}/step12000-unsharded"
        log.info(f"    Creating preprocessor from checkpoint: {checkpoint_path}")

    if "hf:" in checkpoint_path:
        model = Molmo.from_checkpoint(checkpoint_path, device="cpu")
        model_config = model.config
    else:
        config_file = resource_path(checkpoint_path, "config.yaml")
        if not Path(config_file).exists():
            error_msg = f"config.yaml not found at {config_file} - cannot create preprocessor for {checkpoint_name}"
            log.error(f"    ❌ {error_msg}")
            raise RuntimeError(error_msg)

        model_config = ModelConfig.load(config_file, key="model", validate_paths=False)

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


# =============================================================================
# SHARED DATA LOADING FUNCTIONS
# =============================================================================

def find_analysis_files(base_dir: Path, analysis_type: str, data_path: str) -> Dict[int, Path]:
    """
    Find analysis JSON files for a given type and path.

    Supports both main model paths and ablation paths.
    Returns dict mapping layer number -> json file path.

    Args:
        base_dir: Base analysis_results directory
        analysis_type: One of "nn", "logitlens", "contextual"
        data_path: Subdirectory path (checkpoint name with optional suffix)

    Returns:
        Dict mapping layer number to JSON file path
    """
    results = {}

    if analysis_type == "nn":
        search_dir = base_dir / "nearest_neighbors" / data_path
        # Two possible file patterns
        patterns = [
            "nearest_neighbors_analysis_*_layer*.json",  # main models
            "nearest_neighbors_layer*_topk*.json"        # Qwen2-VL style
        ]
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
                    # Parse _visualN_ pattern - visual layer is the key
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


def load_analysis_data_for_type(
    json_files: Dict[int, Path],
    analysis_type: str,
    num_images: int,
    split: str = "validation"
) -> Dict[int, List[Dict]]:
    """
    Load analysis data from JSON files.

    Handles both Format A (chunks/patches) and Format B (patches directly).

    Args:
        json_files: Dict mapping layer -> json file path
        analysis_type: One of "nn", "logitlens", "contextual"
        num_images: Max number of images to load
        split: Dataset split (for NN format A)

    Returns:
        Dict mapping layer -> list of image data dicts
    """
    results = {}

    for layer, json_path in json_files.items():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Different JSON structures for different types
            if analysis_type == "nn":
                # NN can be in splits/validation/images (Format A) or results (Format B)
                images = data.get("splits", {}).get(split, {}).get("images", [])
                if not images:
                    images = data.get("results", [])
            else:
                # LogitLens and Contextual use results directly
                images = data.get("results", [])

            if images:
                results[layer] = images[:num_images]

        except Exception as e:
            log.warning(f"Could not load {analysis_type} data for layer {layer}: {e}")

    return results


def extract_patches_from_data(image_data: Dict) -> List[Dict]:
    """
    Extract patches from image data, handling both Format A and Format B.

    Format A: {chunks: [{patches: [...]}]}
    Format B: {patches: [...]}

    Returns flat list of patch dicts.
    """
    all_patches = []

    if "chunks" in image_data:
        # Format A: chunks/patches structure
        for chunk in image_data.get("chunks", []):
            all_patches.extend(chunk.get("patches", []))
    elif "patches" in image_data:
        # Format B: patches directly
        all_patches = image_data.get("patches", [])

    return all_patches


def get_grid_dimensions(image_data: Dict, default_grid_size: int = 16) -> Tuple[int, int, int]:
    """
    Determine grid dimensions from image data.

    Args:
        image_data: Single image data dict
        default_grid_size: Default grid size (16x16 for most models)

    Returns:
        Tuple of (grid_rows, grid_cols, patches_per_chunk)
    """
    patches = extract_patches_from_data(image_data)

    if not patches:
        return default_grid_size, default_grid_size, default_grid_size * default_grid_size

    patches_per_chunk = len(patches)

    # Check if patches have row/col info (for non-square grids like Qwen2-VL)
    max_row = 0
    max_col = 0
    for patch in patches:
        max_row = max(max_row, patch.get("patch_row", 0))
        max_col = max(max_col, patch.get("patch_col", 0))

    if max_row > 0 or max_col > 0:
        # Use actual dimensions from patch positions (0-indexed, so add 1)
        grid_rows = max_row + 1
        grid_cols = max_col + 1
    else:
        # Assume square grid
        grid_size = int(math.sqrt(patches_per_chunk))
        grid_rows = grid_size
        grid_cols = grid_size

    return grid_rows, grid_cols, patches_per_chunk


def process_nn_patch(patch: Dict, grid_size: int) -> Dict:
    """
    Process a single NN patch into unified format.

    Handles both Format A (nearest_neighbors key) and Format B (top_neighbors key).
    """
    patch_idx = patch.get("patch_idx", -1)
    row = patch.get("patch_row", patch_idx // grid_size)
    col = patch.get("patch_col", patch_idx % grid_size)

    # Handle both key names
    neighbors = patch.get("nearest_neighbors", []) or patch.get("top_neighbors", [])

    nn_list = []
    for i, nn in enumerate(neighbors[:5]):
        nn_list.append({
            "rank": i + 1,
            "token": escape_for_html(nn.get("token", "")),
            "similarity": nn.get("similarity", 0.0)
        })

    return {
        "patch_idx": patch_idx,
        "row": row,
        "col": col,
        "neighbors": nn_list
    }


def process_logit_patch(patch: Dict, grid_size: int) -> Dict:
    """
    Process a single LogitLens patch into unified format.
    """
    patch_idx = patch.get("patch_idx", -1)
    row = patch.get("patch_row", patch_idx // grid_size)
    col = patch.get("patch_col", patch_idx % grid_size)

    pred_list = []
    for i, pred in enumerate(patch.get("top_predictions", [])[:5]):
        pred_list.append({
            "rank": i + 1,
            "token": escape_for_html(pred.get("token", "")),
            "logit": pred.get("logit", 0.0),
            "token_id": pred.get("token_id", 0)
        })

    return {
        "patch_idx": patch_idx,
        "row": row,
        "col": col,
        "predictions": pred_list
    }


def process_contextual_patch(patch: Dict, grid_size: int) -> Dict:
    """
    Process a single contextual (LN-Lens) patch into unified format.
    """
    patch_idx = patch.get("patch_idx", -1)
    row = patch.get("patch_row", patch_idx // grid_size)
    col = patch.get("patch_col", patch_idx % grid_size)

    ctx_list = []
    for i, neighbor in enumerate(patch.get("nearest_contextual_neighbors", [])[:5]):
        token_str = escape_for_html(neighbor.get("token_str", ""))
        caption = escape_for_html(neighbor.get("caption", ""))

        # Highlight token in caption
        highlighted_caption = caption.replace(
            token_str,
            f'<span class="highlight">{token_str}</span>',
            1
        )

        ctx_entry = {
            "rank": i + 1,
            "token": token_str,
            "caption": highlighted_caption,
            "similarity": neighbor.get("similarity", 0.0),
            "contextual_layer": neighbor.get("contextual_layer", None),
            "position": neighbor.get("position", 0)
        }

        # Add optional lowest similarity data if present
        lowest_sim = neighbor.get("lowest_similarity_same_token")
        if lowest_sim:
            ctx_entry["lowest_similarity_same_token"] = {
                "token_str": escape_for_html(lowest_sim.get("token_str", "")),
                "caption": escape_for_html(lowest_sim.get("caption", "")),
                "position": lowest_sim.get("position", 0),
                "similarity": lowest_sim.get("similarity", 0.0),
                "num_instances": lowest_sim.get("num_instances", 0)
            }

        # Add inter-NN similarities if present
        inter_nn_sims = neighbor.get("similarity_to_other_nns")
        if inter_nn_sims:
            ctx_entry["similarity_to_other_nns"] = inter_nn_sims

        ctx_list.append(ctx_entry)

    return {
        "patch_idx": patch_idx,
        "row": row,
        "col": col,
        "contextual_neighbors": ctx_list
    }
