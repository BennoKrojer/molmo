#!/usr/bin/env python3
"""
Shared library for viewer generation.

This module contains common functions used by both create_unified_viewer.py
and generate_ablation_viewers.py to ensure consistency and avoid duplication.
"""

import base64
import html
import io
import math
import numpy as np
from PIL import Image
from typing import Tuple
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
    # Handle checkpoint path - don't double up step12000-unsharded
    if checkpoint_name.endswith("step12000-unsharded"):
        checkpoint_path = f"molmo_data/checkpoints/{checkpoint_name}"
    else:
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
