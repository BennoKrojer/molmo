#!/usr/bin/env python3
"""
Shared preprocessing for LLaVA-NeXT-34B analysis and LLM judge.

CRITICAL: This module defines the SINGLE SOURCE OF TRUTH for LLaVA-NeXT image preprocessing.
ALL scripts (NN, LogitLens, Contextual, LLM Judge, Viewer) MUST import from here.

LLaVA-NeXT uses CLIP ViT-L/14-336 + Yi-34B backbone:
- Input: 336x336 pixels (we force single-tile mode via image_grid_pinpoints=[[336,336]])
- Patch size: 14x14
- Grid: 24x24 = 576 vision tokens
- Vision tokens replace the <image> placeholder (token_id=64000) in the LLM sequence

NOTE: LLaVA-NeXT by default uses AnyRes (multi-tile). We force single-tile to get
a clean 24x24 patch grid for interpretability analysis. This is accomplished by
setting processor.image_processor.image_grid_pinpoints = [[336, 336]].
"""

from PIL import Image
from typing import Tuple


# LLaVA-NeXT-34B constants
MODEL_NAME = "llava-hf/llava-v1.6-34b-hf"
IMAGE_SIZE = 336  # CLIP ViT-L/14-336
PATCH_SIZE = 14
GRID_H = IMAGE_SIZE // PATCH_SIZE  # 24
GRID_W = IMAGE_SIZE // PATCH_SIZE  # 24
NUM_VISION_TOKENS = GRID_H * GRID_W  # 576
IMAGE_TOKEN_ID = 64000  # <image> token in LLaVA-NeXT-34B (Yi) vocabulary


def force_single_tile_mode(processor) -> None:
    """Override processor's image_grid_pinpoints to force single 336x336 tile.

    This disables AnyRes tiling so each image produces exactly 576 vision tokens
    in a clean 24x24 grid, matching LLaVA-1.5 behavior.
    """
    processor.image_processor.image_grid_pinpoints = [[336, 336]]


def preprocess_image_llava(
    image: Image.Image,
    target_size: int = 336,
    force_square: bool = True
) -> Image.Image:
    """Preprocess image for LLaVA-NeXT analysis (CLIP center-crop + resize)."""
    if force_square:
        w, h = image.size
        if w != h:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))

    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.LANCZOS)

    return image


def get_grid_dimensions() -> Tuple[int, int, int]:
    """Returns (grid_h, grid_w, num_tokens) = (24, 24, 576)."""
    return GRID_H, GRID_W, NUM_VISION_TOKENS


def validate_vision_tokens(num_vision_tokens: int) -> None:
    """Validate that the expected number of vision tokens was found."""
    if num_vision_tokens != NUM_VISION_TOKENS:
        raise ValueError(
            f"Expected {NUM_VISION_TOKENS} vision tokens (24x24 CLIP grid), "
            f"got {num_vision_tokens}. Check single-tile mode is enabled."
        )
