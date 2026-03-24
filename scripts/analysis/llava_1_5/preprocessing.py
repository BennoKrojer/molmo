#!/usr/bin/env python3
"""
Shared preprocessing for LLaVA-1.5-7B analysis and LLM judge.

CRITICAL: This module defines the SINGLE SOURCE OF TRUTH for LLaVA-1.5 image preprocessing.
ALL scripts (NN, LogitLens, Contextual, LLM Judge, Viewer) MUST import from here.

LLaVA-1.5 uses CLIP ViT-L/14-336:
- Input: 336x336 pixels
- Patch size: 14x14
- Grid: 24x24 = 576 vision tokens (CLS token is excluded before projection)
- Vision tokens replace the <image> placeholder (token_id=32000) in the LLM sequence
"""

from PIL import Image
from typing import Tuple


# LLaVA-1.5-7B constants
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
IMAGE_SIZE = 336  # CLIP ViT-L/14-336
PATCH_SIZE = 14
GRID_H = IMAGE_SIZE // PATCH_SIZE  # 24
GRID_W = IMAGE_SIZE // PATCH_SIZE  # 24
NUM_VISION_TOKENS = GRID_H * GRID_W  # 576
IMAGE_TOKEN_ID = 32000  # <image> token in LLaVA's vocabulary


def preprocess_image_llava(
    image: Image.Image,
    target_size: int = 336,
    force_square: bool = True
) -> Image.Image:
    """
    Preprocess image for LLaVA-1.5 analysis.

    LLaVA-1.5 uses CLIP preprocessing (center-crop + resize to 336x336).
    For analysis consistency, we force-square and resize.

    Args:
        image: PIL Image
        target_size: Target size (336 for model input, 512 for LLM judge display)
        force_square: If True, center-crop to square before resizing

    Returns:
        Preprocessed PIL Image
    """
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
    """
    Get the spatial grid dimensions for LLaVA-1.5 vision tokens.

    Returns:
        (grid_h, grid_w, num_tokens) = (24, 24, 576)
    """
    return GRID_H, GRID_W, NUM_VISION_TOKENS


def get_vision_token_positions(input_ids):
    """
    Find the positions of vision tokens in the LLM sequence.

    In LLaVA-1.5, the <image> token (id=32000) is replaced by 576 vision
    feature tokens during the forward pass. We identify the position of
    <image> in input_ids — the 576 tokens starting there are vision tokens.

    Args:
        input_ids: tensor of shape (seq_len,) — single example

    Returns:
        list of (seq_pos, row, col) tuples for each vision token

    Raises:
        ValueError if <image> token not found
    """
    import torch

    if isinstance(input_ids, torch.Tensor):
        input_ids_list = input_ids.tolist()
    else:
        input_ids_list = list(input_ids)

    # Find position of <image> token
    image_positions = [i for i, tid in enumerate(input_ids_list) if tid == IMAGE_TOKEN_ID]

    if not image_positions:
        raise ValueError(
            f"<image> token (id={IMAGE_TOKEN_ID}) not found in input_ids. "
            f"Make sure the prompt contains an image placeholder."
        )

    # In LLaVA-1.5, there's exactly one <image> token that gets expanded to 576 tokens
    # After expansion, vision tokens occupy positions [image_pos, image_pos + 576)
    start_pos = image_positions[0]

    positions = []
    for token_i in range(NUM_VISION_TOKENS):
        seq_pos = start_pos + token_i
        row = token_i // GRID_W
        col = token_i % GRID_W
        positions.append((seq_pos, row, col))

    return positions


def validate_vision_tokens(num_vision_tokens: int) -> None:
    """
    Validate that the expected number of vision tokens was found.

    Args:
        num_vision_tokens: Actual number of vision tokens

    Raises:
        ValueError if count doesn't match expected
    """
    if num_vision_tokens != NUM_VISION_TOKENS:
        raise ValueError(
            f"Expected {NUM_VISION_TOKENS} vision tokens (24x24 CLIP grid), "
            f"got {num_vision_tokens}. Check image preprocessing."
        )


def validate_preprocessing(image: Image.Image, expected_size: int = 336) -> None:
    """Validate that image has been preprocessed correctly."""
    if image.size != (expected_size, expected_size):
        raise ValueError(
            f"Image preprocessing failed: expected {expected_size}x{expected_size}, "
            f"got {image.size[0]}x{image.size[1]}"
        )
