#!/usr/bin/env python3
"""
Shared preprocessing for Molmo-72B analysis and LLM judge.

CRITICAL: This module defines the SINGLE SOURCE OF TRUTH for Molmo-72B image preprocessing.
ALL scripts (NN, LogitLens, Contextual, LLM Judge, Viewer) MUST import from here.

Molmo-72B uses a multi-crop approach:
- Base crop (crop 0): 12x12 = 144 tokens covering the FULL image at base resolution (336x336)
- High-res crops (crops 1+): overlapping tiles at higher resolution
- All crops are processed by the ViT and their tokens enter the LLM sequence

For analysis, we use the BASE CROP (crop 0) tokens as our spatial grid (12x12 = 144 tokens).
The full model sees all crops (no information loss), but we analyze only the base crop
for clean spatial correspondence.
"""

from PIL import Image
from typing import Tuple


# Molmo-72B constants
MODEL_NAME = "allenai/Molmo-72B-0924"
BASE_IMAGE_SIZE = 336  # Base crop resolution
TOKENS_PER_CROP_H = 12  # Token grid height per crop
TOKENS_PER_CROP_W = 12  # Token grid width per crop
TOKENS_PER_CROP = TOKENS_PER_CROP_H * TOKENS_PER_CROP_W  # 144
VIT_PATCHES_PER_CROP = 576  # 24x24 ViT patches before pooling
BASE_CROP_INDEX = 0  # Base crop is always the first one


def preprocess_image_molmo(
    image: Image.Image,
    target_size: int = 512,
) -> Image.Image:
    """
    Preprocess image for LLM judge / visualization — matches Molmo's actual preprocessing.

    Molmo uses resize_and_pad: resize preserving aspect ratio to fit target_size,
    then center-pad with black. This is what the model actually sees, including
    black padding bars for non-square images.

    Args:
        image: PIL Image
        target_size: Target display size (512 for LLM judge, 336 for model input)

    Returns:
        Preprocessed PIL Image (target_size x target_size, with black padding)
    """
    w, h = image.size

    # Scale to fit inside target_size preserving aspect ratio (matches Molmo's resize_and_pad)
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center-pad with black to target_size x target_size
    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    padded.paste(image, (left, top))

    return padded


def get_base_crop_grid() -> Tuple[int, int, int]:
    """
    Get the spatial grid dimensions for the base crop.

    Returns:
        (grid_h, grid_w, num_tokens) = (12, 12, 144)
    """
    return TOKENS_PER_CROP_H, TOKENS_PER_CROP_W, TOKENS_PER_CROP


def get_base_crop_token_positions(image_input_idx):
    """
    Extract the LLM sequence positions for the base crop tokens.

    Args:
        image_input_idx: tensor of shape (num_crops, tokens_per_crop)
            from Molmo's processor output

    Returns:
        list of (seq_pos, row, col) tuples for each base crop token
        Only includes valid positions (>= 0).
    """
    base_crop_idx = image_input_idx[BASE_CROP_INDEX]  # (144,)
    positions = []
    for token_i in range(TOKENS_PER_CROP):
        seq_pos = base_crop_idx[token_i].item()
        if seq_pos >= 0:
            row = token_i // TOKENS_PER_CROP_W
            col = token_i % TOKENS_PER_CROP_W
            positions.append((seq_pos, row, col))
    return positions


def get_all_vision_token_positions(image_input_idx):
    """
    Extract ALL vision token positions (all crops) from image_input_idx.

    Args:
        image_input_idx: tensor of shape (num_crops, tokens_per_crop)

    Returns:
        sorted list of sequence positions for ALL valid vision tokens
    """
    import torch
    valid_mask = image_input_idx >= 0
    return sorted(image_input_idx[valid_mask].tolist())


def validate_base_crop(image_input_idx) -> int:
    """
    Validate that the base crop has the expected number of tokens.

    Args:
        image_input_idx: tensor from Molmo processor

    Returns:
        Number of valid base crop tokens

    Raises:
        ValueError if base crop is missing or has unexpected token count
    """
    if image_input_idx.shape[0] < 1:
        raise ValueError("No crops found in image_input_idx")

    base_crop_idx = image_input_idx[BASE_CROP_INDEX]
    num_valid = (base_crop_idx >= 0).sum().item()

    if num_valid != TOKENS_PER_CROP:
        raise ValueError(
            f"Base crop has {num_valid} valid tokens, expected {TOKENS_PER_CROP} "
            f"(12x12 grid). This may indicate an image processing issue."
        )

    return num_valid


def validate_preprocessing(image: Image.Image, expected_size: int = 512) -> None:
    """Validate that display image has been preprocessed correctly."""
    if image.size != (expected_size, expected_size):
        raise ValueError(
            f"Image preprocessing failed: expected {expected_size}x{expected_size}, "
            f"got {image.size[0]}x{image.size[1]}"
        )
