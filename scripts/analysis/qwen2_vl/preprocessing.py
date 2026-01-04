#!/usr/bin/env python3
"""
Shared preprocessing for Qwen2-VL analysis and LLM judge.

CRITICAL: This module defines the SINGLE SOURCE OF TRUTH for Qwen2-VL image preprocessing.
ALL scripts (NN, LogitLens, Contextual, LLM Judge, Viewer) MUST import from here.

Preprocessing strategy:
- Center-crop to square (cuts edges, keeps center)
- Resize to fixed resolution (448x448 for analysis, 512x512 for LLM judge)
- This matches Qwen2-VL's training preprocessing
"""

from PIL import Image
from typing import Tuple


def preprocess_image_qwen2vl(
    image: Image.Image,
    target_size: int = 448,
    force_square: bool = True
) -> Image.Image:
    """
    Preprocess image for Qwen2-VL using center-crop + resize.

    This is the CANONICAL preprocessing for Qwen2-VL across all scripts.

    Args:
        image: PIL Image to preprocess
        target_size: Target size in pixels (default 448 for analysis, use 512 for LLM judge)
        force_square: If True, center-crop to square before resizing

    Returns:
        Preprocessed PIL Image (target_size x target_size, square, center-cropped)

    Example:
        >>> from PIL import Image
        >>> img = Image.open("example.jpg")  # 640x480
        >>> processed = preprocess_image_qwen2vl(img, target_size=448)
        >>> processed.size
        (448, 448)
    """
    if force_square:
        w, h = image.size
        if w != h:
            # Center-crop to square (cuts edges to make square)
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))

    # Resize to exact target size
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.LANCZOS)

    return image


def get_expected_tokens(target_size: int = 448) -> int:
    """
    Calculate expected number of vision tokens for Qwen2-VL.

    Qwen2-VL: 14x14 patches with 2x2 spatial merger = 28 pixels per token

    Args:
        target_size: Image size in pixels

    Returns:
        Expected number of vision tokens

    Example:
        >>> get_expected_tokens(448)
        256  # 16x16 grid
    """
    # Qwen2-VL: 14x14 patches with 2x2 merger = 28 pixels per token
    tokens_per_side = target_size // 28
    return tokens_per_side ** 2


def validate_preprocessing(image: Image.Image, expected_size: int = 448) -> None:
    """
    Validate that image has been preprocessed correctly.

    Raises ValueError if preprocessing is incorrect.

    Args:
        image: PIL Image to validate
        expected_size: Expected size in pixels

    Raises:
        ValueError: If image size is incorrect
    """
    if image.size != (expected_size, expected_size):
        raise ValueError(
            f"Image preprocessing failed: expected {expected_size}x{expected_size}, "
            f"got {image.size[0]}x{image.size[1]}"
        )
