#!/usr/bin/env python3
"""
Create inline figure showing text-in-image interpretation by LatentLens.

Shows consecutive vision patches from a phone screenshot displaying
"The Couch Tomato Café" with LatentLens predictions below each patch.

Uses medium resolution (24x24 per patch) - slightly higher than model's 14x14
for readability while still showing realistic pixelation.

Output: paper/figures/fig_text_patches_inline.pdf (CLIP)
        paper/figures/fig_text_patches_inline_dino.pdf (DINOv2)

Usage:
    python scripts/analysis/create_text_patches_figure.py           # CLIP (default)
    python scripts/analysis/create_text_patches_figure.py --dino    # DINOv2
"""
import sys
sys.path.insert(0, '.')

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from pathlib import Path
import torch
import torchvision
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import InterpolationMode

from olmo.data.pixmo_datasets import PixMoCap


def resize_and_pad_image(image, target_size=336):
    """Same preprocessing as model uses."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width = image.shape[:2]
    scale = min(target_size / height, target_size / width)
    scaled_height = int(height * scale)
    scaled_width = int(width * scale)

    img_tensor = torch.permute(torch.from_numpy(image), [2, 0, 1])
    img_tensor = convert_image_dtype(img_tensor)
    img_tensor = torchvision.transforms.Resize(
        [scaled_height, scaled_width], InterpolationMode.BILINEAR, antialias=True
    )(img_tensor)
    img_tensor = torch.clip(img_tensor, 0.0, 1.0)
    resized = torch.permute(img_tensor, [1, 2, 0]).numpy()

    top_pad = (target_size - scaled_height) // 2
    left_pad = (target_size - scaled_width) // 2

    padded = np.pad(
        resized,
        [[top_pad, target_size - scaled_height - top_pad],
         [left_pad, target_size - scaled_width - left_pad],
         [0, 0]],
        constant_values=0
    )

    return (padded * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dino', action='store_true', help='Generate DINOv2 version')
    args = parser.parse_args()

    # Configuration - use 336 grid for coordinates (what model sees)
    IMAGE_IDX = 2  # Phone screenshot with "The Couch Tomato Café"
    ROW = 10
    MODEL_SIZE = 336
    PATCH_SIZE_MODEL = MODEL_SIZE // 24  # 14 pixels in model space

    # LatentLens predictions
    if args.dino:
        # OLMo+DINOv2, Layer 16 - generic interpretations (cols 6-11)
        # Note: col 5 is in padding area, so we use 6-11 instead
        COLS = list(range(6, 12))  # Columns 6-11
        tokens = ['describing', 'messages', 'letter', 'screenshot', 'letter', 'captions']
        output_suffix = '_dino'
        model_name = 'OLMo+DINOv2'
    else:
        # OLMo+CLIP-ViT, Layer 16 - text-specific interpretations (cols 7-12)
        COLS = list(range(7, 13))  # Columns 7-12
        tokens = ['.the', 'couch', 'acon', 'tomato', 'iro', 'cafe']
        output_suffix = ''
        model_name = 'OLMo+CLIP-ViT'

    print(f"Generating figure for {model_name}")

    # Load original image
    print("Loading PixMoCap dataset...")
    dataset = PixMoCap(split='validation', mode='captions')
    example = dataset.get(IMAGE_IDX, np.random)
    img_path = example['image']
    original = Image.open(img_path).convert("RGB")
    print(f"Original size: {original.size}")

    # Apply EXACT preprocessing model uses (resize + pad to 336x336)
    preprocessed = resize_and_pad_image(original, MODEL_SIZE)
    print(f"Preprocessed size: {preprocessed.shape}")

    # Get patch boundaries in 336 space (directly from preprocessed image)
    y1 = ROW * PATCH_SIZE_MODEL
    y2 = y1 + PATCH_SIZE_MODEL
    x1 = COLS[0] * PATCH_SIZE_MODEL
    x2 = (COLS[-1] + 1) * PATCH_SIZE_MODEL

    print(f"Patch coords in 336 space: ({x1}, {y1}) to ({x2}, {y2})")

    # Extract strip directly from preprocessed image (what model sees)
    strip_orig = preprocessed[y1:y2, x1:x2]
    print(f"Extracted strip: {strip_orig.shape}")

    # Resize to medium resolution (24x24 per patch)
    # Higher than model's 14x14 for readability, but still pixelated
    DISPLAY_PATCH_SIZE = 24
    num_patches = len(COLS)
    target_h = DISPLAY_PATCH_SIZE
    target_w = DISPLAY_PATCH_SIZE * num_patches

    strip_pil = Image.fromarray(strip_orig)
    strip_resized = strip_pil.resize((target_w, target_h), Image.BILINEAR)
    strip_display = np.array(strip_resized)

    print(f"Display strip: {strip_display.shape} ({DISPLAY_PATCH_SIZE}x{target_h} per patch)")

    # Upscale 2x with nearest-neighbor for pixel visibility
    scale_factor = 2
    strip_upscaled = np.repeat(np.repeat(strip_display, scale_factor, axis=0), scale_factor, axis=1)
    print(f"Final strip: {strip_upscaled.shape}")

    # Create figure
    fig_width = num_patches
    fig_height = 2.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    strip_bottom = 1.1
    strip_height = 1.0
    ax.imshow(strip_upscaled, extent=[0, num_patches, strip_bottom, strip_bottom + strip_height],
              aspect='equal', interpolation='nearest')

    # Draw vertical lines between patches
    for i in range(1, num_patches):
        ax.plot([i, i], [strip_bottom, strip_bottom + strip_height],
                color='black', linewidth=1.5)

    # Draw outer border
    rect = mpatches.Rectangle((0, strip_bottom), num_patches, strip_height,
                               linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Draw arrows and yellow boxes
    arrow_top = strip_bottom - 0.05
    arrow_bottom = 0.55
    box_y = 0.2

    for i, token in enumerate(tokens):
        x_center = i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom), xytext=(x_center, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
        ax.text(x_center, box_y, token, ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                         edgecolor='none', alpha=0.8))

    ax.set_xlim(-0.1, num_patches + 0.1)
    ax.set_ylim(0, strip_bottom + strip_height + 0.1)
    ax.axis('off')

    plt.tight_layout()

    # Save to paper/figures/
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / f'fig_text_patches_inline{output_suffix}.pdf'
    output_png = output_dir / f'fig_text_patches_inline{output_suffix}.png'

    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)

    print(f"\nSaved to:")
    print(f"  {output_pdf.absolute()}")
    print(f"  {output_png.absolute()}")


if __name__ == '__main__':
    main()
