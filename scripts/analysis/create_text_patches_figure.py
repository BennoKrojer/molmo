#!/usr/bin/env python3
"""
Create inline figure showing text-in-image interpretation by LatentLens.

Shows consecutive vision patches from a phone screenshot displaying
"The Couch Tomato Café" with LatentLens predictions below each patch.

Output: paper/figures/fig_text_patches_inline.pdf

Usage:
    python scripts/analysis/create_text_patches_figure.py
"""
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from pathlib import Path

from olmo.data.model_preprocessor import resize_and_pad
from olmo.data.pixmo_datasets import PixMoCap


def main():
    # Configuration
    IMAGE_IDX = 2  # Phone screenshot with "The Couch Tomato Café"
    ROW = 10
    COLS = list(range(7, 13))  # Columns 7-12
    TARGET_SIZE = 336
    PATCH_SIZE = TARGET_SIZE // 24  # 14 pixels

    # LatentLens predictions (OLMo+CLIP-ViT, Layer 16)
    tokens = ['.the', 'couch', 'acon', 'tomato', 'iro', 'cafe']

    # Load and preprocess image using EXACT same preprocessing as model
    print("Loading PixMoCap dataset...")
    dataset = PixMoCap(split='validation', mode='captions')
    example = dataset.get(IMAGE_IDX, np.random)
    img_path = example['image']
    image = np.array(Image.open(img_path).convert("RGB"))

    print(f"Original image shape: {image.shape}")

    # Apply aspect-preserving resize + center padding
    processed, mask = resize_and_pad(image, (TARGET_SIZE, TARGET_SIZE))
    processed_uint8 = (processed * 255).astype(np.uint8)

    print(f"Processed image shape: {processed_uint8.shape}")

    # Extract the strip of patches
    y1 = ROW * PATCH_SIZE
    y2 = y1 + PATCH_SIZE
    x1 = COLS[0] * PATCH_SIZE
    x2 = (COLS[-1] + 1) * PATCH_SIZE
    strip = processed_uint8[y1:y2, x1:x2]

    print(f"Strip shape: {strip.shape}")

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 1.8))

    # Show the strip at the top
    strip_height = 0.35
    strip_bottom = 0.55
    ax.imshow(strip, extent=[0, 6, strip_bottom, strip_bottom + strip_height],
              aspect='auto', interpolation='nearest')

    # Draw vertical lines between patches (internal borders)
    for i in range(1, 6):
        ax.axvline(x=i, ymin=0.52, ymax=0.88, color='black', linewidth=1.5,
                   clip_on=False)

    # Draw outer border
    rect = mpatches.Rectangle((0, strip_bottom), 6, strip_height,
                               linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Draw arrows and yellow boxes with tokens
    arrow_top = strip_bottom - 0.02
    arrow_bottom = 0.32
    box_y = 0.08

    for i, token in enumerate(tokens):
        x_center = i + 0.5

        # Arrow from patch to token
        ax.annotate('', xy=(x_center, arrow_bottom), xytext=(x_center, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

        # Yellow box with token
        ax.text(x_center, box_y, token, ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                         edgecolor='none', alpha=0.7))

    # Clean up axes
    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(-0.05, 1.0)
    ax.axis('off')

    plt.tight_layout()

    # Save to paper/figures/
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / 'fig_text_patches_inline.pdf'
    output_png = output_dir / 'fig_text_patches_inline.png'

    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)

    print(f"\nSaved to:")
    print(f"  {output_pdf.absolute()}")
    print(f"  {output_png.absolute()}")


if __name__ == '__main__':
    main()
