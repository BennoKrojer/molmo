#!/usr/bin/env python3
"""
Create inline figure showing text-in-image interpretation by LatentLens.

Shows consecutive vision patches from a phone screenshot displaying
"The Couch Tomato Café" with LatentLens predictions below each patch.

Uses high-res crop from original image for readability, while patch
boundaries correspond to what the model actually sees (14x14 on 336x336).

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

    # Load original image
    print("Loading PixMoCap dataset...")
    dataset = PixMoCap(split='validation', mode='captions')
    example = dataset.get(IMAGE_IDX, np.random)
    img_path = example['image']
    original = Image.open(img_path).convert("RGB")
    orig_w, orig_h = original.size

    print(f"Original size: {orig_w} x {orig_h}")

    # Calculate the scaling used in preprocessing (aspect-preserving)
    scale = min(TARGET_SIZE / orig_h, TARGET_SIZE / orig_w)
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    left_pad = (TARGET_SIZE - scaled_w) // 2
    top_pad = (TARGET_SIZE - scaled_h) // 2

    print(f"Scale: {scale:.4f}, Scaled: {scaled_w}x{scaled_h}, Pad: left={left_pad}, top={top_pad}")

    # Map patch coordinates back to original image
    y1_proc = ROW * PATCH_SIZE
    y2_proc = y1_proc + PATCH_SIZE
    x1_proc = COLS[0] * PATCH_SIZE
    x2_proc = (COLS[-1] + 1) * PATCH_SIZE

    # Remove padding offset and map to original coordinates
    x1_orig = int((x1_proc - left_pad) / scale)
    x2_orig = int((x2_proc - left_pad) / scale)
    y1_orig = int((y1_proc - top_pad) / scale)
    y2_orig = int((y2_proc - top_pad) / scale)

    print(f"Original crop: ({x1_orig}, {y1_orig}) to ({x2_orig}, {y2_orig})")

    # Extract high-res strip from original
    original_array = np.array(original)
    strip_highres = original_array[y1_orig:y2_orig, x1_orig:x2_orig]
    print(f"High-res strip shape: {strip_highres.shape}")

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 1.8))

    strip_height = 0.40
    strip_bottom = 0.52
    ax.imshow(strip_highres, extent=[0, 6, strip_bottom, strip_bottom + strip_height],
              aspect='auto', interpolation='lanczos')

    # Draw vertical lines between patches
    for i in range(1, 6):
        ax.axvline(x=i, ymin=0.50, ymax=0.90, color='black', linewidth=1.5, clip_on=False)

    # Draw outer border
    rect = mpatches.Rectangle((0, strip_bottom), 6, strip_height,
                               linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Draw arrows and yellow boxes
    arrow_top = strip_bottom - 0.02
    arrow_bottom = 0.30
    box_y = 0.08

    for i, token in enumerate(tokens):
        x_center = i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom), xytext=(x_center, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
        ax.text(x_center, box_y, token, ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                         edgecolor='none', alpha=0.8))

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
