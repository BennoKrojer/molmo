#!/usr/bin/env python3
"""
Create comparison figure showing LatentLens vs LogitLens for text-in-image.

Shows consecutive vision patches from a phone screenshot displaying
"The Couch Tomato Café" with BOTH LatentLens and LogitLens predictions.

Layer 30, OLMo+CLIP-ViT model.

Output: paper/figures/fig_text_patches_comparison_v{1,2,3}.pdf

Usage:
    python scripts/analysis/create_text_patches_comparison.py
"""
import sys
sys.path.insert(0, '.')

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


def resize_and_pad_clip(image, target_size=336):
    """CLIP preprocessing: aspect-preserving resize with black padding."""
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


def create_version_a(strip_upscaled, latentlens_tokens, logitlens_top3, num_patches, output_dir):
    """
    Version A: Compact v2 style with BIG labels very close to rows.
    - Large bold labels right next to the token rows
    - Yellow for LatentLens, Pink for LogitLens
    """
    fig_width = num_patches + 1.8
    fig_height = 2.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    label_x = -0.15  # Very close to tokens
    patch_start_x = 0

    strip_bottom = 1.6
    strip_height = 0.85

    # Draw image strip
    ax.imshow(strip_upscaled, extent=[patch_start_x, patch_start_x + num_patches,
                                       strip_bottom, strip_bottom + strip_height],
              aspect='equal', interpolation='nearest')

    # Draw vertical lines between patches
    for i in range(1, num_patches):
        ax.plot([patch_start_x + i, patch_start_x + i],
                [strip_bottom, strip_bottom + strip_height],
                color='black', linewidth=1.2)

    # Draw outer border
    rect = mpatches.Rectangle((patch_start_x, strip_bottom), num_patches, strip_height,
                               linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Row 1: LatentLens (yellow) - Layer 16
    latentlens_y = 1.05
    arrow_top = strip_bottom - 0.03
    arrow_bottom = latentlens_y + 0.18

    # BIG label very close
    ax.text(label_x, latentlens_y, 'LatentLens:', ha='right', va='center',
            fontsize=12, fontweight='bold')

    for i, token in enumerate(latentlens_tokens):
        x_center = patch_start_x + i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom), xytext=(x_center, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
        fontsize = 9 if len(token) <= 6 else 7
        ax.text(x_center, latentlens_y, token, ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFEB3B',
                         edgecolor='#FBC02D', linewidth=0.5, alpha=0.95))

    # Row 2: LogitLens top-2 inline (pink)
    logitlens_y = 0.4
    arrow_top2 = latentlens_y - 0.18
    arrow_bottom2 = logitlens_y + 0.18

    # BIG label very close
    ax.text(label_x, logitlens_y, 'LogitLens:', ha='right', va='center',
            fontsize=12, fontweight='bold')

    for i, top3 in enumerate(logitlens_top3):
        x_center = patch_start_x + i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom2), xytext=(x_center, arrow_top2),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
        combined = '/'.join(top3[:2])
        fontsize = 7 if len(combined) <= 12 else 6
        ax.text(x_center, logitlens_y, combined, ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFCDD2',
                         edgecolor='#EF9A9A', linewidth=0.5, alpha=0.95))

    ax.set_xlim(label_x - 1.1, patch_start_x + num_patches + 0.15)
    ax.set_ylim(0.05, strip_bottom + strip_height + 0.12)
    ax.axis('off')

    plt.tight_layout()

    output_pdf = output_dir / 'fig_text_patches_comparison_v1.pdf'
    output_png = output_dir / 'fig_text_patches_comparison_v1.png'
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()

    return output_pdf, output_png


def create_version_b(strip_upscaled, latentlens_tokens, logitlens_top3, num_patches, output_dir):
    """
    Version B: Big labels with layer info in parentheses.
    - Yellow for LatentLens, Blue for LogitLens
    """
    fig_width = num_patches + 2.2
    fig_height = 2.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    label_x = -0.15
    patch_start_x = 0

    strip_bottom = 1.6
    strip_height = 0.85

    # Draw image strip
    ax.imshow(strip_upscaled, extent=[patch_start_x, patch_start_x + num_patches,
                                       strip_bottom, strip_bottom + strip_height],
              aspect='equal', interpolation='nearest')

    # Draw vertical lines between patches
    for i in range(1, num_patches):
        ax.plot([patch_start_x + i, patch_start_x + i],
                [strip_bottom, strip_bottom + strip_height],
                color='black', linewidth=1.2)

    # Draw outer border
    rect = mpatches.Rectangle((patch_start_x, strip_bottom), num_patches, strip_height,
                               linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Row 1: LatentLens (yellow) - Layer 16
    latentlens_y = 1.05
    arrow_top = strip_bottom - 0.03
    arrow_bottom = latentlens_y + 0.18

    # Big label with layer info
    ax.text(label_x, latentlens_y, 'LatentLens (L16):', ha='right', va='center',
            fontsize=11, fontweight='bold')

    for i, token in enumerate(latentlens_tokens):
        x_center = patch_start_x + i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom), xytext=(x_center, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
        fontsize = 9 if len(token) <= 6 else 7
        ax.text(x_center, latentlens_y, token, ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFEB3B',
                         edgecolor='#FBC02D', linewidth=0.5, alpha=0.95))

    # Row 2: LogitLens top-2 inline (light blue)
    logitlens_y = 0.4
    arrow_top2 = latentlens_y - 0.18
    arrow_bottom2 = logitlens_y + 0.18

    # Big label with layer info
    ax.text(label_x, logitlens_y, 'LogitLens (L30):', ha='right', va='center',
            fontsize=11, fontweight='bold')

    for i, top3 in enumerate(logitlens_top3):
        x_center = patch_start_x + i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom2), xytext=(x_center, arrow_top2),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
        combined = '/'.join(top3[:2])
        fontsize = 7 if len(combined) <= 12 else 6
        ax.text(x_center, logitlens_y, combined, ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#87CEEB',
                         edgecolor='#5DADE2', linewidth=0.5, alpha=0.95))

    ax.set_xlim(label_x - 1.6, patch_start_x + num_patches + 0.15)
    ax.set_ylim(0.05, strip_bottom + strip_height + 0.12)
    ax.axis('off')

    plt.tight_layout()

    output_pdf = output_dir / 'fig_text_patches_comparison_v2.pdf'
    output_png = output_dir / 'fig_text_patches_comparison_v2.png'
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()

    return output_pdf, output_png


def create_version_c(strip_upscaled, latentlens_tokens, logitlens_top3, num_patches, output_dir):
    """
    Version C: Color-matched labels, very tight layout.
    - Gold label + yellow boxes for LatentLens
    - Teal label + teal boxes for LogitLens
    """
    fig_width = num_patches + 2.0
    fig_height = 2.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    label_x = -0.15
    patch_start_x = 0

    strip_bottom = 1.6
    strip_height = 0.85

    # Draw image strip
    ax.imshow(strip_upscaled, extent=[patch_start_x, patch_start_x + num_patches,
                                       strip_bottom, strip_bottom + strip_height],
              aspect='equal', interpolation='nearest')

    # Draw vertical lines between patches
    for i in range(1, num_patches):
        ax.plot([patch_start_x + i, patch_start_x + i],
                [strip_bottom, strip_bottom + strip_height],
                color='black', linewidth=1.2)

    # Draw outer border
    rect = mpatches.Rectangle((patch_start_x, strip_bottom), num_patches, strip_height,
                               linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Row 1: LatentLens (gold/amber) - Layer 16
    latentlens_y = 1.05
    arrow_top = strip_bottom - 0.03
    arrow_bottom = latentlens_y + 0.18

    # Big colored label
    ax.text(label_x, latentlens_y, 'LatentLens:', ha='right', va='center',
            fontsize=12, fontweight='bold', color='#B8860B')

    for i, token in enumerate(latentlens_tokens):
        x_center = patch_start_x + i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom), xytext=(x_center, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='#B8860B', lw=1.0))
        fontsize = 9 if len(token) <= 6 else 7
        ax.text(x_center, latentlens_y, token, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#5D4E00',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFD54F',
                         edgecolor='#B8860B', linewidth=0.8, alpha=0.95))

    # Row 2: LogitLens top-2 inline (teal)
    logitlens_y = 0.4
    arrow_top2 = latentlens_y - 0.18
    arrow_bottom2 = logitlens_y + 0.18

    # Big colored label
    ax.text(label_x, logitlens_y, 'LogitLens:', ha='right', va='center',
            fontsize=12, fontweight='bold', color='#00796B')

    for i, top3 in enumerate(logitlens_top3):
        x_center = patch_start_x + i + 0.5
        ax.annotate('', xy=(x_center, arrow_bottom2), xytext=(x_center, arrow_top2),
                    arrowprops=dict(arrowstyle='->', color='#00796B', lw=1.0))
        combined = '/'.join(top3[:2])
        fontsize = 7 if len(combined) <= 12 else 6
        ax.text(x_center, logitlens_y, combined, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#004D40',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#80CBC4',
                         edgecolor='#00796B', linewidth=0.8, alpha=0.95))

    ax.set_xlim(label_x - 1.15, patch_start_x + num_patches + 0.15)
    ax.set_ylim(0.05, strip_bottom + strip_height + 0.12)
    ax.axis('off')

    plt.tight_layout()

    output_pdf = output_dir / 'fig_text_patches_comparison_v3.pdf'
    output_png = output_dir / 'fig_text_patches_comparison_v3.png'
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()

    return output_pdf, output_png


def main():
    # Configuration - OLMo+CLIP
    # LatentLens @ Layer 16 (best for text reading)
    # LogitLens @ Layer 30 (best for text reading)
    IMAGE_IDX = 2  # Phone screenshot with "The Couch Tomato Café"
    ROW = 10
    MODEL_SIZE = 336
    PATCH_SIZE_MODEL = MODEL_SIZE // 24  # 14 pixels in model space
    COLS = list(range(7, 13))  # Columns 7-12

    # LatentLens Layer 16 predictions (verified from actual data)
    # These are text-specific interpretations that read the actual words
    latentlens_tokens = ['.the', 'couch', 'acon', 'tomato', 'iro', 'cafe']

    # LogitLens Layer 30 top-3 predictions (verified from actual data)
    # Each entry is [top1, top2, top3]
    logitlens_top3 = [
        ['couch', 'Couch', 'Coach'],      # Col 7 - "The"
        ['es', 'potato', 'Tomato'],       # Col 8 - "Couch"
        ['Tom', 'Tomato', 'es'],          # Col 9 - middle
        ['e', 'ato', 'cafe'],             # Col 10 - "Tomato"
        ['Cafe', 'cafe', 'restaurant'],   # Col 11 - middle
        ['"', '&', '-'],                  # Col 12 - "Cafe"
    ]

    print("=== Creating Text Patches Comparison Figure ===")
    print(f"LatentLens (Layer 16): {latentlens_tokens}")
    print(f"LogitLens (Layer 30) top-3: {logitlens_top3}")

    # Load original image
    print("\nLoading PixMoCap dataset...")
    dataset = PixMoCap(split='validation', mode='captions')
    example = dataset.get(IMAGE_IDX, np.random)
    img_path = example['image']
    original = Image.open(img_path).convert("RGB")
    print(f"Original size: {original.size}")

    # Apply EXACT preprocessing model uses
    preprocessed = resize_and_pad_clip(original, MODEL_SIZE)
    print(f"Preprocessed size: {preprocessed.shape}")

    # Get patch boundaries in 336 space
    y1 = ROW * PATCH_SIZE_MODEL
    y2 = y1 + PATCH_SIZE_MODEL
    x1 = COLS[0] * PATCH_SIZE_MODEL
    x2 = (COLS[-1] + 1) * PATCH_SIZE_MODEL

    print(f"Patch coords in 336 space: ({x1}, {y1}) to ({x2}, {y2})")

    # Extract strip directly from preprocessed image
    strip_orig = preprocessed[y1:y2, x1:x2]
    print(f"Extracted strip: {strip_orig.shape}")

    # Resize to medium resolution (24x24 per patch)
    DISPLAY_PATCH_SIZE = 24
    num_patches = len(COLS)
    target_h = DISPLAY_PATCH_SIZE
    target_w = DISPLAY_PATCH_SIZE * num_patches

    strip_pil = Image.fromarray(strip_orig)
    strip_resized = strip_pil.resize((target_w, target_h), Image.BILINEAR)
    strip_display = np.array(strip_resized)

    # Upscale 2x with nearest-neighbor for pixel visibility
    scale_factor = 2
    strip_upscaled = np.repeat(np.repeat(strip_display, scale_factor, axis=0), scale_factor, axis=1)
    print(f"Final strip: {strip_upscaled.shape}")

    # Output directory
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create all three versions
    print("\n=== Creating Version A (Standard two-row, top-3 stacked) ===")
    pdf_a, png_a = create_version_a(strip_upscaled, latentlens_tokens, logitlens_top3, num_patches, output_dir)
    print(f"  {pdf_a.absolute()}")

    print("\n=== Creating Version B (Compact, top-2 inline) ===")
    pdf_b, png_b = create_version_b(strip_upscaled, latentlens_tokens, logitlens_top3, num_patches, output_dir)
    print(f"  {pdf_b.absolute()}")

    print("\n=== Creating Version C (Color-coded, top-3 stacked) ===")
    pdf_c, png_c = create_version_c(strip_upscaled, latentlens_tokens, logitlens_top3, num_patches, output_dir)
    print(f"  {pdf_c.absolute()}")

    print("\n=== All versions created! ===")
    print(f"View PNG files to compare:")
    print(f"  {png_a.absolute()}")
    print(f"  {png_b.absolute()}")
    print(f"  {png_c.absolute()}")


if __name__ == '__main__':
    main()
