#!/usr/bin/env python3
"""
Create figure comparing LN-Lens vs Static NN vs LogitLens.

Shows an image with a highlighted patch and arrows pointing to
text boxes showing the top-5 neighbors for each method.
"""

import sys
sys.path.insert(0, str(__file__).replace('/paper_plots/create_ln_lens_comparison.py', ''))

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision.transforms.functional import convert_image_dtype, InterpolationMode

from olmo.data.pixmo_datasets import PixMoCap

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output"

# Data paths
# Default model for single-panel figure
MODEL_KEY = "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10"
NN_DIR = PROJECT_ROOT / "analysis_results" / "nearest_neighbors" / MODEL_KEY
LOGIT_DIR = PROJECT_ROOT / "analysis_results" / "logit_lens" / MODEL_KEY
CONTEXTUAL_DIR = PROJECT_ROOT / "analysis_results" / "contextual_nearest_neighbors" / MODEL_KEY

# Models for 2x2 grid (different LLM + Vision encoder combinations)
GRID_MODELS = [
    {
        'key': 'train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_lite10',
        'display': 'OLMo-7B + CLIP'
    },
    {
        'key': 'train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded_lite10',
        'display': 'LLaMA3-8B + DINOv2'
    },
    {
        'key': 'train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded_lite10',
        'display': 'Qwen2-7B + SigLIP'
    },
    {
        'key': 'train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded_lite10',
        'display': 'OLMo-7B + SigLIP'
    },
]

# Layers for each subfigure in 2x2 grid
GRID_LAYERS = [0, 4, 16, 24]

# Image preprocessing constants (same as model uses)
IMAGE_SIZE = 336  # CLIP ViT-L/14-336
GRID_SIZE = 24    # 336 / 14 = 24 patches per dimension
PATCH_PIXELS = IMAGE_SIZE // GRID_SIZE  # 14 pixels per patch

# Dataset for loading images
_dataset = None

def get_dataset():
    """Get or create dataset instance."""
    global _dataset
    if _dataset is None:
        _dataset = PixMoCap(split="validation", mode="captions")
    return _dataset


def resize_and_pad_image(image, target_size=336):
    """
    Resize and pad image to square, matching model preprocessing.
    Returns PIL Image with black padding.
    """
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width = image.shape[:2]

    # Calculate scale to fit within target while preserving aspect ratio
    scale = min(target_size / height, target_size / width)
    scaled_height = int(height * scale)
    scaled_width = int(width * scale)

    # Resize using torch (matches training code)
    img_tensor = torch.permute(torch.from_numpy(image), [2, 0, 1])
    img_tensor = convert_image_dtype(img_tensor)
    img_tensor = torchvision.transforms.Resize(
        [scaled_height, scaled_width], InterpolationMode.BILINEAR, antialias=True
    )(img_tensor)
    img_tensor = torch.clip(img_tensor, 0.0, 1.0)
    resized = torch.permute(img_tensor, [1, 2, 0]).numpy()

    # Calculate centered padding
    top_pad = (target_size - scaled_height) // 2
    left_pad = (target_size - scaled_width) // 2

    # Pad with black (0)
    padded = np.pad(
        resized,
        [[top_pad, target_size - scaled_height - top_pad],
         [left_pad, target_size - scaled_width - left_pad],
         [0, 0]],
        constant_values=0
    )

    # Convert back to PIL (0-255 uint8)
    padded_uint8 = (padded * 255).astype(np.uint8)
    return Image.fromarray(padded_uint8)


def load_image(image_idx=0):
    """Load and preprocess image from PixMoCap dataset."""
    dataset = get_dataset()
    example = dataset.get(image_idx, np.random)
    image_data = example.get("image")
    if isinstance(image_data, str):
        raw_image = Image.open(image_data)
    elif isinstance(image_data, Image.Image):
        raw_image = image_data
    else:
        raise FileNotFoundError(f"Could not load image at index {image_idx}")

    # Convert to RGB if needed
    if raw_image.mode != 'RGB':
        raw_image = raw_image.convert('RGB')

    # Apply same preprocessing as model (resize + pad to square)
    return resize_and_pad_image(raw_image, target_size=IMAGE_SIZE)


def calculate_single_patch_bbox(row, col):
    """Calculate bounding box for a SINGLE patch on 336x336 image with 24x24 grid."""
    left = col * PATCH_PIXELS
    top = row * PATCH_PIXELS
    right = left + PATCH_PIXELS
    bottom = top + PATCH_PIXELS
    return (left, top, right, bottom)


def load_static_nn_data(image_idx=0, layer=0, model_key=None):
    """Load static NN data for a specific image and layer.

    Returns dict mapping patch_key -> patch_data
    """
    if model_key:
        nn_dir = PROJECT_ROOT / "analysis_results" / "nearest_neighbors" / model_key
    else:
        nn_dir = NN_DIR
    nn_file = nn_dir / f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{layer}.json"
    if not nn_file.exists():
        print(f"  Warning: NN file not found: {nn_file}")
        return None
    with open(nn_file) as f:
        data = json.load(f)

    # Structure: splits -> validation -> images -> [list] -> chunks -> patches
    images = data.get('splits', {}).get('validation', {}).get('images', [])
    if image_idx >= len(images):
        return None

    image_data = images[image_idx]
    result = {}
    for chunk in image_data.get('chunks', []):
        for patch in chunk.get('patches', []):
            row = patch.get('patch_row', 0)
            col = patch.get('patch_col', 0)
            patch_key = f"patch_{row}_{col}"
            result[patch_key] = patch
    return result


def load_logitlens_data(image_idx=0, layer=0, model_key=None):
    """Load LogitLens data for a specific image and layer.

    Returns dict mapping patch_key -> patch_data
    """
    if model_key:
        logit_dir = PROJECT_ROOT / "analysis_results" / "logit_lens" / model_key
    else:
        logit_dir = LOGIT_DIR
    logit_file = logit_dir / f"logit_lens_layer{layer}_topk5_multi-gpu.json"
    if not logit_file.exists():
        print(f"  Warning: LogitLens file not found: {logit_file}")
        return None
    with open(logit_file) as f:
        data = json.load(f)

    # Structure: results -> [image_idx] -> chunks -> patches
    results = data.get('results', [])
    if image_idx >= len(results):
        return None

    image_data = results[image_idx]
    result = {}
    for chunk in image_data.get('chunks', []):
        for patch in chunk.get('patches', []):
            row = patch.get('patch_row', 0)
            col = patch.get('patch_col', 0)
            patch_key = f"patch_{row}_{col}"
            result[patch_key] = patch
    return result


def load_contextual_nn_data(image_idx=0, model_key=None):
    """Load contextual NN data for a specific image.

    Returns dict mapping patch_key -> patch_data
    """
    if model_key:
        ctx_dir = PROJECT_ROOT / "analysis_results" / "contextual_nearest_neighbors" / model_key
    else:
        ctx_dir = CONTEXTUAL_DIR
    # Try the specific file first, then fallback to visual0 which has all images
    ctx_file = ctx_dir / f"contextual_neighbors_visual{image_idx}_allLayers.json"
    if not ctx_file.exists():
        ctx_file = ctx_dir / "contextual_neighbors_visual0_allLayers.json"
    if not ctx_file.exists():
        print(f"  Warning: Contextual file not found: {ctx_file}")
        return None

    with open(ctx_file) as f:
        data = json.load(f)

    # Structure: results -> [list of images] -> chunks -> patches
    results = data.get('results', [])

    # Find the result for this image_idx
    image_data = None
    for res in results:
        if res.get('image_idx') == image_idx:
            image_data = res
            break

    if image_data is None:
        print(f"  Warning: No contextual data for image {image_idx}")
        return None

    result = {}
    for chunk in image_data.get('chunks', []):
        for patch in chunk.get('patches', []):
            row = patch.get('patch_row', 0)
            col = patch.get('patch_col', 0)
            patch_key = f"patch_{row}_{col}"
            result[patch_key] = patch
    return result


def get_patch_key(row, col):
    """Generate patch key."""
    return f"patch_{row}_{col}"


def draw_elegant_arrow(fig, ax_from, ax_to, xy_from, xy_to, color='#333333', linewidth=1.2):
    """Draw an elegant arrow connecting two axes using ConnectionPatch."""
    arrow = ConnectionPatch(
        xyA=xy_from, xyB=xy_to,
        coordsA="data", coordsB="data",
        axesA=ax_from, axesB=ax_to,
        arrowstyle="-|>",
        shrinkA=2, shrinkB=2,
        mutation_scale=10,
        fc=color, ec=color,
        linewidth=linewidth,
        connectionstyle="arc3,rad=0.1"
    )
    fig.add_artist(arrow)
    return arrow


def create_text_box(ax, neighbors, method_name, is_contextual=False, color='#2196F3'):
    """Create a text box showing top-5 neighbors."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Method label at top
    ax.text(0.5, 0.95, method_name, fontsize=11, fontweight='bold',
            ha='center', va='top', color=color)

    # Draw neighbors
    y_pos = 0.82
    y_step = 0.16

    for i, neighbor in enumerate(neighbors[:5]):
        if is_contextual:
            # For LN-Lens: show sentence with highlighted word + layer badge
            token = neighbor.get('token_str', neighbor.get('token', ''))
            caption = neighbor.get('caption', '')
            layer = neighbor.get('contextual_layer', 0)
            sim = neighbor.get('similarity', 0)

            # Highlight the token in caption
            if caption and token:
                # Find token position and create highlighted text
                token_lower = token.lower().strip()
                caption_lower = caption.lower()

                if token_lower in caption_lower:
                    idx = caption_lower.find(token_lower)
                    before = caption[:idx]
                    matched = caption[idx:idx+len(token)]
                    after = caption[idx+len(token):]

                    # Truncate if too long
                    max_chars = 35
                    if len(caption) > max_chars:
                        if idx > max_chars // 3:
                            before = "..." + before[-(max_chars//3):]
                        if len(after) > max_chars // 3:
                            after = after[:max_chars//3] + "..."

                    # Draw text parts
                    display_text = f'"{before}'
                    ax.text(0.02, y_pos, display_text, fontsize=8,
                           ha='left', va='center', color='#444444',
                           family='monospace')

                    # Position for highlighted token
                    text_len = len(display_text)
                    x_offset = 0.02 + text_len * 0.018

                    # Highlighted token with background
                    token_bbox = dict(boxstyle='round,pad=0.15', facecolor='#FFEB3B',
                                     edgecolor='none', alpha=0.7)
                    ax.text(x_offset, y_pos, matched, fontsize=8, fontweight='bold',
                           ha='left', va='center', color='#333333',
                           family='monospace', bbox=token_bbox)

                    # After text
                    x_after = x_offset + len(matched) * 0.018 + 0.02
                    ax.text(x_after, y_pos, after + '"', fontsize=8,
                           ha='left', va='center', color='#444444',
                           family='monospace')

                    # Layer badge on the right
                    ax.text(0.95, y_pos, f'L{layer}', fontsize=7,
                           ha='right', va='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='#666666',
                                    edgecolor='none'))
                else:
                    # Token not found in caption - show token only
                    ax.text(0.02, y_pos, f'"{token}"', fontsize=8,
                           ha='left', va='center', color='#333333',
                           family='monospace')
                    ax.text(0.95, y_pos, f'L{layer}', fontsize=7,
                           ha='right', va='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='#666666',
                                    edgecolor='none'))
            else:
                ax.text(0.02, y_pos, f'{i+1}. {token}', fontsize=8,
                       ha='left', va='center', color='#333333')
        else:
            # For Static NN / LogitLens: just show token and similarity
            token = neighbor.get('token', neighbor.get('predicted_token', ''))
            sim = neighbor.get('similarity', neighbor.get('probability', 0))

            ax.text(0.02, y_pos, f'{i+1}. "{token}"', fontsize=9,
                   ha='left', va='center', color='#333333', family='monospace')
            ax.text(0.95, y_pos, f'{sim:.2f}', fontsize=8,
                   ha='right', va='center', color='#666666')

        y_pos -= y_step


def create_compact_ln_lens_box(ax, neighbors, top_k=3, font_scale=1.0):
    """Create a compact, fully-bounded text box for LN-Lens neighbors.

    Args:
        ax: Matplotlib axes
        neighbors: List of neighbor dicts
        top_k: Number of neighbors to show
        font_scale: Scale factor for fonts (use <1 for smaller boxes)
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Full box background
    ax.fill_between([0, 1], [0, 0], [1, 1], color='#F0F0F0', zorder=0)

    # Title bar at top
    title_height = 0.20
    ax.fill_between([0, 1], [1, 1], [1 - title_height, 1 - title_height],
                    color='#E8F5E9', zorder=1)
    ax.axhline(y=1 - title_height, color='#388E3C', linewidth=1, zorder=2)

    # Title
    ax.text(0.5, 1 - title_height/2, 'LN-Lens',
            fontsize=7*font_scale, fontweight='bold', ha='center', va='center',
            color='#2E7D32', fontfamily='serif', clip_on=True)

    # Draw top-k neighbors with FIXED positions (no renderer-dependent positioning)
    y_start = 0.70
    y_step = 0.20
    font_size = 6 * font_scale
    max_chars = 18  # Maximum characters for the entire line

    for i, neighbor in enumerate(neighbors[:top_k]):
        token = neighbor.get('token_str', neighbor.get('token', ''))
        caption = neighbor.get('caption', '')
        layer = neighbor.get('contextual_layer', 0)
        sim = neighbor.get('similarity', 0)

        y_pos = y_start - i * y_step

        if caption and token:
            token_lower = token.lower().strip()
            caption_lower = caption.lower()

            if token_lower in caption_lower:
                idx = caption_lower.find(token_lower)
                before = caption[:idx]
                matched = caption[idx:idx+len(token)]
                after = caption[idx+len(token):]

                # Keep full beginning, only truncate end if needed
                if len(after) > 8:
                    after = after[:7] + '…'

                # Use renderer to measure text and position yellow highlight properly
                renderer = ax.figure.canvas.get_renderer()

                # Part 1: number and opening quote + before text
                part1 = f'{i+1}. "{before}'
                t1 = ax.text(0.03, y_pos, part1, fontsize=font_size,
                       ha='left', va='center', color='#444444',
                       fontfamily='serif', clip_on=True)

                # Measure part1 to position highlighted token
                bbox1 = t1.get_window_extent(renderer=renderer)
                bbox1_data = ax.transData.inverted().transform(bbox1)
                x_token = bbox1_data[1][0] + 0.005

                # Part 2: highlighted token with YELLOW background
                t2 = ax.text(x_token, y_pos, matched, fontsize=font_size, fontweight='bold',
                       ha='left', va='center', color='#1B5E20',
                       fontfamily='serif', clip_on=True,
                       bbox=dict(boxstyle='round,pad=0.03', facecolor='#FFEB3B',
                                edgecolor='none', alpha=0.9))

                # Measure part2 to position after text
                bbox2 = t2.get_window_extent(renderer=renderer)
                bbox2_data = ax.transData.inverted().transform(bbox2)
                x_after = bbox2_data[1][0] + 0.005

                # Part 3: after text and closing quote
                ax.text(x_after, y_pos, f'{after}"', fontsize=font_size,
                       ha='left', va='center', color='#444444',
                       fontfamily='serif', clip_on=True)

                # Layer badge at far right - USER REQUIREMENT: "from LLM L{layer}"
                ax.text(0.97, y_pos, f'from LLM L{layer}', fontsize=font_size * 0.7,
                       ha='right', va='center', color='white', fontweight='bold',
                       clip_on=True,
                       bbox=dict(boxstyle='round,pad=0.05', facecolor='#666666',
                                edgecolor='none'))
            else:
                ax.text(0.03, y_pos, f'{i+1}. "{token[:15]}"', fontsize=font_size,
                       ha='left', va='center', color='#333333',
                       fontfamily='serif', clip_on=True)
        else:
            display_token = token[:15] if len(token) > 15 else token
            ax.text(0.03, y_pos, f'{i+1}. {display_token}', fontsize=font_size,
                   ha='left', va='center', color='#333333',
                   fontfamily='serif', clip_on=True)

    # Ellipsis
    ax.text(0.03, y_start - top_k * y_step + 0.04, '...', fontsize=font_size,
           ha='left', va='center', color='#666666', fontfamily='serif',
           fontweight='bold', clip_on=True)

    # Box border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#388E3C')
        spine.set_linewidth(1.5)


def create_compact_baseline_box(ax, neighbors, title, title_color, border_color, top_k=3, font_scale=1.0):
    """Create a compact box for baseline methods (Static NN, LogitLens).

    Args:
        ax: Matplotlib axes
        neighbors: List of neighbor dicts
        title: Box title
        title_color: Color for title text
        border_color: Color for box border
        top_k: Number of items to show
        font_scale: Scale factor for fonts
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Full box background
    ax.fill_between([0, 1], [0, 0], [1, 1], color='#F0F0F0', zorder=0)

    # Title bar at top
    title_height = 0.26
    title_bg = border_color + '20'
    ax.fill_between([0, 1], [1, 1], [1 - title_height, 1 - title_height],
                    color=title_bg, zorder=1)
    ax.axhline(y=1 - title_height, color=border_color, linewidth=1, zorder=2)

    # Title - no truncation, let it fit
    ax.text(0.5, 1 - title_height/2, title,
            fontsize=5.5 * font_scale, fontweight='bold', ha='center', va='center',
            color=title_color, fontfamily='serif', clip_on=True)

    # Draw tokens with FIXED positions
    y_start = 0.65
    y_step = 0.20
    font_size = 5.5 * font_scale

    for i, neighbor in enumerate(neighbors[:top_k]):
        y_pos = y_start - i * y_step

        token = neighbor.get('token', neighbor.get('predicted_token', ''))
        score = neighbor.get('similarity', neighbor.get('probability', neighbor.get('logit', 0)))

        # Truncate token to fit
        token_display = token.strip()
        if len(token_display) > 8:
            token_display = token_display[:7] + '…'

        # Format: "1. token (0.12)"
        ax.text(0.06, y_pos, f'{i+1}. {token_display}', fontsize=font_size,
               ha='left', va='center', color='#444444',
               fontfamily='serif', clip_on=True)

        ax.text(0.92, y_pos, f'({score:.2f})', fontsize=font_size * 0.9,
               ha='right', va='center', color='#666666',
               fontfamily='serif', clip_on=True)

    # Ellipsis
    ax.text(0.06, 0.08, '...', fontsize=font_size,
           ha='left', va='center', color='#666666', fontfamily='serif',
           fontweight='bold', clip_on=True)

    # Box border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(1.2)


def create_comparison_figure_v5(image_idx=0, patch_row=12, patch_col=12):
    """Create figure comparing LN-Lens with baselines.

    Layout:
    - Image on left
    - LN-Lens box (top right, full width of text area)
    - Static NN + LogitLens boxes (bottom right, side by side, half width each)
    """

    # Load all data
    image = load_image(image_idx)
    contextual = load_contextual_nn_data(image_idx)
    static_nn = load_static_nn_data(image_idx, layer=0)
    logitlens = load_logitlens_data(image_idx, layer=16)

    patch_key = get_patch_key(patch_row, patch_col)

    # Get neighbors for this patch
    ctx_neighbors = []
    nn_neighbors = []
    logit_neighbors = []

    if contextual and patch_key in contextual:
        ctx_neighbors = contextual[patch_key].get('nearest_contextual_neighbors', [])
    if static_nn and patch_key in static_nn:
        nn_neighbors = static_nn[patch_key].get('nearest_neighbors', [])
    if logitlens and patch_key in logitlens:
        preds = logitlens[patch_key].get('top_predictions', [])
        logit_neighbors = [{'token': p.get('token', ''), 'logit': p.get('logit', 0)} for p in preds]

    # Create figure
    fig = plt.figure(figsize=(7.5, 3.5))

    # Image on the left
    img_left = 0.02
    img_bottom = 0.05
    img_width = 0.52
    img_height = 0.9
    ax_img = fig.add_axes([img_left, img_bottom, img_width, img_height])

    # Text boxes area starts after image
    boxes_left = img_left + img_width + 0.02
    boxes_width = 0.42  # Total width for all boxes

    # LN-Lens box - top, full width
    ln_height = img_height * 0.52
    ln_bottom = img_bottom + img_height - ln_height
    ax_ln = fig.add_axes([boxes_left, ln_bottom, boxes_width, ln_height])

    # Baseline boxes - below LN-Lens, side by side
    baseline_height = img_height * 0.38
    baseline_bottom = img_bottom + img_height * 0.05
    baseline_width = (boxes_width - 0.02) / 2  # Half width minus gap
    gap = 0.02

    ax_nn = fig.add_axes([boxes_left, baseline_bottom, baseline_width, baseline_height])
    ax_logit = fig.add_axes([boxes_left + baseline_width + gap, baseline_bottom, baseline_width, baseline_height])

    # Draw image
    ax_img.imshow(image)
    ax_img.axis('off')

    # Calculate and draw bounding box
    bbox = calculate_single_patch_bbox(patch_row, patch_col)
    rect = Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='#E53935',
        facecolor='none'
    )
    ax_img.add_patch(rect)

    # Create text boxes
    create_compact_ln_lens_box(ax_ln, ctx_neighbors, top_k=3)
    create_compact_baseline_box(ax_nn, nn_neighbors, 'Embedding Matrix', '#1565C0', '#1976D2', top_k=3)
    create_compact_baseline_box(ax_logit, logit_neighbors, 'LogitLens', '#7B1FA2', '#9C27B0', top_k=3)

    # Draw arrow from patch toward center of all boxes
    # Point to middle vertically between LN-Lens and baselines
    arrow = ConnectionPatch(
        xyA=(bbox[2], (bbox[1] + bbox[3])/2),
        xyB=(0, 0.5),  # Middle of the boxes area
        coordsA="data", coordsB="axes fraction",
        axesA=ax_img, axesB=ax_ln,
        arrowstyle="-|>",
        shrinkA=3, shrinkB=3,
        mutation_scale=12,
        fc='#E53935', ec='#E53935',
        linewidth=1.5,
        connectionstyle="arc3,rad=0"
    )
    fig.add_artist(arrow)

    # Add "Top NNs" label at the arrow peak (near arrowhead, at the text boxes)
    # CRITICAL: ha='left' so text extends RIGHTWARD (away from image, into gap/box)
    # y=0.6 to position ABOVE the arrow line (which is at y=0.5)
    ax_ln.text(-0.02, 0.6, 'Top NNs',
               fontsize=6, ha='left', va='bottom',
               color='#C62828', fontweight='bold', fontfamily='serif',
               style='italic', transform=ax_ln.transAxes,
               clip_on=False)

    return fig


def create_comparison_figure_v4(image_idx=0, patch_row=12, patch_col=12):
    """Create elegant comparison figure - version 4 with proper arrows."""
    # Redirect to v5 for now
    return create_comparison_figure_v5(image_idx, patch_row, patch_col)


def create_single_panel(fig, panel_bounds, image_idx, patch_row, patch_col,
                        model_key, model_display, layer, subfig_label):
    """
    Create a single panel for the 2x2 grid with proper label positioning.

    All labels use FIGURE coordinates to ensure they don't overlap with content.
    """
    panel_left, panel_bottom, panel_width, panel_height = panel_bounds

    # === LAYOUT PARAMETERS ===
    # Reserve space at top for labels (subfig label + title)
    title_space = 0.10  # 10% of panel height for labels at top
    bottom_margin = 0.02

    # Content area proportions
    img_frac = 0.46      # Image takes 46% of panel width
    gap_frac = 0.04      # Gap takes 4% of panel width
    boxes_frac = 0.46    # Boxes take 46% of panel width
    # Total: 96%, leaving 4% margins

    # === CALCULATE CONTENT AREA ===
    content_bottom = panel_bottom + panel_height * bottom_margin
    content_height = panel_height * (1 - title_space - bottom_margin)
    content_left = panel_left + panel_width * 0.02  # Small left margin
    content_width = panel_width * 0.96

    # === IMAGE AXES ===
    img_left = content_left
    img_bottom = content_bottom
    img_width = content_width * img_frac
    img_height = content_height
    ax_img = fig.add_axes([img_left, img_bottom, img_width, img_height])

    # === TEXT BOXES AXES ===
    boxes_left = content_left + content_width * (img_frac + gap_frac)
    boxes_width = content_width * boxes_frac

    # LN-Lens box (top 55% of content height)
    ln_height = content_height * 0.55
    ln_bottom = content_bottom + content_height - ln_height
    ax_ln = fig.add_axes([boxes_left, ln_bottom, boxes_width, ln_height])

    # Baseline boxes (bottom 40%, side by side with small gap)
    baseline_height = content_height * 0.40
    baseline_bottom = content_bottom
    baseline_gap = 0.008
    half_width = (boxes_width - baseline_gap) / 2
    ax_nn = fig.add_axes([boxes_left, baseline_bottom, half_width, baseline_height])
    ax_logit = fig.add_axes([boxes_left + half_width + baseline_gap, baseline_bottom,
                             half_width, baseline_height])

    # === LOAD DATA ===
    image = load_image(image_idx)
    contextual = load_contextual_nn_data(image_idx, model_key=model_key)
    static_nn = load_static_nn_data(image_idx, layer=0, model_key=model_key)
    logitlens = load_logitlens_data(image_idx, layer=layer, model_key=model_key)

    patch_key = get_patch_key(patch_row, patch_col)

    ctx_neighbors = []
    nn_neighbors = []
    logit_neighbors = []

    if contextual and patch_key in contextual:
        all_ctx = contextual[patch_key].get('nearest_contextual_neighbors', [])
        ctx_neighbors = [n for n in all_ctx if n.get('contextual_layer', 0) == layer][:5]
        if not ctx_neighbors:
            ctx_neighbors = all_ctx[:5]
    if static_nn and patch_key in static_nn:
        nn_neighbors = static_nn[patch_key].get('nearest_neighbors', [])
    if logitlens and patch_key in logitlens:
        preds = logitlens[patch_key].get('top_predictions', [])
        logit_neighbors = [{'token': p.get('token', ''), 'logit': p.get('logit', 0)} for p in preds]

    # === DRAW IMAGE ===
    ax_img.imshow(image)
    ax_img.axis('off')

    # Bounding box on patch
    bbox = calculate_single_patch_bbox(patch_row, patch_col)
    rect = Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='#E53935',
        facecolor='none'
    )
    ax_img.add_patch(rect)

    # === DRAW TEXT BOXES ===
    create_compact_ln_lens_box(ax_ln, ctx_neighbors, top_k=3)
    create_compact_baseline_box(ax_nn, nn_neighbors, 'Embedding Matrix', '#1565C0', '#1976D2', top_k=3)
    create_compact_baseline_box(ax_logit, logit_neighbors, 'LogitLens', '#7B1FA2', '#9C27B0', top_k=3)

    # === DRAW ARROW ===
    arrow = ConnectionPatch(
        xyA=(bbox[2], (bbox[1] + bbox[3])/2),
        xyB=(0, 0.5),
        coordsA="data", coordsB="axes fraction",
        axesA=ax_img, axesB=ax_ln,
        arrowstyle="-|>",
        shrinkA=3, shrinkB=3,
        mutation_scale=10,
        fc='#E53935', ec='#E53935',
        linewidth=1.2,
        connectionstyle="arc3,rad=0"
    )
    fig.add_artist(arrow)

    # === LABELS (all using FIGURE coordinates) ===

    # 1. SUBFIGURE LABEL - big and prominent at top-left of panel
    fig.text(panel_left + 0.005, panel_bottom + panel_height - 0.01,
             subfig_label,
             fontsize=16, fontweight='bold',
             ha='left', va='top',
             color='black', fontfamily='serif')

    # 2. MODEL + LAYER - to the right of subfig label
    fig.text(panel_left + 0.05, panel_bottom + panel_height - 0.015,
             f'{model_display}  •  Layer {layer}',
             fontsize=9, ha='left', va='top',
             color='#333333', fontfamily='serif')

    # 3. "TOP NNs" - in the GAP between image and boxes
    # Calculate gap center in figure coordinates
    gap_center_x = content_left + content_width * (img_frac + gap_frac / 2)
    # Position vertically at ~60% of content height (above arrow which is at 50%)
    label_y = content_bottom + content_height * 0.62
    fig.text(gap_center_x, label_y,
             'Top NNs',
             fontsize=5, ha='center', va='bottom',
             color='#C62828', fontweight='bold', fontfamily='serif',
             style='italic', rotation=90)  # Rotated to fit in narrow gap

    return ax_img, ax_ln, ax_nn, ax_logit


def create_2x2_grid_figure():
    """
    Create 2x2 grid comparing LN-Lens across different models, images, and layers.
    """
    import random
    random.seed(42)

    image_indices = random.sample(range(10), 4)
    patch_positions = [(10, 9), (12, 14), (8, 8), (14, 12)]

    fig = plt.figure(figsize=(14, 10))

    # Layout with proper margins
    margin_left = 0.04
    margin_right = 0.02
    margin_top = 0.04
    margin_bottom = 0.04
    h_gap = 0.03
    v_gap = 0.05

    # Calculate panel dimensions
    panel_width = (1 - margin_left - margin_right - h_gap) / 2
    panel_height = (1 - margin_top - margin_bottom - v_gap) / 2

    panels = [
        (margin_left, margin_bottom + panel_height + v_gap, panel_width, panel_height),
        (margin_left + panel_width + h_gap, margin_bottom + panel_height + v_gap, panel_width, panel_height),
        (margin_left, margin_bottom, panel_width, panel_height),
        (margin_left + panel_width + h_gap, margin_bottom, panel_width, panel_height),
    ]

    subfig_labels = ['a)', 'b)', 'c)', 'd)']

    for model_info, layer, (img_idx, (patch_row, patch_col)), panel_bounds, label in zip(
        GRID_MODELS, GRID_LAYERS, zip(image_indices, patch_positions), panels, subfig_labels
    ):
        print(f"  Creating panel {label} - {model_info['display']}, image {img_idx}, layer {layer}")
        create_single_panel(
            fig, panel_bounds,
            image_idx=img_idx,
            patch_row=patch_row,
            patch_col=patch_col,
            model_key=model_info['key'],
            model_display=model_info['display'],
            layer=layer,
            subfig_label=label
        )

    return fig


def create_1x2_grid_figure():
    """
    Create 1x2 grid (two panels side by side) comparing LN-Lens across different models.
    """
    import random
    random.seed(42)

    # Just 2 images for 2 panels
    image_indices = [0, 3]  # Use image 0 and 3
    patch_positions = [(10, 9), (12, 12)]
    layers = [8, 16]  # Different layers

    fig = plt.figure(figsize=(14, 5))  # Wider, shorter for 1x2

    # Layout with proper margins
    margin_left = 0.03
    margin_right = 0.02
    margin_top = 0.08
    margin_bottom = 0.04
    h_gap = 0.03

    # Calculate panel dimensions - two panels side by side
    panel_width = (1 - margin_left - margin_right - h_gap) / 2
    panel_height = 1 - margin_top - margin_bottom

    panels = [
        (margin_left, margin_bottom, panel_width, panel_height),
        (margin_left + panel_width + h_gap, margin_bottom, panel_width, panel_height),
    ]

    subfig_labels = ['a)', 'b)']
    models_to_use = GRID_MODELS[:2]  # Use first 2 models

    for model_info, layer, (img_idx, (patch_row, patch_col)), panel_bounds, label in zip(
        models_to_use, layers, zip(image_indices, patch_positions), panels, subfig_labels
    ):
        print(f"  Creating panel {label} - {model_info['display']}, image {img_idx}, layer {layer}")
        create_single_panel(
            fig, panel_bounds,
            image_idx=img_idx,
            patch_row=patch_row,
            patch_col=patch_col,
            model_key=model_info['key'],
            model_display=model_info['display'],
            layer=layer,
            subfig_label=label
        )

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating LN-Lens comparison figure v4 (single panel)...")

    # Use patch on clock tower (row 10, col 9) - interesting semantic content
    fig = create_comparison_figure_v4(image_idx=0, patch_row=10, patch_col=9)

    for ext in ['pdf', 'png']:
        output_file = OUTPUT_DIR / f'ln_lens_comparison_v4.{ext}'
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved {output_file}")

    plt.close(fig)

    print("\nCreating 1x2 grid comparison figure...")
    fig_1x2 = create_1x2_grid_figure()

    for ext in ['pdf', 'png']:
        output_file = OUTPUT_DIR / f'ln_lens_comparison_1x2.{ext}'
        fig_1x2.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved {output_file}")

    plt.close(fig_1x2)
    print("Done!")


if __name__ == '__main__':
    main()
