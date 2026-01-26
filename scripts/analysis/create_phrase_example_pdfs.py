#!/usr/bin/env python3
"""
Generate PDFs for phrase annotation examples.

Shows images as vision encoders see them:
- CLIP (vit-l-14-336): resize + pad to 336x336
- SigLIP: squash to 384x384
- DINOv2: squash to 336x336
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Preprocessing parameters per vision encoder
VISION_CONFIGS = {
    'vit-l-14-336': {'size': 336, 'method': 'resize_pad', 'grid': 24},
    'siglip': {'size': 384, 'method': 'squash', 'grid': 27},
    'dinov2-large-336': {'size': 336, 'method': 'squash', 'grid': 24},
}


def preprocess_image(image, vision_encoder, patch_row, patch_col):
    """
    Apply vision encoder preprocessing and draw bbox.
    Returns (processed_image_array, grid_size).
    """
    config = VISION_CONFIGS.get(vision_encoder, VISION_CONFIGS['vit-l-14-336'])
    target_size = config['size']
    method = config['method']
    grid_size = config['grid']

    img_w, img_h = image.size

    if method == 'resize_pad':
        # CLIP: resize to fit, pad to square (preserves aspect ratio)
        scale = min(target_size / img_w, target_size / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Create padded square with black background (consistent with paper)
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        padded.paste(resized, (offset_x, offset_y))

        # Adjust patch coordinates for padding
        cell_orig_w = img_w / grid_size
        cell_orig_h = img_h / grid_size
        cell_new = target_size / grid_size

        # Patch position in original -> pixel -> new grid
        patch_x_orig = patch_col * cell_orig_w
        patch_y_orig = patch_row * cell_orig_h
        patch_x_new = patch_x_orig * scale + offset_x
        patch_y_new = patch_y_orig * scale + offset_y

        new_patch_col = patch_x_new / cell_new
        new_patch_row = patch_y_new / cell_new

        processed = padded

    else:  # squash
        # SigLIP/DINOv2: direct squash to square
        processed = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
        new_patch_col = patch_col
        new_patch_row = patch_row

    # Draw bbox
    draw = ImageDraw.Draw(processed)
    cell_size = target_size / grid_size
    x1 = int(new_patch_col * cell_size)
    y1 = int(new_patch_row * cell_size)
    x2 = int((new_patch_col + 1) * cell_size)
    y2 = int((new_patch_row + 1) * cell_size)

    for i in range(3):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='red')

    return np.array(processed), grid_size


def find_token_in_phrase(phrase, token):
    """Find token in phrase, expand to word boundaries."""
    token_clean = token.strip()
    idx = phrase.lower().find(token_clean.lower())
    if idx == -1:
        return phrase, "", ""

    start, end = idx, idx + len(token_clean)
    while start > 0 and (phrase[start-1].isalnum() or phrase[start-1] == '-'):
        start -= 1
    while end < len(phrase) and (phrase[end].isalnum() or phrase[end] == '-'):
        end += 1

    return phrase[:start], phrase[start:end], phrase[end:]


def draw_phrase_with_highlight(ax, x, y, phrase, token, fontsize=8):
    """Draw phrase with yellow-highlighted token."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    before, match, after = find_token_in_phrase(phrase, token)
    full_text = f'"{phrase}"'

    # Draw full phrase
    ax.text(x, y, full_text, fontsize=fontsize, style='italic',
            transform=ax.transAxes, verticalalignment='center', fontfamily='serif')

    # Overlay highlighted token
    if match:
        prefix = f'"{before}'
        t_temp = ax.text(x, y, prefix, fontsize=fontsize, style='italic',
                        transform=ax.transAxes, verticalalignment='center',
                        fontfamily='serif', alpha=0)
        bbox = t_temp.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        match_x = x + bbox.width

        ax.text(match_x, y, match, fontsize=fontsize, style='italic', fontweight='bold',
                transform=ax.transAxes, verticalalignment='center', fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.06', facecolor='#FFEB3B', edgecolor='none'))


def create_paired_pdf(example1, example2, img1_array, img2_array, output_path):
    """Create PDF with two examples side by side."""
    # Create figure with GridSpec for precise control
    fig = plt.figure(figsize=(6.8, 2.4), facecolor='#FAFAFA')

    # Layout: 2 columns, each with image on top and text below
    # Using manual axes placement for full control

    def draw_example(fig, example, img_array, x_offset):
        """Draw one example at the given x position (0-0.5 for left, 0.5-1 for right)."""
        width = 0.48

        # Model info at top
        model_info = example.get('model_info', '')
        model_short = (model_info.replace('llama3-8b', 'LLaMA3')
                       .replace('olmo-7b', 'OLMo').replace('qwen2-7b', 'Qwen2')
                       .replace('dinov2-large-336', 'DINOv2')
                       .replace('siglip', 'SigLIP').replace('vit-l-14-336', 'CLIP')
                       .replace(' + ', '+'))
        fig.text(x_offset + width/2, 0.96, model_short, fontsize=7, color='#555555',
                 ha='center', va='top', fontfamily='sans-serif')

        # Image - square, centered in upper portion
        img_size = 0.38  # fraction of figure width
        img_left = x_offset + (width - img_size) / 2
        ax_img = fig.add_axes([img_left, 0.38, img_size, 0.55])
        ax_img.imshow(img_array)
        ax_img.axis('off')

        # Text below image
        phrase = example['phrase']
        token = example['token']
        comparison = example.get('_comparison_phrase', '')

        # Create text axes
        ax_txt = fig.add_axes([x_offset + 0.02, 0.02, width - 0.04, 0.34])
        ax_txt.axis('off')
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        # LatentLens label and phrase
        ax_txt.text(0, 0.78, "LatentLens:", fontsize=8, fontweight='bold', color='#2E7D32',
                    transform=ax_txt.transAxes, va='center', fontfamily='sans-serif')
        draw_phrase_with_highlight(ax_txt, 0, 0.52, phrase, token, fontsize=7.5)

        # VG baseline label and phrase
        if comparison:
            ax_txt.text(0, 0.30, "VG baseline:", fontsize=8, fontweight='bold', color='#616161',
                        transform=ax_txt.transAxes, va='center', fontfamily='sans-serif')
            draw_phrase_with_highlight(ax_txt, 0, 0.04, comparison, token, fontsize=7.5)

    # Draw left example
    draw_example(fig, example1, img1_array, 0.01)

    # Draw right example
    draw_example(fig, example2, img2_array, 0.51)

    # Vertical divider
    fig.patches.append(mpatches.Rectangle(
        (0.50, 0.05), 0.003, 0.90,
        facecolor='#E0E0E0', edgecolor='none',
        transform=fig.transFigure
    ))

    # Outer border
    border = mpatches.FancyBboxPatch(
        (0.005, 0.015), 0.99, 0.97,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor='none', edgecolor='#CCCCCC',
        linewidth=0.7, transform=fig.transFigure
    )
    fig.patches.append(border)

    plt.savefig(output_path, format='pdf', dpi=300, facecolor='#FAFAFA')
    plt.close(fig)


def main():
    base_dir = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo")

    # Load metadata
    metadata_path = base_dir / "analysis_results" / "phrase_annotation_examples" / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load manifest
    manifest_path = base_dir / "paper" / "figures" / "phrase_examples" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = base_dir / "paper" / "figures" / "phrase_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset for images
    import sys
    sys.path.insert(0, str(base_dir))
    from olmo.data.pixmo_datasets import PixMoCap
    import random
    dataset = PixMoCap(split="validation", mode="captions")

    # Extract examples from manifest (handle both flat and nested formats)
    flat_manifest = []
    for entry in manifest:
        if 'examples' in entry:
            # Nested format from previous run
            flat_manifest.extend(entry['examples'])
        else:
            # Original flat format
            flat_manifest.append(entry)

    # Map manifest entries to metadata
    examples_with_images = []
    for entry in flat_manifest:
        target_phrase = entry['phrase']
        target_token = entry['token']

        for ex in metadata['examples']:
            if ex.get('phrase') == target_phrase and ex.get('token') == target_token:
                # Load and preprocess image
                img_idx = ex['img_idx']
                example_data = dataset.get(img_idx, random)
                image_path = example_data['image']

                if not Path(image_path).exists():
                    print(f"Warning: Image not found: {image_path}")
                    continue

                image = Image.open(image_path).convert('RGB')

                # Apply 3x3 bbox center offset
                patch_row = ex['patch_row'] + 1
                patch_col = ex['patch_col'] + 1

                # Get vision encoder from metadata
                vision = ex.get('vision', 'vit-l-14-336')
                # Normalize vision encoder name
                if 'siglip' in vision.lower():
                    vision_key = 'siglip'
                elif 'dinov2' in vision.lower():
                    vision_key = 'dinov2-large-336'
                else:
                    vision_key = 'vit-l-14-336'

                img_array, grid_size = preprocess_image(image, vision_key, patch_row, patch_col)

                examples_with_images.append((ex, img_array))
                break

    print(f"Loaded {len(examples_with_images)} examples")

    # Create paired PDFs (2 examples per PDF)
    for i in range(0, len(examples_with_images), 2):
        if i + 1 >= len(examples_with_images):
            # Odd number - skip last one or handle differently
            break

        ex1, img1 = examples_with_images[i]
        ex2, img2 = examples_with_images[i + 1]

        output_path = output_dir / f"phrase_example_{i//2:02d}.pdf"
        create_paired_pdf(ex1, ex2, img1, img2, output_path)
        print(f"Created: {output_path}")

    # Update manifest for paired format
    new_manifest = []
    for i in range(0, len(manifest), 2):
        if i + 1 >= len(manifest):
            break
        new_manifest.append({
            "pdf": f"phrase_example_{i//2:02d}.pdf",
            "examples": [manifest[i], manifest[i+1]]
        })

    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(new_manifest, f, indent=2)

    print(f"\nDone! Created {len(examples_with_images)//2} paired PDFs")


if __name__ == "__main__":
    main()
