#!/usr/bin/env python3
"""
Create annotation set for layer evolution study.

Samples patches where LLM judge says "interpretable" at layer 0 OR layer 16,
then creates visualizations showing the image with red-boxed patch plus
top-5 LatentLens phrases (full captions) for both layers.

Output: 80 PNGs (8 per model × 10 models)

Usage:
    python scripts/analysis/create_layer_evolution_annotation_set.py
"""
import sys
sys.path.insert(0, '.')

import json
import random
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from llm_judge.utils import (
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    load_image,
    resize_and_pad,
)
from olmo.data.model_preprocessor import siglip_resize_and_pad, dino_resize_and_pad
from olmo.data.pixmo_datasets import PixMoCap


def process_image_for_encoder(image_path, vision_encoder):
    """
    Process image with the correct preprocessing for the vision encoder.

    - CLIP (vit-l-14-336): aspect-preserving resize with black padding
    - SigLIP: simple resize to square (no padding)
    - DINOv2: simple resize to square (no padding)
    - Qwen2-VL: center-crop (handled separately)

    Returns PIL Image at 512x512 for visualization.
    """
    image = load_image(image_path)

    if vision_encoder == 'vit-l-14-336':
        # CLIP: aspect-preserving resize with black padding
        processed, image_mask = resize_and_pad(image, (512, 512), normalize=False)
        processed = (processed * 255).astype(np.uint8)
    elif vision_encoder == 'siglip':
        # SigLIP: simple resize to square (no padding)
        processed, image_mask = siglip_resize_and_pad(image, (512, 512))
        processed = (processed * 255).astype(np.uint8)
    elif vision_encoder == 'dinov2-large-336':
        # DINOv2: simple resize to square (no padding)
        processed, image_mask = dino_resize_and_pad(image, (512, 512))
        processed = (processed * 255).astype(np.uint8)
    else:
        # Qwen2-VL or unknown: use center-crop (import canonical preprocessing)
        from PIL import Image as PILImage
        from scripts.analysis.qwen2_vl.preprocessing import preprocess_image_qwen2vl
        pil_img = PILImage.fromarray(image)
        return preprocess_image_qwen2vl(pil_img, target_size=512, force_square=True)

    from PIL import Image as PILImage
    return PILImage.fromarray(processed)


def highlight_token_in_caption(caption, token_str):
    """
    Return caption with token highlighted using brackets.
    E.g., "the train has the number 465907" + "907" -> "the train has the number 465[907]"
    """
    if not caption or not token_str:
        return caption or token_str or ""

    token_clean = token_str.strip()
    # Find token in caption (case-insensitive)
    idx = caption.lower().find(token_clean.lower())
    if idx == -1:
        # Token not found, just return caption
        return f"{caption} [{token_clean}]"

    # Insert brackets around the token
    return f"{caption[:idx]}[{caption[idx:idx+len(token_clean)]}]{caption[idx+len(token_clean):]}"


def create_visualization(image_path, patch_row, patch_col,
                         neighbors_layer0, neighbors_layer16,
                         interp_layer0, interp_layer16,
                         output_path, model_name, vision_encoder):
    """Create visualization with image, red bbox, and LatentLens phrases for both layers."""

    # Load and process image with correct preprocessing for this encoder
    processed_image = process_image_for_encoder(image_path, vision_encoder)

    # Calculate bbox (3x3 patch area)
    patch_size = 512 / 24
    bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=patch_size, size=3)

    # Draw bbox
    image_with_bbox = draw_bbox_on_image(processed_image, bbox, outline_color="red", width=3, fill_alpha=40)

    # Create figure layout - wider for phrases
    img_w, img_h = 512, 512
    text_width = 550
    margin = 20
    total_width = img_w + text_width + 3 * margin
    total_height = max(img_h + 2 * margin, 700)  # Taller for phrases

    vis_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(vis_img)

    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        title_font = subtitle_font = body_font = ImageFont.load_default()

    # Paste image
    vis_img.paste(image_with_bbox, (margin, margin))

    # Text area
    text_x = img_w + 2 * margin
    text_y = margin
    max_text_width = text_width - 20

    # Title
    draw.text((text_x, text_y), model_name, fill='black', font=title_font)
    text_y += 25
    draw.text((text_x, text_y), f"Patch ({patch_row}, {patch_col})", fill='gray', font=body_font)
    text_y += 30

    # Layer 0
    interp_marker = "[interpretable]" if interp_layer0 else "[NOT interpretable]"
    color = '#228B22' if interp_layer0 else '#DC143C'
    draw.text((text_x, text_y), f"Layer 0 {interp_marker}", fill=color, font=subtitle_font)
    text_y += 22

    for i, neighbor in enumerate(neighbors_layer0[:5]):
        token_str = neighbor.get('token_str', '')
        caption = neighbor.get('caption', '')

        # Create phrase with highlighted token
        phrase = highlight_token_in_caption(caption, token_str)

        # Truncate if too long
        if len(phrase) > 60:
            phrase = phrase[:57] + "..."

        # Yellow highlight background
        phrase_bbox = draw.textbbox((text_x + 10, text_y), phrase, font=body_font)
        draw.rectangle([phrase_bbox[0]-3, phrase_bbox[1]-2, phrase_bbox[2]+3, phrase_bbox[3]+2],
                      fill='#FFFF99')
        draw.text((text_x + 10, text_y), phrase, fill='black', font=body_font)
        text_y += 22

    text_y += 25

    # Layer 16
    interp_marker = "[interpretable]" if interp_layer16 else "[NOT interpretable]"
    color = '#228B22' if interp_layer16 else '#DC143C'
    draw.text((text_x, text_y), f"Layer 16 {interp_marker}", fill=color, font=subtitle_font)
    text_y += 22

    for i, neighbor in enumerate(neighbors_layer16[:5]):
        token_str = neighbor.get('token_str', '')
        caption = neighbor.get('caption', '')

        phrase = highlight_token_in_caption(caption, token_str)

        if len(phrase) > 60:
            phrase = phrase[:57] + "..."

        phrase_bbox = draw.textbbox((text_x + 10, text_y), phrase, font=body_font)
        draw.rectangle([phrase_bbox[0]-3, phrase_bbox[1]-2, phrase_bbox[2]+3, phrase_bbox[3]+2],
                      fill='#FFFF99')
        draw.text((text_x + 10, text_y), phrase, fill='black', font=body_font)
        text_y += 22

    vis_img.save(output_path, quality=95)
    return True


def load_contextual_nn_data(nn_file):
    """Load contextual NN data and index by (image_idx, row, col)."""
    if not nn_file.exists():
        return {}

    with open(nn_file) as f:
        data = json.load(f)

    index = {}
    for img_result in data.get('results', []):
        image_idx = img_result.get('image_idx', 0)

        # Handle both formats: trained models have chunks[], Qwen2-VL has patches[] directly
        chunks = img_result.get('chunks', [])
        if chunks:
            all_patches = [p for chunk in chunks for p in chunk.get('patches', [])]
        else:
            all_patches = img_result.get('patches', [])

        for patch in all_patches:
            row = patch.get('patch_row', 0)
            col = patch.get('patch_col', 0)
            key = (image_idx, row, col)
            neighbors = patch.get('nearest_contextual_neighbors', [])
            index[key] = neighbors

    return index


def main():
    random.seed(42)
    np.random.seed(42)

    # Output directory
    output_dir = Path('analysis_results/layer_evolution_annotation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset for image paths
    print("Loading PixMoCap dataset...")
    dataset = PixMoCap(split='validation', mode="captions")

    # Define models: (llm, encoder, llm_judge_template, contextual_nn_template)
    models = [
        # 9 trained models
        ('olmo-7b', 'vit-l-14-336',
         'llm_judge_olmo-7b_vit-l-14-336_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded'),
        ('olmo-7b', 'siglip',
         'llm_judge_olmo-7b_siglip_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded'),
        ('olmo-7b', 'dinov2-large-336',
         'llm_judge_olmo-7b_dinov2-large-336_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded'),
        ('llama3-8b', 'vit-l-14-336',
         'llm_judge_llama3-8b_vit-l-14-336_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded'),
        ('llama3-8b', 'siglip',
         'llm_judge_llama3-8b_siglip_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded'),
        ('llama3-8b', 'dinov2-large-336',
         'llm_judge_llama3-8b_dinov2-large-336_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded'),
        ('qwen2-7b', 'vit-l-14-336',
         'llm_judge_qwen2-7b_vit-l-14-336_seed10_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10_step12000-unsharded'),
        ('qwen2-7b', 'siglip',
         'llm_judge_qwen2-7b_siglip_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded'),
        ('qwen2-7b', 'dinov2-large-336',
         'llm_judge_qwen2-7b_dinov2-large-336_contextual{layer}_gpt5_cropped',
         'train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded'),
        # Qwen2-VL (off-the-shelf)
        ('qwen2vl', None,
         'qwen2-vl/llm_judge_qwen2vl_contextual{layer}_gpt5_cropped',
         'qwen2_vl/Qwen_Qwen2-VL-7B-Instruct'),
    ]

    llm_judge_base = Path('analysis_results/llm_judge_contextual_nn')
    contextual_nn_base = Path('analysis_results/contextual_nearest_neighbors')

    total_created = 0

    for llm, encoder, judge_template, nn_dir in models:
        model_name = f"{llm}+{encoder}" if encoder else llm
        print(f"\nProcessing {model_name}...")

        # Load LLM judge results for interpretability labels
        judge_dir0 = llm_judge_base / judge_template.format(layer=0)
        judge_dir16 = llm_judge_base / judge_template.format(layer=16)

        results0_file = judge_dir0 / "results_validation.json"
        results16_file = judge_dir16 / "results_validation.json"

        if not results0_file.exists() or not results16_file.exists():
            print(f"  Missing LLM judge files, skipping")
            continue

        with open(results0_file) as f:
            judge_data0 = json.load(f)
        with open(results16_file) as f:
            judge_data16 = json.load(f)

        # Index LLM judge results
        judge0_idx = {}
        for r in judge_data0.get('results', []):
            key = (r['image_idx'], r['patch_row'], r['patch_col'])
            judge0_idx[key] = r

        judge16_idx = {}
        for r in judge_data16.get('results', []):
            key = (r['image_idx'], r['patch_row'], r['patch_col'])
            judge16_idx[key] = r

        # Load contextual NN data for full phrases
        nn_file0 = contextual_nn_base / nn_dir / "contextual_neighbors_visual0_allLayers.json"
        nn_file16 = contextual_nn_base / nn_dir / "contextual_neighbors_visual16_allLayers.json"

        print(f"  Loading contextual NN data...")
        nn0_idx = load_contextual_nn_data(nn_file0)
        nn16_idx = load_contextual_nn_data(nn_file16)

        if not nn0_idx or not nn16_idx:
            print(f"  Missing contextual NN files, skipping")
            continue

        # Find patches interpretable at either layer
        candidates = []
        common_keys = set(judge0_idx.keys()) & set(judge16_idx.keys()) & set(nn0_idx.keys()) & set(nn16_idx.keys())

        for key in common_keys:
            interp0 = judge0_idx[key].get('interpretable', False)
            interp16 = judge16_idx[key].get('interpretable', False)

            if interp0 or interp16:
                candidates.append({
                    'key': key,
                    'neighbors0': nn0_idx[key],
                    'neighbors16': nn16_idx[key],
                    'interp0': interp0,
                    'interp16': interp16,
                })

        print(f"  Found {len(candidates)} interpretable patches")

        # Sample 8
        random.shuffle(candidates)
        sampled = candidates[:8]

        for i, sample in enumerate(sampled):
            image_idx, row, col = sample['key']

            # Get image path
            try:
                example = dataset.get(image_idx, np.random)
                image_path = example.get('image', '')
            except:
                print(f"  Could not load image {image_idx}")
                continue

            if not image_path or not Path(image_path).exists():
                print(f"  Image not found: {image_path}")
                continue

            # Create output filename
            safe_model = model_name.replace('+', '_').replace('-', '_')
            output_file = output_dir / f"{safe_model}_{i+1:02d}_img{image_idx}_r{row}c{col}.png"

            success = create_visualization(
                image_path=image_path,
                patch_row=row,
                patch_col=col,
                neighbors_layer0=sample['neighbors0'],
                neighbors_layer16=sample['neighbors16'],
                interp_layer0=sample['interp0'],
                interp_layer16=sample['interp16'],
                output_path=output_file,
                model_name=model_name,
                vision_encoder=encoder,
            )

            if success:
                total_created += 1
                print(f"  Created: {output_file.name}")

    print(f"\n{'='*60}")
    print(f"Created {total_created} visualizations")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
