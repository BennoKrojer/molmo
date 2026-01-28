#!/usr/bin/env python3
"""
Visualize examples where color words appear in early layers but change in later layers.

Creates PNG visualizations showing:
- The full image with bbox highlighting the patch
- How top-5 NNs evolve across visual layers
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import random
import re
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'llm_judge'))

from visual_attribute_words import VISUAL_ATTRIBUTES, get_attribute_type

# Import robust utilities from llm_judge
from llm_judge.utils import (
    process_image_with_mask,
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    sample_valid_patch_positions,
)


def extract_full_word_from_token(sentence: str, token: str) -> str:
    """Extract the full word containing the token from the sentence."""
    if not sentence:
        return token.strip() if token else ""

    token = token.strip() if token else ""
    if not token:
        return ""

    low_sent = sentence.lower()
    low_tok = token.lower()
    if not low_tok:
        return token

    idx = low_sent.find(low_tok)
    if idx == -1:
        return token

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == '_'

    start = idx
    end = idx + len(low_tok)

    token_has_space = any(ch.isspace() for ch in token)

    if not token_has_space:
        left_is_word = start > 0 and is_word_char(sentence[start - 1])
        right_is_word = end < len(sentence) and is_word_char(sentence[end])
        
        if left_is_word or right_is_word:
            exp_start = start
            exp_end = end
            while exp_start > 0 and is_word_char(sentence[exp_start - 1]):
                exp_start -= 1
            while exp_end < len(sentence) and is_word_char(sentence[exp_end]):
                exp_end += 1
            
            expanded = sentence[exp_start:exp_end]
            if not any(ch.isspace() for ch in expanded):
                return expanded

    return sentence[start:end]


def is_color_word(word):
    """Check if a word is a color word."""
    word_clean = word.lower().strip()
    attr_types = get_attribute_type(word_clean)
    return 'color' in attr_types


def find_color_transition_examples(contextual_nn_dir, checkpoint_name, max_examples=20):
    """
    Find patches where early layers have color words but later layers don't.
    Returns list of dicts with all necessary info including layer_words.
    """
    contextual_dir = Path(contextual_nn_dir) / checkpoint_name
    if not contextual_dir.exists():
        print(f"  WARNING: Directory not found: {contextual_dir}")
        return []
    
    nn_files = sorted(contextual_dir.glob("contextual_neighbors_visual*_allLayers.json"))
    if not nn_files:
        print(f"  WARNING: No NN files found in {contextual_dir}")
        return []
    
    # Get available layers
    available_layers = []
    for nn_file in nn_files:
        match = re.search(r'contextual_neighbors_visual(\d+)_allLayers\.json$', str(nn_file))
        if match:
            available_layers.append(int(match.group(1)))
    available_layers = sorted(available_layers)
    
    if not available_layers:
        return []
    
    early_layers = [l for l in available_layers if l <= 4]
    late_layers = [l for l in available_layers if l >= 16]
    
    if not early_layers or not late_layers:
        print(f"  WARNING: Need both early (<=4) and late (>=16) layers. Found: {available_layers}")
        return []
    
    print(f"  Available layers: {available_layers}")
    print(f"  Early layers: {early_layers}, Late layers: {late_layers}")
    
    # Load ALL data into memory once (keyed by (image_idx, row, col, layer))
    all_data = {}  # (image_idx, row, col) -> {layer -> words}
    image_paths = {}  # image_idx -> path
    
    for nn_file in tqdm(nn_files, desc="  Loading all layer data"):
        match = re.search(r'contextual_neighbors_visual(\d+)_allLayers\.json$', str(nn_file))
        if not match:
            continue
        visual_layer = int(match.group(1))
        
        with open(nn_file, 'r') as f:
            nn_results = json.load(f)
        
        for img_result in nn_results.get('results', []):
            image_idx = img_result.get('image_idx', 0)
            image_path = img_result.get('image_path', '')
            if image_path:
                image_paths[image_idx] = image_path
            
            for chunk in img_result.get('chunks', []):
                for patch in chunk.get('patches', []):
                    row = patch.get('patch_row', 0)
                    col = patch.get('patch_col', 0)
                    key = (image_idx, row, col)
                    
                    if key not in all_data:
                        all_data[key] = {}
                    
                    neighbors = patch.get('nearest_contextual_neighbors', [])
                    words = []
                    for neighbor in neighbors[:5]:
                        token_str = neighbor.get('token_str', '')
                        caption = neighbor.get('caption', '')
                        full_word = extract_full_word_from_token(caption, token_str)
                        words.append(full_word)
                    all_data[key][visual_layer] = words
    
    print(f"  Loaded data for {len(all_data)} unique patches")
    
    # Find transitions: color in early layers, no color in late layers
    transitions = []
    for key, layer_words in tqdm(all_data.items(), desc="  Finding transitions"):
        image_idx, row, col = key
        
        # Check early layers for color
        has_early_color = False
        for layer in early_layers:
            if layer in layer_words:
                if any(is_color_word(w) for w in layer_words[layer]):
                    has_early_color = True
                    break
        
        if not has_early_color:
            continue
        
        # Check late layers for no color
        has_late_color = False
        for layer in late_layers:
            if layer in layer_words:
                if any(is_color_word(w) for w in layer_words[layer]):
                    has_late_color = True
                    break
        
        if not has_late_color:
            transitions.append({
                'image_idx': image_idx,
                'patch_row': row,
                'patch_col': col,
                'image_path': image_paths.get(image_idx, ''),
                'layer_words': layer_words,
            })
    
    print(f"  Found {len(transitions)} color word transitions")
    
    # Sample
    random.shuffle(transitions)
    return transitions[:max_examples]


def create_visualization(example, output_path, llm, vision_encoder, dataset=None):
    """Create a visualization using the robust llm_judge utilities."""
    image_path = example.get('image_path', '')
    patch_row = example['patch_row']
    patch_col = example['patch_col']
    layer_words = example.get('layer_words', {})
    
    # Try to load the image
    if not image_path or not os.path.exists(image_path):
        if dataset is not None:
            try:
                img_example = dataset.get(example['image_idx'], np.random)
                image_path = img_example.get('image', '')
            except:
                pass
    
    if not image_path or not os.path.exists(image_path):
        print(f"    WARNING: Image not found: {image_path}")
        return False
    
    # Use robust utilities from llm_judge
    processed_image, image_mask = process_image_with_mask(image_path)
    
    # Calculate bbox using llm_judge utility
    patch_size = 512 / 24  # ~21.33 pixels per patch
    bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=patch_size, size=3)
    
    # Draw bbox using robust utility (handles clipping automatically)
    image_with_bbox = draw_bbox_on_image(processed_image, bbox, outline_color="red", width=3, fill_alpha=40)
    
    # Crop the region for display
    left, top, right, bottom = bbox
    # Clip manually for cropping
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(512, int(right))
    bottom = min(512, int(bottom))
    
    if right <= left or bottom <= top:
        print(f"    WARNING: Invalid crop region")
        return False
    
    cropped_image = processed_image.crop((left, top, right, bottom))
    crop_size = 120
    cropped_resized = cropped_image.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    
    # Create visualization layout
    img_w, img_h = 512, 512
    text_width = 500
    margin = 20
    
    layers = sorted(layer_words.keys())
    min_height = max(img_h + 2 * margin, len(layers) * 70 + 250)
    
    total_width = img_w + text_width + 4 * margin
    total_height = min_height
    
    vis_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(vis_img)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = subtitle_font = body_font = ImageFont.load_default()
    
    # Paste main image
    vis_img.paste(image_with_bbox, (margin, margin))
    
    # Text area
    text_x = img_w + 2 * margin
    text_y = margin
    
    # Title
    draw.text((text_x, text_y), f"Color Transition: {llm} + {vision_encoder}", fill='black', font=title_font)
    text_y += 30
    
    draw.text((text_x, text_y), f"Image {example['image_idx']}, Patch ({patch_row}, {patch_col})", fill='gray', font=body_font)
    text_y += 25
    
    # Cropped region
    vis_img.paste(cropped_resized, (text_x, text_y))
    text_y += crop_size + 15
    
    # Layer evolution
    draw.text((text_x, text_y), "NN Evolution (color words highlighted):", fill='black', font=subtitle_font)
    text_y += 25
    
    for layer in layers:
        words = layer_words.get(layer, [])
        
        # Color the layer label
        has_color = any(is_color_word(w) for w in words)
        label_color = '#E91E63' if has_color else '#333333'
        
        draw.text((text_x, text_y), f"Layer {layer}:", fill=label_color, font=subtitle_font)
        text_y += 18
        
        # Format words with color highlighting
        word_strs = []
        for w in words:
            if is_color_word(w):
                word_strs.append(f"[{w}]")
            else:
                word_strs.append(w)
        
        words_text = ", ".join(word_strs)
        
        # Word wrap
        max_chars = 50
        while len(words_text) > max_chars:
            split_idx = words_text[:max_chars].rfind(' ')
            if split_idx == -1:
                split_idx = max_chars
            draw.text((text_x + 10, text_y), words_text[:split_idx], fill='black', font=body_font)
            text_y += 16
            words_text = words_text[split_idx:].strip()
        
        draw.text((text_x + 10, text_y), words_text, fill='black', font=body_font)
        text_y += 22
    
    vis_img.save(output_path, quality=95)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize examples where color words appear in early layers but change later'
    )
    parser.add_argument('--num-examples', type=int, default=50)
    parser.add_argument('--contextual-nn-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.contextual_nn_dir is None:
        args.contextual_nn_dir = str(repo_root / 'analysis_results' / 'contextual_nearest_neighbors')
    
    if args.output_dir is None:
        args.output_dir = str(repo_root / 'analysis_results' / 'layer_evolution' / 'color_transitions')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = None
    try:
        from olmo.data.pixmo_datasets import PixMoCap
        dataset = PixMoCap(split='validation', mode="captions")
        print("Loaded PixMoCap dataset")
    except Exception as e:
        print(f"Could not load dataset: {e}")
    
    all_llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    all_encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    examples_per_model = max(1, args.num_examples // 9)
    
    print(f"\n{'='*80}")
    print(f"Finding color word transition examples")
    print(f"Target: {args.num_examples} total ({examples_per_model} per model)")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    all_examples = []
    
    for llm in all_llms:
        for vision_encoder in all_encoders:
            if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
                checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_seed10_step12000-unsharded"
            else:
                checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_step12000-unsharded"
            
            print(f"\n{llm} + {vision_encoder}:")
            
            examples = find_color_transition_examples(
                args.contextual_nn_dir,
                checkpoint_name,
                max_examples=examples_per_model
            )
            
            for ex in examples:
                ex['llm'] = llm
                ex['vision_encoder'] = vision_encoder
            
            all_examples.extend(examples)
            print(f"  Added {len(examples)} (total: {len(all_examples)})")
    
    if len(all_examples) > args.num_examples:
        random.shuffle(all_examples)
        all_examples = all_examples[:args.num_examples]
    
    print(f"\n{'='*80}")
    print(f"Creating {len(all_examples)} visualizations...")
    print(f"{'='*80}\n")
    
    success_count = 0
    for i, example in enumerate(tqdm(all_examples, desc="Creating visualizations")):
        output_path = output_dir / f"color_transition_{i+1:03d}_{example['llm']}_{example['vision_encoder']}.png"
        
        if create_visualization(example, output_path, example['llm'], example['vision_encoder'], dataset=dataset):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Created {success_count}/{len(all_examples)} visualizations")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
