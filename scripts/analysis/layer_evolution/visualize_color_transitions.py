#!/usr/bin/env python3
"""
Visualize examples where color words appear in early layers but change in later layers.

Creates PNG visualizations showing:
- The full image with bbox highlighting the patch
- The cropped patch region
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
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent.parent.parent.parent))

from visual_attribute_words import VISUAL_ATTRIBUTES, get_attribute_type


def extract_full_word_from_token(sentence: str, token: str) -> str:
    """
    Extract the full word containing the token from the sentence.
    """
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


def load_nn_data_for_patch(contextual_nn_dir, checkpoint_name, image_idx, patch_row, patch_col):
    """
    Load NN data for a specific patch across all visual layers.
    
    Returns:
        dict: {visual_layer: [list of top-5 full words]}
    """
    contextual_dir = Path(contextual_nn_dir) / checkpoint_name
    if not contextual_dir.exists():
        return {}
    
    result = {}
    nn_files = sorted(contextual_dir.glob("contextual_neighbors_visual*_allLayers.json"))
    
    for nn_file in nn_files:
        match = re.search(r'contextual_neighbors_visual(\d+)_allLayers\.json$', str(nn_file))
        if not match:
            continue
        
        visual_layer = int(match.group(1))
        
        with open(nn_file, 'r') as f:
            nn_results = json.load(f)
        
        results_list = nn_results.get('results', [])
        
        # Find the specific image
        for img_result in results_list:
            if img_result.get('image_idx') != image_idx:
                continue
            
            # Find the specific patch
            for chunk in img_result.get('chunks', []):
                for patch in chunk.get('patches', []):
                    if patch.get('patch_row') == patch_row and patch.get('patch_col') == patch_col:
                        neighbors = patch.get('nearest_contextual_neighbors', [])
                        words = []
                        for neighbor in neighbors[:5]:
                            token_str = neighbor.get('token_str', '')
                            caption = neighbor.get('caption', '')
                            full_word = extract_full_word_from_token(caption, token_str)
                            words.append(full_word)
                        result[visual_layer] = words
                        break
    
    return result


def find_color_transition_examples(contextual_nn_dir, checkpoint_name, max_examples=20):
    """
    Find patches where early layers have color words but later layers don't.
    
    Returns:
        list of dicts: [{image_idx, patch_row, patch_col, layer_words, image_path, ...}, ...]
    """
    contextual_dir = Path(contextual_nn_dir) / checkpoint_name
    if not contextual_dir.exists():
        print(f"  WARNING: Directory not found: {contextual_dir}")
        return []
    
    examples = []
    
    # First, load data from layer 0 or earliest available layer
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
    
    # Load all data and find transitions
    # First pass: collect all patches with color words in early layers
    candidates = []
    
    for early_layer in early_layers:
        nn_file = contextual_dir / f"contextual_neighbors_visual{early_layer}_allLayers.json"
        if not nn_file.exists():
            continue
        
        print(f"  Scanning layer {early_layer} for color words...")
        with open(nn_file, 'r') as f:
            nn_results = json.load(f)
        
        results_list = nn_results.get('results', [])
        
        for img_result in tqdm(results_list, desc=f"    Layer {early_layer}", leave=False):
            image_idx = img_result.get('image_idx', 0)
            image_path = img_result.get('image_path', '')
            
            for chunk in img_result.get('chunks', []):
                for patch in chunk.get('patches', []):
                    patch_row = patch.get('patch_row', 0)
                    patch_col = patch.get('patch_col', 0)
                    
                    neighbors = patch.get('nearest_contextual_neighbors', [])
                    words = []
                    has_color = False
                    
                    for neighbor in neighbors[:5]:
                        token_str = neighbor.get('token_str', '')
                        caption = neighbor.get('caption', '')
                        full_word = extract_full_word_from_token(caption, token_str)
                        words.append(full_word)
                        if is_color_word(full_word):
                            has_color = True
                    
                    if has_color:
                        candidates.append({
                            'image_idx': image_idx,
                            'patch_row': patch_row,
                            'patch_col': patch_col,
                            'image_path': image_path,
                            'early_layer': early_layer,
                            'early_words': words,
                        })
    
    print(f"  Found {len(candidates)} patches with color words in early layers")
    
    if not candidates:
        return []
    
    # Sample candidates to check
    random.shuffle(candidates)
    candidates = candidates[:min(500, len(candidates))]  # Limit to check
    
    # Second pass: check if color words disappear in late layers
    for late_layer in late_layers:
        nn_file = contextual_dir / f"contextual_neighbors_visual{late_layer}_allLayers.json"
        if not nn_file.exists():
            continue
        
        print(f"  Checking layer {late_layer} for transitions...")
        with open(nn_file, 'r') as f:
            nn_results = json.load(f)
        
        # Build lookup for fast access
        lookup = {}
        for img_result in nn_results.get('results', []):
            image_idx = img_result.get('image_idx', 0)
            for chunk in img_result.get('chunks', []):
                for patch in chunk.get('patches', []):
                    patch_row = patch.get('patch_row', 0)
                    patch_col = patch.get('patch_col', 0)
                    key = (image_idx, patch_row, patch_col)
                    
                    neighbors = patch.get('nearest_contextual_neighbors', [])
                    words = []
                    for neighbor in neighbors[:5]:
                        token_str = neighbor.get('token_str', '')
                        caption = neighbor.get('caption', '')
                        full_word = extract_full_word_from_token(caption, token_str)
                        words.append(full_word)
                    lookup[key] = words
        
        # Check candidates
        for cand in candidates:
            key = (cand['image_idx'], cand['patch_row'], cand['patch_col'])
            if key in lookup:
                late_words = lookup[key]
                has_color_late = any(is_color_word(w) for w in late_words)
                
                if not has_color_late and 'transition_layer' not in cand:
                    cand['transition_layer'] = late_layer
                    cand['late_words'] = late_words
    
    # Filter to only those with transitions
    transitions = [c for c in candidates if 'transition_layer' in c]
    print(f"  Found {len(transitions)} color word transitions")
    
    # Load full layer data for selected examples
    random.shuffle(transitions)
    selected = transitions[:max_examples]
    
    for example in tqdm(selected, desc="  Loading full layer data"):
        layer_words = load_nn_data_for_patch(
            contextual_nn_dir,
            checkpoint_name,
            example['image_idx'],
            example['patch_row'],
            example['patch_col']
        )
        example['layer_words'] = layer_words
    
    return selected


def create_visualization(example, output_path, llm, vision_encoder, dataset=None):
    """
    Create a visualization showing the patch, full image, and NN evolution.
    """
    image_path = example.get('image_path', '')
    patch_row = example['patch_row']
    patch_col = example['patch_col']
    layer_words = example.get('layer_words', {})
    
    # Try to load the image
    if not image_path or not os.path.exists(image_path):
        # Try to get from dataset
        if dataset is not None:
            try:
                img_example = dataset.get(example['image_idx'], np.random)
                image_path = img_example.get('image', '')
            except:
                pass
    
    if not image_path or not os.path.exists(image_path):
        print(f"    WARNING: Image not found: {image_path}")
        return False
    
    # Load and process image
    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"    WARNING: Could not load image: {e}")
        return False
    
    # Resize to 512x512 (standard processing size)
    processed_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Calculate bbox (3x3 patch region)
    patch_size = 512 / 24  # ~21.33 pixels per patch
    bbox_size = 3
    left = int(patch_col * patch_size)
    top = int(patch_row * patch_size)
    right = int((patch_col + bbox_size) * patch_size)
    bottom = int((patch_row + bbox_size) * patch_size)
    
    # Clip to image bounds
    left = max(0, left)
    top = max(0, top)
    right = min(512, right)
    bottom = min(512, bottom)
    
    # Draw bbox on image
    image_with_bbox = processed_image.copy()
    draw_bbox = ImageDraw.Draw(image_with_bbox)
    draw_bbox.rectangle([left, top, right, bottom], outline='red', width=3)
    
    # Crop the patch region
    cropped_image = processed_image.crop((left, top, right, bottom))
    crop_size = max(100, min(150, right - left, bottom - top))
    cropped_resized = cropped_image.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    
    # Create visualization layout
    # Left: image with bbox (512x512)
    # Middle: cropped region (150x150)
    # Right: layer evolution text
    
    margin = 20
    img_w, img_h = 512, 512
    crop_w, crop_h = crop_size, crop_size
    text_width = 450
    
    # Calculate total height based on number of layers
    layers = sorted(layer_words.keys())
    num_layers = len(layers)
    layer_height = 100  # Approximate height per layer
    min_height = max(img_h + 2 * margin, num_layers * layer_height + 200)
    
    total_width = img_w + crop_w + text_width + 5 * margin
    total_height = min_height
    
    # Create canvas
    vis_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(vis_img)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        title_font = subtitle_font = body_font = small_font = ImageFont.load_default()
    
    # Paste image with bbox
    vis_img.paste(image_with_bbox, (margin, margin))
    
    # Paste cropped region
    crop_x = img_w + 2 * margin
    crop_y = margin + 30
    vis_img.paste(cropped_resized, (crop_x, crop_y))
    draw.text((crop_x, margin), "Cropped Region:", fill='black', font=subtitle_font)
    
    # Add title
    title = f"Color Word Transition - {llm} + {vision_encoder}"
    draw.text((margin, img_h + 2 * margin), title, fill='black', font=title_font)
    
    # Add patch info
    info_text = f"Image {example['image_idx']}, Patch ({patch_row}, {patch_col})"
    draw.text((margin, img_h + 2 * margin + 25), info_text, fill='gray', font=body_font)
    
    # Draw layer evolution on the right side
    text_x = crop_x + crop_w + 2 * margin
    text_y = margin
    
    draw.text((text_x, text_y), "NN Evolution Across Visual Layers:", fill='black', font=subtitle_font)
    text_y += 30
    
    for layer in layers:
        words = layer_words.get(layer, [])
        
        # Check which words are color words
        word_strs = []
        for w in words:
            if is_color_word(w):
                word_strs.append(f"[{w}]")  # Highlight color words
            else:
                word_strs.append(w)
        
        layer_text = f"Layer {layer}: " + ", ".join(word_strs)
        
        # Color the layer label based on whether it has color words
        has_color = any(is_color_word(w) for w in words)
        label_color = '#E91E63' if has_color else '#333333'
        
        # Draw layer label
        draw.text((text_x, text_y), f"Layer {layer}:", fill=label_color, font=subtitle_font)
        text_y += 18
        
        # Draw words (wrap if needed)
        words_text = ", ".join(word_strs)
        # Simple word wrap
        max_chars = 45
        while len(words_text) > max_chars:
            # Find last space before limit
            split_idx = words_text[:max_chars].rfind(' ')
            if split_idx == -1:
                split_idx = max_chars
            draw.text((text_x + 10, text_y), words_text[:split_idx], fill='black', font=body_font)
            text_y += 16
            words_text = words_text[split_idx:].strip()
        draw.text((text_x + 10, text_y), words_text, fill='black', font=body_font)
        text_y += 25
    
    # Save
    vis_img.save(output_path, quality=95)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize examples where color words appear in early layers but change later'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=50,
        help='Number of examples to visualize (default: 50)'
    )
    parser.add_argument(
        '--contextual-nn-dir',
        type=str,
        default=None,
        help='Directory containing contextual nearest neighbors results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up paths
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    
    if args.contextual_nn_dir is None:
        args.contextual_nn_dir = str(repo_root / 'analysis_results' / 'contextual_nearest_neighbors')
    
    if args.output_dir is None:
        args.output_dir = str(repo_root / 'analysis_results' / 'layer_evolution' / 'color_transitions')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load dataset for image paths
    dataset = None
    try:
        from olmo.data.pixmo_datasets import PixMoCap
        dataset = PixMoCap(split='validation', mode="captions")
        print("Loaded PixMoCap dataset for image paths")
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Will try to use image paths from JSON files")
    
    # Define model combinations
    all_llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    all_encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    # Calculate examples per model
    examples_per_model = max(1, args.num_examples // 9)
    
    print(f"\n{'='*80}")
    print(f"Finding color word transition examples")
    print(f"{'='*80}")
    print(f"Target: {args.num_examples} total examples ({examples_per_model} per model combo)")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    all_examples = []
    
    for llm in all_llms:
        for vision_encoder in all_encoders:
            # Build checkpoint name
            if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
                checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_seed10_step12000-unsharded"
            else:
                checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_step12000-unsharded"
            
            print(f"\n{llm} + {vision_encoder}:")
            print(f"  Checkpoint: {checkpoint_name}")
            
            examples = find_color_transition_examples(
                args.contextual_nn_dir,
                checkpoint_name,
                max_examples=examples_per_model
            )
            
            for ex in examples:
                ex['llm'] = llm
                ex['vision_encoder'] = vision_encoder
                ex['checkpoint_name'] = checkpoint_name
            
            all_examples.extend(examples)
            print(f"  Added {len(examples)} examples (total: {len(all_examples)})")
    
    # Trim to exact number requested
    if len(all_examples) > args.num_examples:
        random.shuffle(all_examples)
        all_examples = all_examples[:args.num_examples]
    
    print(f"\n{'='*80}")
    print(f"Creating {len(all_examples)} visualizations...")
    print(f"{'='*80}\n")
    
    success_count = 0
    for i, example in enumerate(tqdm(all_examples, desc="Creating visualizations")):
        output_path = output_dir / f"color_transition_{i+1:03d}_{example['llm']}_{example['vision_encoder']}.png"
        
        success = create_visualization(
            example,
            output_path,
            example['llm'],
            example['vision_encoder'],
            dataset=dataset
        )
        
        if success:
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Created {success_count}/{len(all_examples)} visualizations")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

