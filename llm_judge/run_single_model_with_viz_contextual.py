#!/usr/bin/env python3
"""
Run LLM judge evaluation for contextual nearest neighbors with visualizations.

This script consumes outputs produced by scripts like
  scripts/analysis/contextual_nearest_neighbors.py
which save JSON files containing, per image/patch, the top-k nearest contextual
neighbors (token in the context of a sentence).

It extracts the full word containing each token (expanding subwords to complete words),
and asks the LLM judge to evaluate these words using the IMAGE_PROMPT
(same as run_single_model_with_viz.py). Saves both JSON and visualization outputs.
Supports OpenAI and OpenRouter (e.g., Gemini) via flags consistent with other LLM judge scripts.
"""

import os
import sys
import io
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from openai import OpenAI

from prompts import IMAGE_PROMPT, IMAGE_PROMPT_WITH_CROP
from utils import (
    process_image_with_mask,
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    sample_valid_patch_positions,
)
from olmo.data.pixmo_datasets import PixMoCap


def encode_image_to_data_url(pil_image: Image.Image) -> str:
    import base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def get_llm_response(client, image_with_bbox, cropped_image, prompt_text,
                     api_provider: str, api_model: str):
    """
    Call LLM with IMAGE_PROMPT text and images; return parsed JSON response.
    """
    main_url = encode_image_to_data_url(image_with_bbox)
    crop_url = encode_image_to_data_url(cropped_image) if cropped_image is not None else None

    if api_provider == "openrouter":
        # OpenRouter uses OpenAI-compatible chat.completions
        content = [
            {"type": "image_url", "image_url": {"url": main_url}},
            {"type": "text", "text": prompt_text},
        ]
        if crop_url is not None:
            content.append({"type": "image_url", "image_url": {"url": crop_url}})

        resp = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": content}]
        )
        response_text = resp.choices[0].message.content
    else:
        # OpenAI Responses API (for GPT-5 style)
        content = [
            {"type": "input_image", "image_url": main_url},
            {"type": "input_text", "text": prompt_text},
        ]
        if crop_url is not None:
            content.append({"type": "input_image", "image_url": crop_url})

        resp = client.responses.create(
            model=api_model,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        )
        response_text = resp.output_text
    
    # Parse JSON from response
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        json_str = response_text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return {
        "interpretable": False,
        "concrete_words": [],
        "abstract_words": [],
        "global_words": [],
        "reasoning": f"Could not parse response as JSON. Response: {response_text[:200]}"
    }


def extract_full_word_from_token(sentence: str, token: str) -> str:
    """
    Extract the full word containing the token from the sentence.
    If the token is a subword within a larger word (e.g., "ing" in "rendering"),
    expand to return the entire containing word. Case-insensitive match.
    If not found, fall back to returning the token itself.
    """
    if not sentence:
        return token.strip() if token else ""

    # Strip whitespace from token (may have trailing/leading spaces from tokenizer)
    token = token.strip() if token else ""
    if not token:
        return ""

    low_sent = sentence.lower()
    low_tok = token.lower()
    if not low_tok:
        return token

    idx = low_sent.find(low_tok)
    if idx == -1:
        # No occurrence; return the token as-is
        return token

    # Expand to full word boundaries around the found occurrence
    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == '_'

    start = idx
    end = idx + len(low_tok)  # Use length of lowercase token for consistency

    # If token already contains whitespace, do not expand across words
    token_has_space = any(ch.isspace() for ch in token)

    if not token_has_space:
        # Check if token is embedded in a word (has word chars before or after)
        left_is_word = start > 0 and is_word_char(sentence[start - 1])
        right_is_word = end < len(sentence) and is_word_char(sentence[end])
        
        if left_is_word or right_is_word:
            # Expand to full word boundaries
            exp_start = start
            exp_end = end
            # Expand left to word boundary
            while exp_start > 0 and is_word_char(sentence[exp_start - 1]):
                exp_start -= 1
            # Expand right to word boundary
            while exp_end < len(sentence) and is_word_char(sentence[exp_end]):
                exp_end += 1
            
            expanded = sentence[exp_start:exp_end]
            # Only use expansion if it doesn't contain whitespace
            if not any(ch.isspace() for ch in expanded):
                return expanded

    # Default: return only the matched token range (no expansion)
    # Use original sentence slice to preserve case
    return sentence[start:end]


def extract_words_from_contextual(nearest_list):
    """
    Take top-5 nearest contextual neighbors and extract the full words.
    Each entry has fields like token_str, caption, position, similarity.
    Returns a list of expanded words.
    """
    words = []
    for i in range(min(5, len(nearest_list))):
        entry = nearest_list[i]
        token = entry.get('token_str', '')
        caption = entry.get('caption', '')
        word = extract_full_word_from_token(caption, token)
        if word:
            words.append(word)
    return words


def create_visualization(output_path: Path, image_with_bbox, words, gpt_response, cropped_image=None):
    """
    Draw image with bbox and a panel showing the candidate words and LLM decisions.
    Matches the format from run_single_model_with_viz.py
    """
    # Layout: left image, right text panel
    img_w, img_h = image_with_bbox.size
    margin = 20
    text_width = 700
    total_width = img_w + text_width + 3 * margin
    total_height = max(img_h + 2 * margin, 900)
    
    vis_img = Image.new('RGB', (total_width, total_height), color='white')
    vis_img.paste(image_with_bbox, (margin, margin))
    draw = ImageDraw.Draw(vis_img)

    # Fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = subtitle_font = body_font = small_font = ImageFont.load_default()
    
    # Helper functions
    def safe_wrap_text(text, font, max_width):
        words_list = text.split()
        lines = []
        current = ""
        for word in words_list:
            test = (current + " " + word) if current else word
            try:
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] <= max_width - 20:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            except:
                if len(test) * 8 <= max_width:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
        if current:
            lines.append(current)
        return lines
    
    def draw_text_block(y, lines, font, color, spacing=4):
        for line in lines:
            draw.text((text_x + 10, y), line, fill=color, font=font)
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                y += (bbox[3] - bbox[1]) + spacing
            except:
                y += 18 + spacing
        return y
    
    # Text position
    text_x = img_w + 2 * margin
    text_y = margin
    
    # Title
    draw.text((text_x, text_y), "Contextual Word Interpretability", fill='black', font=title_font)
    text_y += 40
    
    # Show cropped region if available
    if cropped_image is not None:
        # Resize cropped image to fit in text area
        crop_width = min(200, text_width - 20)
        crop_height = min(150, crop_width * cropped_image.height // cropped_image.width)
        cropped_resized = cropped_image.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
        
        # Paste cropped image
        crop_x = text_x + (text_width - crop_width) // 2
        vis_img.paste(cropped_resized, (crop_x, text_y))
        text_y += crop_height + 15
        
        # Label for cropped region
        draw.text((text_x, text_y), "Cropped Region:", fill='black', font=subtitle_font)
        text_y += 25
    
    # Result
    concrete_words = gpt_response.get('concrete_words', [])
    abstract_words = gpt_response.get('abstract_words', [])
    global_words = gpt_response.get('global_words', [])
    is_interpretable = len(concrete_words) > 0 or len(abstract_words) > 0 or len(global_words) > 0
    status_text = "INTERPRETABLE ✓" if is_interpretable else "NOT INTERPRETABLE ✗"
    status_color = 'green' if is_interpretable else 'red'
    draw.text((text_x, text_y), f"Result: {status_text}", fill=status_color, font=subtitle_font)
    text_y += 35
    
    # Candidate words
    draw.text((text_x, text_y), "Candidate Words:", fill='black', font=subtitle_font)
    text_y += 25
    words_str = ", ".join([f'"{w}"' for w in words])
    word_lines = safe_wrap_text(words_str, body_font, text_width)
    text_y = draw_text_block(text_y, word_lines, body_font, 'blue')
    text_y += 15
    
    # Concrete words
    if concrete_words:
        draw.text((text_x, text_y), "Concrete Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in concrete_words])
        word_lines = safe_wrap_text(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'green')
        text_y += 15
    
    # Abstract words
    if abstract_words:
        draw.text((text_x, text_y), "Abstract Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in abstract_words])
        word_lines = safe_wrap_text(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'blue')
        text_y += 15
    
    # Global words
    if global_words:
        draw.text((text_x, text_y), "Global Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in global_words])
        word_lines = safe_wrap_text(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'orange')
        text_y += 15
    
    # Reasoning
    reasoning = gpt_response.get('reasoning', '')
    if reasoning:
        draw.text((text_x, text_y), "GPT Reasoning:", fill='black', font=subtitle_font)
        text_y += 25
        reasoning_lines = safe_wrap_text(reasoning, body_font, text_width)
        text_y = draw_text_block(text_y, reasoning_lines, body_font, 'black')
        text_y += 15
    
    vis_img.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description='Run LLM judge for contextual nearest neighbors with visualizations')

    parser.add_argument('--llm', type=str, required=True, help='LLM name (e.g., olmo-7b)')
    parser.add_argument('--vision-encoder', type=str, required=True, help='Vision encoder name (e.g., vit-l-14-336)')
    parser.add_argument('--api-key_file', type=str, default="llm_judge/api_key.txt", help='API key file (OpenAI or OpenRouter)')
    parser.add_argument('--api-provider', type=str, default='openai', choices=['openai', 'openrouter'], help='API provider')
    parser.add_argument('--api-model', type=str, default='gpt-5', help='Model ID (e.g., gpt-5 or google/gemini-2.5-pro)')

    parser.add_argument('--base-dir', type=str, default='analysis_results/contextual_nearest_neighbors', help='Base directory with contextual NN outputs')
    parser.add_argument('--output-base', type=str, default='analysis_results/llm_judge_contextual_nn', help='Output base directory for judge results')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of patches per image')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'validation'])
    parser.add_argument('--layer', type=str, default='contextual16', help='Contextual layer identifier (e.g., contextual16); visual layer inferred in filename')
    parser.add_argument('--use-cropped-region', action='store_true', help='Optionally pass cropped region as second image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging (skip reasons always shown)')
    parser.add_argument('--checkpoint-name', type=str, default=None, help='Override checkpoint name (for ablations, etc.)')
    parser.add_argument('--model-name', type=str, default=None, help='Override model name for output directory (for ablations, etc.)')

    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build checkpoint name from llm/encoder (matching other scripts)
    if args.checkpoint_name is not None:
        # Use provided checkpoint name (for ablations)
        checkpoint_name = args.checkpoint_name
        if args.model_name is not None:
            model_name = args.model_name
        else:
            # Extract model name from checkpoint name
            model_name = checkpoint_name.replace("train_mlp-only_pixmo_cap_resize_", "").replace("train_mlp-only_pixmo_cap_", "").replace("train_mlp-only_pixmo_points_resize_", "").replace("train_mlp-only_pixmo_topbottom_", "").replace("_step12000-unsharded", "")
    elif args.llm == "qwen2-7b" and args.vision_encoder == "vit-l-14-336":
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}_seed10"
        model_name = f"{args.llm}_{args.vision_encoder}_seed10"
    else:
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}"
        model_name = f"{args.llm}_{args.vision_encoder}"

    # Input discovery: contextual_nearest_neighbors_allLayers.py saves to
    # analysis_results/contextual_nearest_neighbors/<ckpt_name_step>/contextual_neighbors_visual{v}_allLayers_multi-gpu.json
    # These files contain neighbors from ALL contextual layers, with each neighbor having a 'contextual_layer' field
    base_dir = Path(args.base_dir)
    # Try multiple path patterns (with/without _step12000-unsharded suffix)
    ckpt_dir_candidates = [
        base_dir / f"{checkpoint_name}_step12000-unsharded",
        base_dir / checkpoint_name,  # For off-the-shelf models like Qwen2-VL
    ]
    ckpt_dir = None
    for cand in ckpt_dir_candidates:
        if cand.exists():
            ckpt_dir = cand
            break
    if ckpt_dir is None:
        print(f"ERROR: Checkpoint directory not found. Tried:")
        for cand in ckpt_dir_candidates:
            print(f"  - {cand}")
        sys.exit(1)
    # Try to find an allLayers file (we'll filter by contextual layer later)
    input_json = None
    target_layer = int(args.layer.replace('contextual', ''))
    
    # Look for allLayers files first (new format - try both with and without _multi-gpu suffix)
    allLayers_files = sorted(ckpt_dir.glob("contextual_neighbors_visual*_allLayers.json"))
    if not allLayers_files:
        # Fallback to old naming with _multi-gpu suffix
        allLayers_files = sorted(ckpt_dir.glob("contextual_neighbors_visual*_allLayers_multi-gpu.json"))
    if allLayers_files:
        # FIXED: Use the file matching the target visual layer
        # (contextual16 → visual16_allLayers.json, NOT visual0)
        target_file = ckpt_dir / f"contextual_neighbors_visual{target_layer}_allLayers.json"
        if not target_file.exists():
            # Try with _multi-gpu suffix
            target_file = ckpt_dir / f"contextual_neighbors_visual{target_layer}_allLayers_multi-gpu.json"
        if target_file.exists():
            input_json = target_file
        else:
            print(f"ERROR: Could not find visual{target_layer}_allLayers.json in {ckpt_dir}")
            print(f"Available files: {[f.name for f in allLayers_files]}")
            sys.exit(1)
    else:
        # Fallback to old format: contextual_neighbors_visual{v}_contextual{c}_multi-gpu.json
        for cand in sorted(ckpt_dir.glob("contextual_neighbors_visual*_contextual*_multi-gpu.json")):
            # Extract the contextual layer number from filename using regex
            match = re.search(r'_contextual(\d+)_multi-gpu\.json$', str(cand))
            if match:
                file_layer = int(match.group(1))
                # Match exact layer number (handles contextual0 specially)
                if file_layer == target_layer or (args.layer == 'contextual0' and file_layer == 0):
                    input_json = cand
                    break
    if input_json is None:
        print(f"ERROR: Contextual NN JSON not found in {ckpt_dir} for layer {args.layer}")
        sys.exit(1)

    # Output dir with model suffix
    model_suffix = args.api_model
    if "/" in model_suffix:
        model_suffix = model_suffix.split("/")[-1]
    model_suffix = model_suffix.replace("-", "").replace(".", "").replace(":", "_").lower()
    model_suffix = model_suffix.replace("_free", "").replace("free", "")

    # Extract layer number from layer argument (e.g., "contextual16" -> "16")
    layer_num = args.layer.replace('contextual', '')

    output_dir_name = f"llm_judge_{model_name}_contextual{layer_num}_{model_suffix}"
    if args.use_cropped_region:
        output_dir_name += "_cropped"
    output_dir = Path(args.output_base) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    output_json_path = output_dir / f"results_{args.split}.json"

    print(f"\n{'='*60}")
    print(f"Contextual NN Judge - Model: {args.llm} + {args.vision_encoder}")
    print(f"Input: {input_json}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # API client
    with open(args.api_key_file, 'r', encoding='utf-8') as f:
        api_key = f.read().strip()
    if args.api_provider == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)

    # Load input data and dataset
    # Use streaming JSON parser for large files to avoid MemoryError
    file_size_gb = os.path.getsize(input_json) / (1024**3)
    print(f"Loading contextual neighbors JSON file ({file_size_gb:.2f} GB - this may take a moment for large files)...")
    print(f"Processing {args.num_images} images using streaming parser to avoid MemoryError...")
    
    dataset = PixMoCap(split=args.split, mode="captions")

    # Limit to num_images
    results = []
    images_processed = 0
    
    # Debug counters
    debug_stats = {
        'images_skipped_no_file': 0,
        'images_skipped_no_valid_patches': 0,
        'patches_skipped_no_neighbors': 0,
        'patches_skipped_no_words': 0,
        'patches_processed': 0,
    }

    # Use streaming parser - process items one at a time instead of loading all into memory
    import ijson
    with open(input_json, 'rb') as f:
        # Parse the 'results' array items one by one and process immediately
        parser = ijson.items(f, 'results.item')
        
        # Process images one at a time as we stream them - don't accumulate in memory
        for i, image_obj in enumerate(tqdm(parser, desc="Processing images", total=args.num_images)):
            if args.num_images is not None and i >= args.num_images:
                break
            
            image_idx = image_obj.get("image_idx", 0)
            example = dataset.get(image_idx, np.random)
            image_path = example["image"]
            if not os.path.exists(image_path):
                debug_stats['images_skipped_no_file'] += 1
                print(f"SKIP Image {image_idx}: File not found at {image_path}")
                continue

            # Handle both formats to calculate grid_size FIRST:
            # - Molmo format: chunks[...][patches][...]['nearest_contextual_neighbors']
            # - Qwen2-VL format: patches[...]['nearest_contextual_neighbors'] (no chunks)
            patch_map = defaultdict(list)  # key = (row,col) -> nearest_contextual list
            max_row, max_col = 0, 0

            if 'chunks' in image_obj and image_obj['chunks']:
                # Molmo format
                for ch in image_obj['chunks']:
                    for p in ch.get('patches', []):
                        row = p.get('patch_row', 0)
                        col = p.get('patch_col', 0)
                        max_row = max(max_row, row)
                        max_col = max(max_col, col)
                        all_neighbors = p.get('nearest_contextual_neighbors', [])
                        patch_map[(row, col)] = all_neighbors
            elif 'patches' in image_obj:
                # Qwen2-VL format (patches directly, no chunks)
                for p in image_obj['patches']:
                    row = p.get('patch_row', 0)
                    col = p.get('patch_col', 0)
                    max_row = max(max_row, row)
                    max_col = max(max_col, col)
                    all_neighbors = p.get('nearest_contextual_neighbors', [])
                    patch_map[(row, col)] = all_neighbors

            # Calculate actual grid dimensions from patches (handles variable grids)
            grid_rows = max_row + 1
            grid_cols = max_col + 1
            grid_size = max(grid_rows, grid_cols)  # Use larger dimension for square bbox calculation

            processed_image, image_mask = process_image_with_mask(image_path, model_name=model_name)

            # Sample patches uniformly from valid positions (using correct grid_size)
            sampled_positions = sample_valid_patch_positions(image_mask, bbox_size=3, num_samples=args.num_samples, grid_size=grid_size)

            if not sampled_positions:
                debug_stats['images_skipped_no_valid_patches'] += 1
                print(f"SKIP Image {image_idx}: No valid patch positions found")
                continue

            image_result_entries = []
            
            if args.debug:
                print(f"[DEBUG] Image {image_idx}: Found {len(patch_map)} patches, grid {grid_rows}x{grid_cols}, sampled {len(sampled_positions)} positions")

            for patch_row, patch_col in sampled_positions:
                bbox_size = 3
                # Use dynamic grid size from actual data (not hardcoded 24)
                actual_patch_size = 512 / grid_size
                bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=bbox_size)
                image_with_bbox = draw_bbox_on_image(processed_image, bbox)

                # Calculate center patch coordinates (same as run_single_model_with_viz.py)
                # The JSON stores patches at their CENTER positions, not top-left corner
                center_row = patch_row + bbox_size // 2
                center_col = patch_col + bbox_size // 2
                nearest_contextual = patch_map.get((center_row, center_col), [])
                if not nearest_contextual:
                    debug_stats['patches_skipped_no_neighbors'] += 1
                    print(f"SKIP Image {image_idx} patch ({patch_row},{patch_col}) center ({center_row},{center_col}): No neighbors found in patch_map")
                    continue

                # Extract full words from tokens
                words = extract_words_from_contextual(nearest_contextual)
                if not words:
                    debug_stats['patches_skipped_no_words'] += 1
                    print(f"SKIP Image {image_idx} patch ({patch_row},{patch_col}): No words extracted from {len(nearest_contextual)} neighbors")
                    continue
                
                debug_stats['patches_processed'] += 1

                # Select prompt variant based on whether we send a cropped region
                cropped_image = None
                if args.use_cropped_region:
                    # Crop using dynamic grid size (not hardcoded 24)
                    left = int(patch_col * actual_patch_size)
                    top = int(patch_row * actual_patch_size)
                    right = int((patch_col + bbox_size) * actual_patch_size)
                    bottom = int((patch_row + bbox_size) * actual_patch_size)
                    left = max(0, left); top = max(0, top)
                    right = min(right, processed_image.size[0]); bottom = min(bottom, processed_image.size[1])
                    cropped_image = processed_image.crop((left, top, right, bottom))

                    prompt_text = IMAGE_PROMPT_WITH_CROP.format(candidate_words=str(words))
                else:
                    prompt_text = IMAGE_PROMPT.format(candidate_words=str(words))

                # Get LLM response (returns parsed dict)
                gpt_response = get_llm_response(
                    client, image_with_bbox, cropped_image, prompt_text,
                    api_provider=args.api_provider, api_model=args.api_model
                )

                # Check if interpretable
                concrete_words = gpt_response.get('concrete_words', [])
                abstract_words = gpt_response.get('abstract_words', [])
                global_words = gpt_response.get('global_words', [])
                is_interpretable = len(concrete_words) > 0 or len(abstract_words) > 0 or len(global_words) > 0

                # Save visualization
                status_str = 'pass' if is_interpretable else 'fail'
                vis_name = f"img{image_idx}_r{patch_row}_c{patch_col}_{status_str}.jpg"
                create_visualization(viz_dir / vis_name, image_with_bbox, words, gpt_response, cropped_image=cropped_image)

                # Accumulate result entry
                image_result_entries.append({
                    'image_idx': image_idx,
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'words': words,
                    'gpt_response': gpt_response,
                    'interpretable': is_interpretable,
                })

                # Incremental save
                tmp = {
                    'llm': args.llm,
                    'vision_encoder': args.vision_encoder,
                    'api_provider': args.api_provider,
                    'api_model': args.api_model,
                    'split': args.split,
                    'input_json': str(input_json),
                    'results': results + image_result_entries,
                }
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(tmp, f, indent=2, ensure_ascii=False)

            # Finalize per-image aggregation
            if image_result_entries:
                results.extend(image_result_entries)
                images_processed += 1

    # Final save
    out_payload = {
        'llm': args.llm,
        'vision_encoder': args.vision_encoder,
        'api_provider': args.api_provider,
        'api_model': args.api_model,
        'split': args.split,
        'input_json': str(input_json),
        'num_images_processed': images_processed,
        'results': results,
    }
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    # Print brief summary
    total = len(results)
    acc = sum(1 for r in results if r.get('interpretable'))
    pct = (acc / total * 100.0) if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Results: {acc}/{total} ({pct:.1f}%)")
    print(f"Saved to: {output_json_path}")
    print(f"Visualizations: {viz_dir}")
    print(f"{'='*60}")
    
    # Always print skip statistics - important to know why things failed
    print(f"\n{'='*60}")
    print("SKIP STATISTICS:")
    print(f"  Images skipped (file not found): {debug_stats['images_skipped_no_file']}")
    print(f"  Images skipped (no valid patches): {debug_stats['images_skipped_no_valid_patches']}")
    print(f"  Patches skipped (no neighbors): {debug_stats['patches_skipped_no_neighbors']}")
    print(f"  Patches skipped (no words): {debug_stats['patches_skipped_no_words']}")
    print(f"  Patches processed: {debug_stats['patches_processed']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()



