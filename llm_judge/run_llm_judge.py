#!/usr/bin/env python3
"""
Unified LLM Judge for all analysis types: Static NN, LogitLens, and Contextual NN.

This script consolidates run_single_model_with_viz.py, run_single_model_with_viz_logitlens.py,
and run_single_model_with_viz_contextual.py into a single unified interface.

Usage:
    python run_llm_judge.py --analysis-type nn --llm olmo-7b --vision-encoder vit-l-14-336 ...
    python run_llm_judge.py --analysis-type logitlens --llm olmo-7b --vision-encoder vit-l-14-336 ...
    python run_llm_judge.py --analysis-type contextual --llm olmo-7b --vision-encoder vit-l-14-336 ...
"""

import os
import sys
import io
import json
import argparse
import re
import time
import base64
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


# =============================================================================
# API CLIENT & LLM RESPONSE
# =============================================================================

def get_llm_response(client, image_with_bbox, cropped_image, prompt_text,
                     api_provider: str, api_model: str):
    """
    Call LLM with image(s) and prompt; return parsed JSON response.
    Works with both OpenAI and OpenRouter APIs.
    """
    def encode_image(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    
    main_url = encode_image(image_with_bbox)
    crop_url = encode_image(cropped_image) if cropped_image is not None else None

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


# =============================================================================
# PATH CONSTRUCTION
# =============================================================================

def construct_paths(args):
    """
    Construct checkpoint name, model name, and input JSON path based on arguments.
    Returns: (checkpoint_name, model_name, input_json_path)
    """
    # Determine checkpoint and model name
    if args.checkpoint_name is not None:
        checkpoint_name = args.checkpoint_name
        if args.model_name is not None:
            model_name = args.model_name
        else:
            # Extract model name from checkpoint name
            model_name = checkpoint_name
            for prefix in ["train_mlp-only_pixmo_cap_resize_", "train_mlp-only_pixmo_cap_", 
                          "train_mlp-only_pixmo_points_resize_", "train_mlp-only_pixmo_topbottom_",
                          "_step12000-unsharded"]:
                model_name = model_name.replace(prefix, "")
    elif args.vision_encoder == "qwen2-vl":
        # Qwen2-VL off-the-shelf model
        checkpoint_name = "qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"
        model_name = "qwen2vl"
    elif args.llm == "qwen2-7b" and args.vision_encoder == "vit-l-14-336":
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}_seed10"
        model_name = f"{args.llm}_{args.vision_encoder}_seed10"
    else:
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}"
        model_name = f"{args.llm}_{args.vision_encoder}"
    
    base_dir = Path(args.base_dir)
    
    # Normalize layer format
    layer_num = str(args.layer).replace('contextual', '').replace('layer', '')
    
    # Try multiple path patterns based on analysis type
    input_json = None
    
    if args.analysis_type == "nn":
        candidates = [
            base_dir / f"{checkpoint_name}_step12000-unsharded" / f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{layer_num}.json",
            base_dir / f"{checkpoint_name}_step12000-unsharded" / "nearest_neighbors_analysis_pixmo_cap_multi-gpu.json",
            base_dir / f"{checkpoint_name}_step12000-unsharded" / "nearest_neighbors_analysis_pixmo_cap.json",
            base_dir / checkpoint_name / f"nearest_neighbors_layer{layer_num}_topk5.json",  # Qwen2-VL format
        ]
    elif args.analysis_type == "logitlens":
        candidates = [
            base_dir / checkpoint_name / f"logit_lens_layer{layer_num}_topk5.json",  # Qwen2-VL format
            base_dir / f"{checkpoint_name}_step12000-unsharded" / f"logit_lens_layer{layer_num}_topk5_multi-gpu.json",
        ]
    else:  # contextual
        # For contextual, we need to find allLayers files
        ckpt_dir_candidates = [
            base_dir / f"{checkpoint_name}_step12000-unsharded",
            base_dir / checkpoint_name,
        ]
        for ckpt_dir in ckpt_dir_candidates:
            if ckpt_dir.exists():
                # Look for allLayers files
                allLayers_files = sorted(ckpt_dir.glob("contextual_neighbors_visual*_allLayers.json"))
                if not allLayers_files:
                    allLayers_files = sorted(ckpt_dir.glob("contextual_neighbors_visual*_allLayers_multi-gpu.json"))
                if allLayers_files:
                    input_json = allLayers_files[0]
                    break
                # Fallback to layer-specific files
                for cand in sorted(ckpt_dir.glob("contextual_neighbors_visual*_contextual*_multi-gpu.json")):
                    match = re.search(r'_contextual(\d+)_multi-gpu\.json$', str(cand))
                    if match and int(match.group(1)) == int(layer_num):
                        input_json = cand
                        break
                if input_json:
                    break
        candidates = [input_json] if input_json else []
    
    # Find first existing file
    if not input_json:
        for cand in candidates:
            if cand and cand.exists():
                input_json = cand
                break
    
    if not input_json or not input_json.exists():
        print(f"ERROR: Input JSON not found. Tried:")
        for cand in candidates:
            if cand:
                print(f"  - {cand}")
        sys.exit(1)
    
    return checkpoint_name, model_name, input_json


# =============================================================================
# WORD/TOKEN EXTRACTION
# =============================================================================

def extract_full_word_from_token(sentence: str, token: str) -> str:
    """
    Extract the full word containing the token from the sentence.
    ONLY USED FOR CONTEXTUAL NN - expands subword tokens to full words.
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


def extract_tokens(patch_data, analysis_type: str, num_tokens: int = 5):
    """
    Extract tokens/words from patch data based on analysis type.
    
    Returns: list of token strings
    """
    if analysis_type == "nn":
        # Static NN: nearest_neighbors list
        neighbors = patch_data.get("nearest_neighbors", patch_data.get("top_neighbors", []))
        return [n['token'] for n in neighbors[:num_tokens]]
    
    elif analysis_type == "logitlens":
        # LogitLens: top_predictions list
        predictions = patch_data.get("top_predictions", [])
        return [p['token'] for p in predictions[:num_tokens]]
    
    else:  # contextual
        # Contextual NN: expand subwords to full words using sentence context
        neighbors = patch_data.get("nearest_contextual_neighbors", [])
        words = []
        for entry in neighbors[:num_tokens]:
            token = entry.get('token_str', '')
            caption = entry.get('caption', '')
            word = extract_full_word_from_token(caption, token)
            if word:
                words.append(word)
        return words


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(output_path, image_with_bbox, tokens, gpt_response, 
                        patch_row, patch_col, bbox_size, cropped_image=None,
                        ground_truth_caption="", analysis_type="nn"):
    """
    Create comprehensive visualization for patch evaluation.
    Unified across all analysis types.
    """
    img_width, img_height = image_with_bbox.size
    
    margin = 20
    text_width = 700
    total_width = img_width + text_width + 3 * margin
    total_height = max(img_height + 2 * margin, 900)
    
    vis_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(vis_img)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = subtitle_font = body_font = small_font = ImageFont.load_default()
    
    def safe_wrap(text, font, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
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
    
    # Paste image
    vis_img.paste(image_with_bbox, (margin, margin))
    
    # Text position
    text_x = img_width + 2 * margin
    text_y = margin
    
    # Title based on analysis type
    titles = {
        "nn": "Static NN Interpretability",
        "logitlens": "LogitLens Interpretability",
        "contextual": "Contextual Word Interpretability"
    }
    draw.text((text_x, text_y), titles.get(analysis_type, "Patch Interpretability"), fill='black', font=title_font)
    text_y += 40
    
    # Patch info
    patch_info = f"Patch: ({patch_row}, {patch_col}) - {bbox_size}x{bbox_size} bbox"
    draw.text((text_x, text_y), patch_info, fill='gray', font=small_font)
    text_y += 30
    
    # Show cropped region if available
    if cropped_image is not None:
        crop_width = min(200, text_width - 20)
        crop_height = min(150, crop_width * cropped_image.height // cropped_image.width)
        cropped_resized = cropped_image.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
        crop_x = text_x + (text_width - crop_width) // 2
        vis_img.paste(cropped_resized, (crop_x, text_y))
        text_y += crop_height + 15
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
    
    # Candidate tokens/words
    label = "Candidate Words:" if analysis_type == "contextual" else "Candidate Tokens:"
    draw.text((text_x, text_y), label, fill='black', font=subtitle_font)
    text_y += 25
    tokens_str = ", ".join([f'"{t}"' for t in tokens])
    token_lines = safe_wrap(tokens_str, body_font, text_width)
    text_y = draw_text_block(text_y, token_lines, body_font, 'blue')
    text_y += 15
    
    # Word categories
    if concrete_words:
        draw.text((text_x, text_y), "Concrete Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in concrete_words])
        word_lines = safe_wrap(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'green')
        text_y += 15
    
    if abstract_words:
        draw.text((text_x, text_y), "Abstract Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in abstract_words])
        word_lines = safe_wrap(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'blue')
        text_y += 15
    
    if global_words:
        draw.text((text_x, text_y), "Global Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in global_words])
        word_lines = safe_wrap(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'orange')
        text_y += 15
    
    # Reasoning
    reasoning = gpt_response.get('reasoning', '')
    if reasoning:
        draw.text((text_x, text_y), "LLM Reasoning:", fill='black', font=subtitle_font)
        text_y += 25
        reasoning_lines = safe_wrap(reasoning, body_font, text_width)
        text_y = draw_text_block(text_y, reasoning_lines, body_font, 'black')
        text_y += 15
    
    # Ground truth (if available)
    if ground_truth_caption and text_y < total_height - 100:
        draw.text((text_x, text_y), "Ground Truth Caption:", fill='black', font=subtitle_font)
        text_y += 25
        caption_lines = safe_wrap(ground_truth_caption, small_font, text_width)
        text_y = draw_text_block(text_y, caption_lines[:5], small_font, 'gray', spacing=3)
    
    vis_img.save(output_path, quality=95)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_images_data(input_json, analysis_type: str, num_images: int = None):
    """
    Load image data from JSON, handling different formats.
    For contextual, uses streaming parser to avoid memory issues.
    
    Yields: (image_idx, image_data_dict) tuples
    """
    if analysis_type == "contextual":
        # Use streaming parser for large contextual files
        import ijson
        with open(input_json, 'rb') as f:
            parser = ijson.items(f, 'results.item')
            for i, image_obj in enumerate(parser):
                if num_images is not None and i >= num_images:
                    break
                yield i, image_obj
    else:
        # Direct load for smaller files
        with open(input_json, 'r') as f:
            data = json.load(f)
        
        # Handle both Molmo format (splits.validation.images) and Qwen2-VL format (results)
        if "splits" in data:
            images = data.get("splits", {}).get("validation", {}).get("images", [])
        elif "results" in data:
            images = data.get("results", [])
        else:
            print(f"ERROR: Unknown JSON format. Keys: {list(data.keys())}")
            sys.exit(1)
        
        for i, image_data in enumerate(images):
            if num_images is not None and i >= num_images:
                break
            yield i, image_data


def get_patches_from_image(image_data, analysis_type: str):
    """
    Extract patches from image data, handling different formats.
    Returns: (patches_list, grid_size)
    """
    patches = []
    max_row, max_col = 0, 0
    
    # Handle both formats: chunks (Molmo) and patches directly (Qwen2-VL)
    if 'chunks' in image_data and image_data['chunks']:
        for chunk in image_data['chunks']:
            for patch in chunk.get("patches", []):
                row = patch.get("patch_row", 0)
                col = patch.get("patch_col", 0)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                patches.append(patch)
    elif 'patches' in image_data:
        for patch in image_data['patches']:
            row = patch.get("patch_row", 0)
            col = patch.get("patch_col", 0)
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            patches.append(patch)
    
    grid_size = max(max_row + 1, max_col + 1) if (max_row > 0 or max_col > 0) else 24
    return patches, grid_size


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified LLM Judge for Static NN, LogitLens, and Contextual NN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_llm_judge.py --analysis-type nn --llm olmo-7b --vision-encoder vit-l-14-336 --layer 0
  python run_llm_judge.py --analysis-type logitlens --llm qwen2-7b --vision-encoder qwen2-vl --layer 16
  python run_llm_judge.py --analysis-type contextual --llm olmo-7b --vision-encoder vit-l-14-336 --layer 16
        """
    )
    
    # Required arguments
    parser.add_argument('--analysis-type', type=str, required=True, 
                        choices=['nn', 'logitlens', 'contextual'],
                        help='Type of analysis to evaluate')
    parser.add_argument('--llm', type=str, required=True, help='LLM name (e.g., olmo-7b)')
    parser.add_argument('--vision-encoder', type=str, required=True, help='Vision encoder name (e.g., vit-l-14-336)')
    
    # API configuration
    parser.add_argument('--api-key-file', type=str, default="llm_judge/api_key.txt", 
                        help='Path to API key file (default: llm_judge/api_key.txt)')
    parser.add_argument('--api-provider', type=str, default='openai', choices=['openai', 'openrouter'], 
                        help='API provider: openai or openrouter')
    parser.add_argument('--api-model', type=str, default='gpt-5', 
                        help='Model to use (e.g., gpt-5 for OpenAI, google/gemini-2.0-flash-exp for OpenRouter)')
    
    # Path configuration
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory with analysis results (auto-detected based on analysis-type if not specified)')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Output base directory (auto-detected based on analysis-type if not specified)')
    parser.add_argument('--checkpoint-name', type=str, default=None, 
                        help='Override checkpoint name (for ablations, etc.)')
    parser.add_argument('--model-name', type=str, default=None, 
                        help='Override model name for output directory')
    
    # Processing configuration
    parser.add_argument('--layer', type=str, default='0', 
                        help='Layer to evaluate (e.g., 0, 16, layer0, contextual16)')
    parser.add_argument('--num-images', type=int, default=300, help='Number of images to process')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of patches per image')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'validation'])
    parser.add_argument('--use-cropped-region', action='store_true', 
                        help='Pass cropped region as second image to LLM')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Control flags
    parser.add_argument('--skip-if-complete', action='store_true', help='Skip if output already exists and is complete')
    parser.add_argument('--resume', action='store_true', help='Resume incomplete runs')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if output exists')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Auto-detect base-dir and output-base if not specified
    default_dirs = {
        'nn': ('analysis_results/nearest_neighbors', 'analysis_results/llm_judge_nearest_neighbors'),
        'logitlens': ('analysis_results/logit_lens', 'analysis_results/llm_judge_logitlens'),
        'contextual': ('analysis_results/contextual_nearest_neighbors', 'analysis_results/llm_judge_contextual_nn'),
    }
    if args.base_dir is None:
        args.base_dir = default_dirs[args.analysis_type][0]
    if args.output_base is None:
        args.output_base = default_dirs[args.analysis_type][1]
    
    # Construct paths
    checkpoint_name, model_name, input_json = construct_paths(args)
    
    # Normalize layer for output naming
    layer_num = str(args.layer).replace('contextual', '').replace('layer', '')
    
    # Create output directory
    model_suffix = args.api_model
    if "/" in model_suffix:
        model_suffix = model_suffix.split("/")[-1]
    model_suffix = model_suffix.replace("-", "").replace(".", "").replace(":", "_").lower()
    model_suffix = model_suffix.replace("_free", "").replace("free", "")
    
    layer_prefix = "contextual" if args.analysis_type == "contextual" else "layer"
    output_dir_name = f"llm_judge_{model_name}_{layer_prefix}{layer_num}_{model_suffix}"
    if args.use_cropped_region:
        output_dir_name += "_cropped"
    output_dir = Path(args.output_base) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    output_json = output_dir / f"results_{args.split}.json"
    
    # Check for existing output
    if output_json.exists() and not args.force:
        with open(output_json, 'r') as f:
            existing = json.load(f)
        existing_total = existing.get('total', len(existing.get('results', [])))
        if args.skip_if_complete and existing_total >= args.num_images:
            print(f"SKIP: Already complete ({existing_total}/{args.num_images})")
            return
        if args.resume:
            print(f"Resuming from {existing_total} existing examples...")
    
    print(f"\n{'='*60}")
    print(f"LLM Judge - {args.analysis_type.upper()}")
    print(f"Model: {args.llm} + {args.vision_encoder}")
    print(f"Layer: {layer_num}")
    print(f"Input: {input_json}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load API key
    with open(args.api_key_file, 'r') as f:
        api_key = f.read().strip()
    
    # Initialize API client
    if args.api_provider == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)
    
    # Load dataset
    dataset = PixMoCap(split=args.split, mode="captions")
    
    # Process images
    results = []
    accuracy = 0
    total = 0
    
    for list_idx, image_data in tqdm(load_images_data(input_json, args.analysis_type, args.num_images), 
                                      desc="Processing images", total=args.num_images):
        image_idx = image_data.get("image_idx", list_idx)
        
        # Get image path
        example = dataset.get(image_idx, np.random)
        image_path = example["image"]
        
        if not os.path.exists(image_path):
            if args.debug:
                print(f"SKIP Image {image_idx}: File not found")
            continue
        
        # Process image
        processed_image, image_mask = process_image_with_mask(image_path, model_name=model_name)
        
        # Get patches
        patches, grid_size = get_patches_from_image(image_data, args.analysis_type)
        
        # Build patch map for lookup
        patch_map = {}
        for patch in patches:
            key = (patch.get("patch_row", 0), patch.get("patch_col", 0))
            patch_map[key] = patch
        
        # Sample positions
        sampled_positions = sample_valid_patch_positions(image_mask, bbox_size=3, num_samples=args.num_samples, grid_size=grid_size)
        if not sampled_positions:
            print(f"SKIP Image {image_idx}: No valid patch positions (black padding or invalid mask)")
            continue
        
        ground_truth_caption = image_data.get("ground_truth_caption", "")
        
        for patch_row, patch_col in sampled_positions:
            bbox_size = 3
            actual_patch_size = 512 / grid_size
            bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=bbox_size)
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            
            # Get center patch
            center_row = patch_row + bbox_size // 2
            center_col = patch_col + bbox_size // 2
            
            patch_data = patch_map.get((center_row, center_col))
            if not patch_data:
                if args.debug:
                    print(f"  SKIP Image {image_idx} patch ({patch_row},{patch_col}) center ({center_row},{center_col}): No neighbor data in patch_map")
                continue

            # Extract tokens/words
            tokens = extract_tokens(patch_data, args.analysis_type)
            if not tokens:
                if args.debug:
                    print(f"  SKIP Image {image_idx} patch ({patch_row},{patch_col}) center ({center_row},{center_col}): No tokens extracted")
                continue
            
            # Prepare cropped image if needed
            cropped_image = None
            if args.use_cropped_region:
                left = int(patch_col * actual_patch_size)
                top = int(patch_row * actual_patch_size)
                right = int((patch_col + bbox_size) * actual_patch_size)
                bottom = int((patch_row + bbox_size) * actual_patch_size)
                left = max(0, left); top = max(0, top)
                right = min(right, processed_image.size[0])
                bottom = min(bottom, processed_image.size[1])
                cropped_image = processed_image.crop((left, top, right, bottom))
            
            # Get prompt
            if args.use_cropped_region:
                prompt = IMAGE_PROMPT_WITH_CROP.format(candidate_words=str(tokens))
            else:
                prompt = IMAGE_PROMPT.format(candidate_words=str(tokens))
            
            # Get LLM response
            response = get_llm_response(client, image_with_bbox, cropped_image, prompt,
                                        args.api_provider, args.api_model)
            
            # Check interpretability
            concrete_words = response.get('concrete_words', [])
            abstract_words = response.get('abstract_words', [])
            global_words = response.get('global_words', [])
            is_interpretable = len(concrete_words) > 0 or len(abstract_words) > 0 or len(global_words) > 0
            
            # Save visualization
            status_str = 'pass' if is_interpretable else 'fail'
            viz_filename = f"image_{image_idx:04d}_patch_{patch_row}_{patch_col}_{status_str}.jpg"
            create_visualization(
                viz_dir / viz_filename, image_with_bbox, tokens, response,
                patch_row, patch_col, bbox_size, cropped_image,
                ground_truth_caption, args.analysis_type
            )
            
            # Accumulate result
            result = {
                'image_idx': image_idx,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'tokens': tokens,
                'gpt_response': response,
                'interpretable': is_interpretable,
                'original_image_path': image_path,
            }
            results.append(result)
            
            if is_interpretable:
                accuracy += 1
            total += 1
            
            print(f"  Patch ({patch_row}, {patch_col}): {tokens[:3]}... -> {'PASS' if is_interpretable else 'FAIL'}")
            
            # Incremental save
            save_data = {
                'analysis_type': args.analysis_type,
                'llm': args.llm,
                'vision_encoder': args.vision_encoder,
                'layer': layer_num,
                'split': args.split,
                'total': total,
                'correct': accuracy,
                'accuracy': (accuracy / total * 100) if total > 0 else 0,
                'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                'results': results,
            }
            with open(output_json, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
    
    # Final summary
    final_accuracy = (accuracy / total * 100) if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Results: {accuracy}/{total} ({final_accuracy:.1f}%)")
    print(f"Saved to: {output_json}")
    print(f"Visualizations: {viz_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
