#!/usr/bin/env python3
"""
Run LLM judge evaluation for a single model combination using logit lens results with visualizations.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import time

# Import existing utilities
from prompts import IMAGE_PROMPT, IMAGE_PROMPT_WITH_CROP
from utils import (
    process_image_with_mask,
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    sample_valid_patch_positions
)
from olmo.data.pixmo_datasets import PixMoCap


def get_gpt_response(client, image, cropped_image, prompt, api_provider="openai", model="gpt-5"):
    """Get LLM response for image interpretability analysis."""
    import base64
    import io
    
    # Encode main image
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Add cropped image if provided
    crop_str = None
    if cropped_image is not None:
        buffered_crop = io.BytesIO()
        cropped_image.save(buffered_crop, format="PNG")
        crop_str = base64.b64encode(buffered_crop.getvalue()).decode("utf-8")
    
    if api_provider == "openrouter":
        # OpenRouter uses OpenAI-compatible format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add cropped image if provided
        if crop_str is not None:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{crop_str}"}
            })
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        response_text = response.choices[0].message.content
    else:
        # OpenAI GPT-5 format
        content = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_str}"},
            {"type": "input_text", "text": prompt},
        ]
        
        if crop_str is not None:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{crop_str}"})
        
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": content,
            }],
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        )
        
        response_text = response.output_text
    
    # Try to extract JSON
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
        "words": [],
        "reasoning": f"Could not parse response as JSON. Response: {response_text[:200]}"
    }


def crop_image_region(processed_image, patch_row, patch_col, bbox_size):
    """Crop the image region based on patch coordinates and bbox size."""
    from PIL import Image
    
    img_width, img_height = processed_image.size
    
    # Use the same patch size calculation as the bounding box
    actual_patch_size = img_width / 24  # Same as in create_visualization
    
    # Calculate bounding box coordinates using the same method as calculate_square_bbox_from_patch
    left = int(patch_col * actual_patch_size)
    top = int(patch_row * actual_patch_size)
    right = int((patch_col + bbox_size) * actual_patch_size)
    bottom = int((patch_row + bbox_size) * actual_patch_size)
    
    # Ensure coordinates are within image bounds
    left = max(0, left)
    top = max(0, top)
    right = min(right, img_width)
    bottom = min(bottom, img_height)
    
    # Crop the region
    cropped = processed_image.crop((left, top, right, bottom))
    return cropped


def create_visualization(image_path, patch_row, patch_col, bbox_size, tokens, gpt_response, output_path, ground_truth_caption="", cropped_image=None, grid_size=24):
    """Create comprehensive visualization for patch evaluation."""
    
    # Process image and create bbox
    processed_image, _ = process_image_with_mask(image_path)
    actual_patch_size = 512 / grid_size  # Dynamic grid size (Molmo=24, Qwen2-VL=16)
    bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=bbox_size)
    
    image_with_bbox = draw_bbox_on_image(processed_image, bbox)
    
    img_width, img_height = image_with_bbox.size
    
    # Create canvas with space for text
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
    
    # Helper functions
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
    
    def draw_text_block(y, lines, font, color):
        spacing = 5
        for line in lines:
            try:
                bbox = draw.textbbox((text_x, y), line, font=font)
                draw.text((text_x, y), line, fill=color, font=font)
                y += (bbox[3] - bbox[1]) + spacing
            except:
                y += 18 + spacing
        return y
    
    # Paste image
    vis_img.paste(image_with_bbox, (margin, margin))
    
    # Text position
    text_x = img_width + 2 * margin
    text_y = margin
    
    # Title
    draw.text((text_x, text_y), "Patch Interpretability", fill='black', font=title_font)
    text_y += 40
    
    # Patch info
    patch_info = f"Patch: ({patch_row}, {patch_col}) - {bbox_size}x{bbox_size} bbox"
    draw.text((text_x, text_y), patch_info, fill='gray', font=small_font)
    text_y += 30
    
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
    
    # Candidate tokens
    draw.text((text_x, text_y), "Candidate Tokens:", fill='black', font=subtitle_font)
    text_y += 25
    tokens_str = ", ".join([f'"{t}"' for t in tokens])
    token_lines = safe_wrap(tokens_str, body_font, text_width)
    text_y = draw_text_block(text_y, token_lines, body_font, 'blue')
    text_y += 15
    
    # Concrete words
    if concrete_words:
        draw.text((text_x, text_y), "Concrete Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in concrete_words])
        word_lines = safe_wrap(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'green')
        text_y += 15
    
    # Abstract words
    if abstract_words:
        draw.text((text_x, text_y), "Abstract Words:", fill='black', font=subtitle_font)
        text_y += 25
        words_str = ", ".join([f'"{w}"' for w in abstract_words])
        word_lines = safe_wrap(words_str, body_font, text_width)
        text_y = draw_text_block(text_y, word_lines, body_font, 'blue')
        text_y += 15
    
    # Global words
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
        draw.text((text_x, text_y), "GPT Reasoning:", fill='black', font=subtitle_font)
        text_y += 25
        reasoning_lines = safe_wrap(reasoning, body_font, text_width)
        text_y = draw_text_block(text_y, reasoning_lines, body_font, 'black')
        text_y += 15
    
    # Ground truth caption
    if ground_truth_caption:
        draw.text((text_x, text_y), "Ground Truth Caption:", fill='black', font=subtitle_font)
        text_y += 25
        caption_lines = safe_wrap(ground_truth_caption, small_font, text_width)
        text_y = draw_text_block(text_y, caption_lines, small_font, 'gray')
    
    # Save visualization
    vis_img.save(output_path, "JPEG", quality=95)


def get_high_confidence_tokens_from_logitlens(logitlens_data, patch_start_row, patch_start_col, size=3, num_tokens=5):
    """
    Extract top tokens from logit lens data within a specified patch area.
    
    Args:
        logitlens_data (list): List of logit lens results
        patch_start_row (int): Starting row of the patch area
        patch_start_col (int): Starting column of the patch area 
        size (int): Size of the square area (e.g., 3 for 3x3) (default: 3)
        num_tokens (int): Number of tokens to return (default: 5)
        
    Returns:
        list: List of token strings
    """
    target_row = patch_start_row + size // 2
    target_col = patch_start_col + size // 2

    for result in logitlens_data:
        if result['patch_row'] == target_row and result['patch_col'] == target_col:
            # Extract tokens from top_predictions
            tokens = [pred['token'] for pred in result['top_predictions'][:num_tokens]]
            return tokens

    return []


def find_existing_output(output_base, model_name, layer, split, use_cropped_region):
    """Find existing output directory regardless of model suffix."""
    output_base = Path(output_base)
    
    # Pattern: llm_judge_{model_name}_{layer}_*_cropped or llm_judge_{model_name}_{layer}_cropped
    suffix = "_cropped" if use_cropped_region else ""
    
    # Try to find existing directories
    pattern_prefix = f"llm_judge_{model_name}_{layer}"
    for dir_path in output_base.glob(f"{pattern_prefix}*{suffix}"):
        if dir_path.is_dir():
            results_file = dir_path / f"results_{split}.json"
            if results_file.exists():
                return dir_path, results_file
    
    return None, None


def check_output_completeness(output_json, expected_total):
    """Check if output exists and has the expected number of examples."""
    if not output_json.exists():
        return False, 0, "Output file does not exist"
    
    with open(output_json, 'r') as f:
        data = json.load(f)
    
    total = data.get('total', 0)
    
    if total >= expected_total:
        return True, total, f"Complete: {total}/{expected_total} examples"
    else:
        return False, total, f"Incomplete: {total}/{expected_total} examples"


def main():
    parser = argparse.ArgumentParser(description='Run LLM judge for one model with visualizations using logit lens results')
    
    parser.add_argument('--llm', type=str, required=True, help='LLM name (e.g., olmo-7b)')
    parser.add_argument('--vision-encoder', type=str, required=True, help='Vision encoder name (e.g., vit-l-14-336)')
    parser.add_argument('--api-key', type=str, required=True, help='API key (OpenAI or OpenRouter)')
    parser.add_argument('--api-provider', type=str, default='openai', choices=['openai', 'openrouter'], 
                        help='API provider: openai or openrouter')
    parser.add_argument('--api-model', type=str, default='gpt-5', 
                        help='Model to use (e.g., gpt-5 for OpenAI, google/gemini-2.0-flash-exp:free or google/gemini-2.0-flash-exp for OpenRouter)')
    parser.add_argument('--base-dir', type=str, default='analysis_results/logit_lens', help='Base directory with logit lens results')
    parser.add_argument('--output-base', type=str, default='analysis_results/llm_judge_logitlens', help='Output base directory')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of patches per image')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'validation'])
    parser.add_argument('--use-cropped-region', action='store_true', help='Use cropped region prompt (passes both full image and cropped region to LLM)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--layer', type=str, default='layer0', help='Layer to analyze (e.g., layer0, layer1)')
    parser.add_argument('--skip-if-complete', action='store_true', help='Skip if output already exists and is complete')
    parser.add_argument('--skip-if-exists', action='store_true', help='Skip if any output exists (complete or incomplete)')
    parser.add_argument('--resume', action='store_true', help='Resume incomplete runs instead of regenerating from scratch')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if output exists (overrides skip flags)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Construct paths
    if args.vision_encoder == "qwen2-vl":
        # Qwen2-VL off-the-shelf model uses different path structure
        checkpoint_name = "qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"
        model_name = "qwen2vl"
    elif args.llm == "qwen2-7b" and args.vision_encoder == "vit-l-14-336":
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}_seed10"
        model_name = f"{args.llm}_{args.vision_encoder}_seed10"
    else:
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}"
        model_name = f"{args.llm}_{args.vision_encoder}"
    
    base_dir = Path(args.base_dir)
    
    # Try Qwen2-VL format first: {base_dir}/{checkpoint_name}/logit_lens_layer{X}_topk5.json
    input_json = base_dir / checkpoint_name / f"logit_lens_layer{args.layer}_topk5.json"
    
    # Fallback to Molmo format
    if not input_json.exists():
        input_json = base_dir / f"{checkpoint_name}_step12000-unsharded" / f"logit_lens_{args.layer}_topk5_multi-gpu.json"
    
    if not input_json.exists():
        print(f"ERROR: Input JSON not found: {input_json}")
        sys.exit(1)
    
    # Create output directory with API model suffix to distinguish different judges
    # Extract a short model identifier (e.g., "gpt5" from "gpt-5", "gemini25pro" from "google/gemini-2.5-pro")
    model_suffix = args.api_model
    # Remove common prefixes and clean up
    if "/" in model_suffix:
        model_suffix = model_suffix.split("/")[-1]  # Get everything after the last /
    # Remove special characters, make lowercase, and remove "free" suffix
    model_suffix = model_suffix.replace("-", "").replace(".", "").replace(":", "_").lower()
    model_suffix = model_suffix.replace("_free", "").replace("free", "")  # Remove free tier indicator
    
    output_dir_name = f"llm_judge_{model_name}_{args.layer}_{model_suffix}"
    if args.use_cropped_region:
        output_dir_name += "_cropped"
    output_dir = Path(args.output_base) / output_dir_name
    output_json = output_dir / f"results_{args.split}.json"
    
    # Check if we should skip this run BEFORE creating any directories or writing anything
    if (args.skip_if_complete or args.skip_if_exists) and not args.force:
        existing_dir, existing_json = find_existing_output(
            args.output_base, model_name, args.layer, args.split, args.use_cropped_region
        )
        
        if existing_json is not None:
            is_complete, actual_count, message = check_output_completeness(existing_json, args.num_images)
            
            # Skip if exists (regardless of completeness) when --skip-if-exists is set
            if args.skip_if_exists:
                print("=" * 60)
                print(f"SKIPPING: {args.llm} + {args.vision_encoder} {args.layer}")
                print(f"Reason: Output exists ({actual_count}/{args.num_images} examples)")
                print(f"Existing output: {existing_dir}")
                print(f"Use --force to regenerate")
                print("=" * 60)
                return
            
            # Only skip complete runs when --skip-if-complete is set
            if args.skip_if_complete and is_complete:
                print("=" * 60)
                print(f"SKIPPING: {args.llm} + {args.vision_encoder} {args.layer}")
                print(f"Reason: {message}")
                print(f"Existing output: {existing_dir}")
                print("=" * 60)
                return
            elif args.skip_if_complete and not is_complete:
                print("=" * 60)
                print(f"REGENERATING: {args.llm} + {args.vision_encoder} {args.layer}")
                print(f"Reason: {message}")
                print(f"Will overwrite: {existing_dir}")
                print(f"New output: {output_dir}")
                print("=" * 60)
    
    # NOW create directories (after skip check)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(f"Model: {args.llm} + {args.vision_encoder}")
    print(f"Layer: {args.layer}")
    print(f"Input: {input_json}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load logit lens data
    with open(input_json, 'r') as f:
        logitlens_data = json.load(f)
    
    # Initialize API client
    if args.api_provider == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=args.api_key
        )
    else:
        client = OpenAI(api_key=args.api_key)
    
    # Load dataset for image paths and captions
    dataset = PixMoCap(split=args.split, mode="captions")
    
    # Check if we should resume from existing progress
    existing_responses = {}
    if args.resume and output_json.exists():
        print("Resuming from existing progress...")
        with open(output_json, 'r') as f:
            existing_data = json.load(f)
        existing_responses = existing_data.get('responses', {})
        accuracy = existing_data.get('correct', 0)
        total = existing_data.get('total', 0)
        gpt_responses = defaultdict(list, existing_responses)
        print(f"Found {total} existing examples, resuming from there...")
    else:
        gpt_responses = defaultdict(list)
        accuracy = 0
        total = 0
    
    # Sample images to process
    num_images_to_process = min(args.num_images, len(dataset))
    
    # Determine which images to process
    if args.resume and existing_responses:
        # Find already processed images
        processed_indices = set()
        for image_path, patches in existing_responses.items():
            for patch in patches:
                processed_indices.add(patch['image_index'])
        
        # Only process remaining images
        all_indices = set(range(num_images_to_process))
        remaining_indices = sorted(all_indices - processed_indices)
        image_indices = remaining_indices
        print(f"Already processed {len(processed_indices)} images, {len(remaining_indices)} remaining")
    else:
        image_indices = list(range(num_images_to_process))
    
    for idx in tqdm(image_indices, desc="Processing images"):
        image_path = dataset[idx]['image']
        
        # Get logit lens data for this image
        image_logitlens = logitlens_data['results'][idx]
        
        # Sample patches for this image - handle both chunks (Molmo) and patches (Qwen2-VL) formats
        if 'chunks' in image_logitlens and image_logitlens['chunks']:
            patches = image_logitlens['chunks'][0]['patches']  # Use "Full Image" chunk
        elif 'patches' in image_logitlens:
            patches = image_logitlens['patches']  # Qwen2-VL format
        else:
            print(f"SKIP Image {idx}: No patches found")
            continue
        
        # Calculate grid size from actual patch data
        max_row = max(p.get('patch_row', 0) for p in patches) if patches else 23
        max_col = max(p.get('patch_col', 0) for p in patches) if patches else 23
        grid_size = max(max_row + 1, max_col + 1)
        
        # Sample valid patch positions
        processed_image, image_mask = process_image_with_mask(image_path)
        valid_positions = sample_valid_patch_positions(image_mask, bbox_size=3, num_samples=args.num_samples)
        
        for patch_row, patch_col in valid_positions:
            bbox_size = 3
            actual_patch_size = 512 / grid_size  # Dynamic grid size (Molmo=24, Qwen2-VL=16)
            bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=bbox_size)
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            
            # Get center patch tokens from logit lens
            center_row = patch_row + bbox_size // 2
            center_col = patch_col + bbox_size // 2
            
            tokens = get_high_confidence_tokens_from_logitlens(patches, center_row, center_col, size=bbox_size, num_tokens=5)
            if not tokens:
                continue
            
            # Get GPT response
            if args.use_cropped_region:
                # Use cropped region prompt
                cropped_image = crop_image_region(processed_image, patch_row, patch_col, bbox_size)
                formatted_prompt = IMAGE_PROMPT_WITH_CROP.format(candidate_words=str(tokens))
                response = get_gpt_response(client, image_with_bbox, cropped_image, formatted_prompt, 
                                           api_provider=args.api_provider, model=args.api_model)
            else:
                # Use regular prompt
                formatted_prompt = IMAGE_PROMPT.format(candidate_words=str(tokens))
                response = get_gpt_response(client, image_with_bbox, None, formatted_prompt,
                                           api_provider=args.api_provider, model=args.api_model)
            
            # Save result
            result = {
                'patch_row': patch_row,
                'patch_col': patch_col,
                'bbox_size': bbox_size,
                'tokens_used': tokens,
                'gpt_response': response,
                'original_image_path': image_path,
                'image_index': idx
            }
            gpt_responses[image_path].append(result)
            
            # Create visualization
            concrete_words = response.get('concrete_words', [])
            abstract_words = response.get('abstract_words', [])
            global_words = response.get('global_words', [])
            is_interpretable = len(concrete_words) > 0 or len(abstract_words) > 0 or len(global_words) > 0
            viz_filename = f"image_{idx:04d}_patch_{patch_row}_{patch_col}_{'pass' if is_interpretable else 'fail'}.jpg"
            viz_path = viz_dir / viz_filename
            
            # Get cropped image if using cropped region mode
            cropped_img = None
            if args.use_cropped_region:
                cropped_img = crop_image_region(processed_image, patch_row, patch_col, bbox_size)
            
            create_visualization(
                image_path=image_path,
                patch_row=patch_row,
                patch_col=patch_col,
                bbox_size=bbox_size,
                tokens=tokens,
                gpt_response=response,
                output_path=str(viz_path),
                ground_truth_caption="",
                cropped_image=cropped_img
            )
            
            # Check if interpretable (any type of related words)
            concrete_words = response.get('concrete_words', [])
            abstract_words = response.get('abstract_words', [])
            global_words = response.get('global_words', [])
            is_interpretable = len(concrete_words) > 0 or len(abstract_words) > 0 or len(global_words) > 0
            
            if is_interpretable:
                accuracy += 1
            total += 1
            
            print(f"  Patch ({patch_row}, {patch_col}): {tokens} -> {'PASS' if is_interpretable else 'FAIL'}")
            
            # Save intermediate results
            results_data = {
                'model': f"{args.llm}_{args.vision_encoder}",
                'layer': args.layer,
                'split': args.split,
                'total': total,
                'correct': accuracy,
                'accuracy': accuracy / total if total > 0 else 0,
                'last_updated': time.time(),
                'responses': dict(gpt_responses)
            }
            
            with open(output_json, 'w') as f:
                json.dump(results_data, f, indent=2)
    
    print("=" * 60)
    print(f"Results: {accuracy}/{total} ({accuracy/total*100:.1f}%)")
    print(f"Saved to: {output_json}")
    print(f"Visualizations: {viz_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
