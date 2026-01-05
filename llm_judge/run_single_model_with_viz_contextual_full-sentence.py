#!/usr/bin/env python3
"""
Run LLM judge evaluation for contextual nearest neighbors with visualizations.

This script consumes outputs produced by scripts like
  scripts/analysis/contextual_nearest_neighbors.py
which save JSON files containing, per image/patch, the top-k nearest contextual
neighbors (token in the context of a sentence).

It formats 5 sentence candidates by highlighting the key token inside each
sentence (using **word**), asks the LLM judge with the SENTENCE_LEVEL_PROMPT,
and saves both JSON and visualization outputs. Supports OpenAI and OpenRouter
(e.g., Gemini) via flags consistent with other LLM judge scripts.
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

from prompts import SENTENCE_LEVEL_PROMPT, SENTENCE_LEVEL_PROMPT_WITH_CROP
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


def get_llm_response_sentences(client, image_with_bbox, cropped_image, prompt_text,
                               api_provider: str, api_model: str) -> str:
    """
    Call LLM with SENTENCE_LEVEL_PROMPT text and images; return raw text response.
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
        return resp.choices[0].message.content
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
        return resp.output_text


def highlight_token_in_sentence(sentence: str, token: str) -> str:
    """
    Highlight the first occurrence of token in the sentence by surrounding it with ** **.
    If the token is a subword within a larger word (e.g., "ing" in "rendering"),
    expand to highlight the entire containing word. Case-insensitive match.
    If not found, fall back to appending (**token**).
    """
    if not sentence:
        return f"**{token}**"

    # Strip whitespace from token (may have trailing/leading spaces from tokenizer)
    token = token.strip() if token else ""
    if not token:
        return sentence

    low_sent = sentence.lower()
    low_tok = token.lower()
    if not low_tok:
        return sentence

    idx = low_sent.find(low_tok)
    if idx == -1:
        # No occurrence; keep sentence and append token for context
        return f"{sentence} (**{token}** )"

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
                return sentence[:exp_start] + f"**{expanded}**" + sentence[exp_end:]

    # Default: highlight only the matched token range (no expansion)
    # Use original sentence slice to preserve case
    matched = sentence[start:end]
    return sentence[:start] + f"**{matched}**" + sentence[end:]


def format_prompt_from_contextual(nearest_list):
    """
    Take top-5 nearest contextual neighbors entries and build the SENTENCE_LEVEL_PROMPT.
    Each entry has fields like token_str, caption, position, similarity.
    """
    # Build up to 5 sentences with highlighting
    sentences = []
    for i in range(min(5, len(nearest_list))):
        entry = nearest_list[i]
        token = entry.get('token_str', '')
        caption = entry.get('caption', '')
        sentences.append(highlight_token_in_sentence(caption, token))

    # Pad to exactly 5 for the prompt placeholders
    while len(sentences) < 5:
        sentences.append("")

    return SENTENCE_LEVEL_PROMPT.format(
        candidate_sentence_1=sentences[0],
        candidate_sentence_2=sentences[1],
        candidate_sentence_3=sentences[2],
        candidate_sentence_4=sentences[3],
        candidate_sentence_5=sentences[4],
    )


def parse_sentence_level_json(response_text: str):
    """
    Extract JSON array from response text and load. Returns list or None.
    """
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        payload = response_text[start:end+1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def safe_wrap(text, font, max_width):
    draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    words = (text or "").split()
    if not words:
        return []
    lines = []
    current = words[0]
    for w in words[1:]:
        if draw.textlength(current + ' ' + w, font=font) <= max_width:
            current += ' ' + w
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def create_visualization(output_path: Path, image_with_bbox, sentences, llm_results, cropped_image=None):
    """
    Draw image with bbox and a panel showing the 5 sentences and LLM decisions.
    """
    # Layout: left image, right text panel
    img_w, img_h = image_with_bbox.size
    panel_w = max(520, int(img_w * 0.9))
    out_w = img_w + panel_w

    # Fonts
    try:
        header_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Pre-compute required panel height to avoid bottom overflow
    def get_reasoning_text(item_dict):
        if not isinstance(item_dict, dict):
            return ""
        # Prefer 'reasoning' but fall back to common variants
        return (
            item_dict.get("reasoning")
            or item_dict.get("explanation")
            or item_dict.get("rationale")
            or item_dict.get("why")
            or ""
        )

    def compute_panel_height():
        temp_draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))
        x_margin = 16
        y = 16  # top margin
        y += 32  # header height
        
        # Account for cropped image if present
        if cropped_image is not None:
            max_crop_w = panel_w - 56
            max_crop_h = 180
            cw, ch = cropped_image.size
            if cw > 0 and ch > 0:
                scale = min(max_crop_w / cw, max_crop_h / ch, 1.0)
                crop_h = max(1, int(ch * scale))
                y += 18  # "Cropped region:" label
                y += crop_h + 12  # cropped image + spacing
        
        wrap_w = panel_w - 56
        for i, sent in enumerate(sentences[:5]):
            # index label + wrapped sentence
            lines = safe_wrap(sent, body_font, wrap_w)
            # at least one line even if empty, to keep spacing consistent
            line_count = max(1, len(lines))
            y_line = y + (20 * line_count)

            # Outcome block if present
            item = llm_results[i] if (llm_results and i < len(llm_results) and isinstance(llm_results[i], dict)) else None
            if item is not None:
                # status/id line
                y_line += 2
                y_line += 18
                # optional relation if present
                relation = item.get("relation")
                if relation:
                    y_line += 16
                # optional phrase lines (cap 3)
                phrase = item.get("interpretable_phrase")
                if phrase:
                    y_line += 16  # 'interpretable_phrase:' label
                    wrap_phrase = safe_wrap(phrase, small_font, wrap_w)
                    y_line += 16 * min(3, len(wrap_phrase))
                # reasoning lines (always show label; at least one line)
                reasoning_text = get_reasoning_text(item)
                y_line += 16  # 'reasoning:' label
                wrap_reason = safe_wrap(reasoning_text, small_font, wrap_w)
                # ensure at least one line if empty
                reason_lines = max(1, len(wrap_reason))
                y_line += 16 * min(4, reason_lines)

            # block spacing between items
            y = max(y_line + 14, y + 36)

        # Add bottom margin
        y += 16
        
        # Ensure at least as tall as the original image
        return max(img_h, y)

    out_h = max(520, compute_panel_height())

    # Create canvas with computed height
    canvas = Image.new('RGB', (out_w, out_h), 'white')
    canvas.paste(image_with_bbox, (0, 0))
    draw = ImageDraw.Draw(canvas)

    x0 = img_w + 16
    y = 16
    draw.text((x0, y), "Contextual NN Judgments", fill='black', font=header_font)
    y += 32

    # If a cropped region is provided, display a small thumbnail for reference
    if cropped_image is not None:
        # Fit crop within the panel width while keeping aspect ratio
        max_crop_w = panel_w - 56
        max_crop_h = 180
        cw, ch = cropped_image.size
        if cw > 0 and ch > 0:
            scale = min(max_crop_w / cw, max_crop_h / ch, 1.0)
            new_w = max(1, int(cw * scale))
            new_h = max(1, int(ch * scale))
            crop_resized = cropped_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            draw.text((x0, y), "Cropped region:", fill='black', font=small_font)
            y += 18
            # Paste onto canvas
            canvas.paste(crop_resized, (x0, y))
            y += new_h + 12

    # Show each sentence and outcome
    for i, sent in enumerate(sentences[:5]):
        draw.text((x0, y), f"{i+1}.", fill='black', font=body_font)
        wrap_w = panel_w - 56
        lines = safe_wrap(sent, body_font, wrap_w)
        if not lines:
            lines = [""]
        y_line = y
        for ln in lines:
            draw.text((x0 + 28, y_line), ln, fill='black', font=body_font)
            y_line += 20

        # Outcome
        if llm_results and i < len(llm_results) and isinstance(llm_results[i], dict):
            item = llm_results[i]
            status_is_interpretable = bool(item.get("interpretable"))
            status = "interpretable" if status_is_interpretable else "not interpretable"
            relation = item.get("relation")
            phrase = item.get("interpretable_phrase")
            item_id = item.get("id", i)
            y_line += 2
            # Status and id line
            draw.text((x0 + 28, y_line), f"→ id: {item_id} • interpretable: {str(status_is_interpretable)}", fill='darkgreen' if status_is_interpretable else 'red', font=small_font)
            y_line += 18
            # Optional fields shown only if present
            if relation:
                draw.text((x0 + 28, y_line), f"relation: {relation}", fill='black', font=small_font)
                y_line += 16
            if phrase:
                draw.text((x0 + 28, y_line), "interpretable_phrase:", fill='black', font=small_font)
                y_line += 16
                wrap_phrase = safe_wrap(phrase, small_font, wrap_w)
                for ln in wrap_phrase[:3]:
                    draw.text((x0 + 28, y_line), ln, fill='gray', font=small_font)
                    y_line += 16
            # Always show model reasoning (prompt specifies it is always present)
            reasoning_text = get_reasoning_text(item)
            draw.text((x0 + 28, y_line), "reasoning:", fill='black', font=small_font)
            y_line += 16
            wrap_reason = safe_wrap(reasoning_text, small_font, wrap_w)
            if wrap_reason:
                for ln in wrap_reason[:4]:  # cap lines to keep layout tidy
                    draw.text((x0 + 28, y_line), ln, fill='black', font=small_font)
                    y_line += 16
            else:
                draw.text((x0 + 28, y_line), "-", fill='gray', font=small_font)
                y_line += 16
        y = max(y_line + 14, y + 36)

    canvas.save(output_path, quality=95)


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

    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build checkpoint name from llm/encoder (matching other scripts)
    if args.llm == "qwen2-7b" and args.vision_encoder == "vit-l-14-336":
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}_seed10"
        model_name = f"{args.llm}_{args.vision_encoder}_seed10"
    else:
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{args.llm}_{args.vision_encoder}"
        model_name = f"{args.llm}_{args.vision_encoder}"

    # Input discovery: contextual_nearest_neighbors.py saves to
    # analysis_results/contextual_nearest_neighbors/<ckpt_name_step>/contextual_neighbors_visual{v}_contextual{c}_multi-gpu.json
    base_dir = Path(args.base_dir)
    ckpt_dir = base_dir / f"{checkpoint_name}_step12000-unsharded"
    # Try to find a file with "contextual_neighbors_visual" and matching contextual layer number from args.layer
    input_json = None
    target_layer = args.layer.replace('contextual', '')
    if ckpt_dir.exists():
        for cand in sorted(ckpt_dir.glob("contextual_neighbors_visual*_contextual*_multi-gpu.json")):
            # Extract the contextual layer number from filename using regex
            match = re.search(r'_contextual(\d+)_multi-gpu\.json$', str(cand))
            if match:
                file_layer = match.group(1)
                # Match exact layer number (handles contextual0 specially)
                if file_layer == target_layer or (args.layer == 'contextual0' and file_layer == '0'):
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

    output_dir_name = f"llm_judge_{model_name}_contextual{layer_num}_{model_suffix}_full-sentence"
    if args.use_cropped_region:
        output_dir_name += "_cropped"
    output_dir = Path(args.output_base) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    output_json_path = output_dir / f"results_{args.split}.json"

    print(f"\n{'='*60}")
    print(f"Contextual NN Judge (Full Sentence) - Model: {args.llm} + {args.vision_encoder}")
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
                print(f"Warning: Image not found at {image_path}")
                continue

            # contextual JSON holds chunks[...][patches][...]['nearest_contextual_neighbors']
            chunks = image_obj.get('chunks', [])
            # Flatten patches with coordinates and calculate grid_size FIRST
            patch_map = defaultdict(list)  # key = (row,col) -> nearest_contextual list
            max_row, max_col = 0, 0
            for ch in chunks:
                for p in ch.get('patches', []):
                    row = p.get('patch_row', 0)
                    col = p.get('patch_col', 0)
                    max_row = max(max_row, row)
                    max_col = max(max_col, col)
                    patch_map[(row, col)] = p.get('nearest_contextual_neighbors', [])

            # Calculate grid size from actual patch data
            grid_size = max(max_row + 1, max_col + 1) if (max_row > 0 or max_col > 0) else 24

            processed_image, image_mask = process_image_with_mask(image_path)

            # Sample patches uniformly from valid positions (using correct grid_size)
            sampled_positions = sample_valid_patch_positions(image_mask, bbox_size=3, num_samples=args.num_samples, grid_size=grid_size)

            image_result_entries = []

            for patch_row, patch_col in sampled_positions:
                bbox_size = 3
                actual_patch_size = 512 / grid_size
                bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=bbox_size)
                image_with_bbox = draw_bbox_on_image(processed_image, bbox)

                center_row = patch_row + bbox_size // 2
                center_col = patch_col + bbox_size // 2
                nearest_contextual = patch_map.get((center_row, center_col), [])
                if not nearest_contextual:
                    continue

                sentences_formatted = [highlight_token_in_sentence(e.get('caption', ''), e.get('token_str', '')) for e in nearest_contextual[:5]]
                while len(sentences_formatted) < 5:
                    sentences_formatted.append("")

                # Select prompt variant based on whether we send a cropped region
                if args.use_cropped_region:
                    prompt_text = SENTENCE_LEVEL_PROMPT_WITH_CROP.format(
                        candidate_sentence_1=sentences_formatted[0],
                        candidate_sentence_2=sentences_formatted[1],
                        candidate_sentence_3=sentences_formatted[2],
                        candidate_sentence_4=sentences_formatted[3],
                        candidate_sentence_5=sentences_formatted[4],
                    )
                else:
                    prompt_text = SENTENCE_LEVEL_PROMPT.format(
                        candidate_sentence_1=sentences_formatted[0],
                        candidate_sentence_2=sentences_formatted[1],
                        candidate_sentence_3=sentences_formatted[2],
                        candidate_sentence_4=sentences_formatted[3],
                        candidate_sentence_5=sentences_formatted[4],
                    )

                cropped_image = None
                if args.use_cropped_region:
                    # Crop using same coordinate logic with dynamic patch size
                    left = int(patch_col * actual_patch_size)
                    top = int(patch_row * actual_patch_size)
                    right = int((patch_col + bbox_size) * actual_patch_size)
                    bottom = int((patch_row + bbox_size) * actual_patch_size)
                    left = max(0, left); top = max(0, top)
                    right = min(right, processed_image.size[0]); bottom = min(bottom, processed_image.size[1])
                    cropped_image = processed_image.crop((left, top, right, bottom))

                raw_text = get_llm_response_sentences(
                    client, image_with_bbox, cropped_image, prompt_text,
                    api_provider=args.api_provider, api_model=args.api_model
                )

                parsed = parse_sentence_level_json(raw_text)

                # Compute a simple interpretable flag: any item has interpretable true
                interpretable_any = False
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get('interpretable'):
                            interpretable_any = True
                            break

                # Save visualization
                status_str = 'pass' if interpretable_any else 'fail'
                vis_name = f"img{image_idx}_r{patch_row}_c{patch_col}_{status_str}.jpg"
                create_visualization(viz_dir / vis_name, image_with_bbox, sentences_formatted, parsed, cropped_image=cropped_image)

                # Accumulate result entry
                image_result_entries.append({
                    'image_idx': image_idx,
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'sentences': sentences_formatted,
                    'llm_raw': raw_text,
                    'llm_parsed': parsed,
                    'interpretable_any': interpretable_any,
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
    acc = sum(1 for r in results if r.get('interpretable_any'))
    pct = (acc / total * 100.0) if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Results: {acc}/{total} ({pct:.1f}%)")
    print(f"Saved to: {output_json_path}")
    print(f"Visualizations: {viz_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

