#!/usr/bin/env python3
"""
Extract interpretable V-Lens examples for phrase-level annotation.

This script:
1. Reads LLM judge results and filters for examples where TOP-1 word is interpretable
2. Reads contextual NN data to get full phrases (not just words)
3. Shows: full phrase with highlighted token vs random corpus phrase
4. Saves examples for manual annotation
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from olmo.data.pixmo_datasets import PixMoCap


def calculate_square_bbox_from_patch(patch_row, patch_col, grid_size, img_w, img_h):
    """Calculate bbox coords for a patch in the image."""
    cell_w = img_w / grid_size
    cell_h = img_h / grid_size
    x1 = int(patch_col * cell_w)
    y1 = int(patch_row * cell_h)
    x2 = int((patch_col + 1) * cell_w)
    y2 = int((patch_row + 1) * cell_h)
    return x1, y1, x2, y2


def crop_around_patch(image, patch_row, patch_col, grid_size=24, context_patches=2):
    """Extract a crop around the target patch with some context."""
    img_w, img_h = image.size
    cell_w = img_w / grid_size
    cell_h = img_h / grid_size

    # Calculate bounds with context
    row_start = max(0, patch_row - context_patches)
    row_end = min(grid_size, patch_row + context_patches + 1)
    col_start = max(0, patch_col - context_patches)
    col_end = min(grid_size, patch_col + context_patches + 1)

    x1 = int(col_start * cell_w)
    y1 = int(row_start * cell_h)
    x2 = int(col_end * cell_w)
    y2 = int(row_end * cell_h)

    crop = image.crop((x1, y1, x2, y2))

    # Draw red box around center patch
    draw = ImageDraw.Draw(crop)
    center_x1 = int((patch_col - col_start) * cell_w)
    center_y1 = int((patch_row - row_start) * cell_h)
    center_x2 = int((patch_col - col_start + 1) * cell_w)
    center_y2 = int((patch_row - row_start + 1) * cell_h)
    draw.rectangle([center_x1, center_y1, center_x2, center_y2], outline='red', width=3)

    return crop


def highlight_token_in_sentence(sentence: str, token: str) -> str:
    """
    Return sentence with the token marked using **bold** markers.
    Expands subwords to full words when possible.
    """
    if not sentence or not token:
        return sentence or ""

    token = token.strip()
    if not token:
        return sentence

    low_sent = sentence.lower()
    low_tok = token.lower()
    idx = low_sent.find(low_tok)

    if idx == -1:
        return f"{sentence} (**{token}**)"

    # Expand to word boundaries
    def is_word_char(ch):
        return ch.isalnum() or ch == '_'

    start = idx
    end = idx + len(low_tok)

    # Expand left
    while start > 0 and is_word_char(sentence[start - 1]):
        start -= 1
    # Expand right
    while end < len(sentence) and is_word_char(sentence[end]):
        end += 1

    word = sentence[start:end]
    return sentence[:start] + f"**{word}**" + sentence[end:]


def highlight_by_token_position(sentence, token_position, llm, expected_token=None):
    """
    Highlight the word at the given token position using the actual tokenizer.
    Returns sentence with **word** markers around the correct word.
    If expected_token is provided, verifies the position matches and falls back to string search if not.
    """
    # Map LLM names to tokenizer names
    llm_to_tokenizer = {
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'olmo-7b': 'allenai/OLMo-7B-1024-preview',
        'qwen2-7b': 'Qwen/Qwen2-7B',
    }

    tokenizer_name = llm_to_tokenizer.get(llm)
    if not tokenizer_name:
        # Fallback to string matching
        if expected_token:
            return highlight_token_in_sentence(sentence, expected_token)
        return sentence

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        token_strs = tokenizer.convert_ids_to_tokens(tokens)

        # Adjust for BOS token offset in corpus positions
        adjusted_position = token_position - 1

        if adjusted_position < 0 or adjusted_position >= len(offsets):
            # Fall back to string matching
            if expected_token:
                return highlight_token_in_sentence(sentence, expected_token)
            return sentence

        # Verify the token at this position matches expected (if provided)
        if expected_token:
            actual_token = token_strs[adjusted_position].replace('Ġ', ' ').replace('▁', ' ').strip()
            expected_clean = expected_token.strip()
            if actual_token.lower() != expected_clean.lower():
                # Position doesn't match - fall back to string matching
                return highlight_token_in_sentence(sentence, expected_token)

        token_position = adjusted_position

        # Get the character span for this token
        start_char, end_char = offsets[token_position]

        # Skip leading whitespace (tokenizers often include leading space in token)
        while start_char < end_char and not sentence[start_char].isalnum():
            start_char += 1

        # Skip trailing whitespace
        while end_char > start_char and not sentence[end_char - 1].isalnum():
            end_char -= 1

        if start_char >= end_char:
            return sentence  # No alphanumeric content

        # Expand to word boundaries (only alphanumeric characters)
        while start_char > 0 and sentence[start_char - 1].isalnum():
            start_char -= 1
        while end_char < len(sentence) and sentence[end_char].isalnum():
            end_char += 1

        word = sentence[start_char:end_char]
        return sentence[:start_char] + f"**{word}**" + sentence[end_char:]

    except Exception as e:
        # Fallback if tokenizer fails
        return sentence


def draw_bbox_on_image(image, patch_row, patch_col, grid_size=24):
    """Draw a red bounding box on the full image at the patch location."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    img_w, img_h = image.size
    cell_w = img_w / grid_size
    cell_h = img_h / grid_size

    x1 = int(patch_col * cell_w)
    y1 = int(patch_row * cell_h)
    x2 = int((patch_col + 1) * cell_w)
    y2 = int((patch_row + 1) * cell_h)

    # Draw thick red rectangle
    for i in range(3):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='red')

    return img_copy


# Cache for token_embeddings.json files
_TOKEN_DATA_CACHE = {}

# Cache for contextual NN results
_CONTEXTUAL_NN_CACHE = {}


def load_contextual_nn_data(llm, vision, layer, base_dir):
    """Load contextual NN results for a given model/layer combination."""
    global _CONTEXTUAL_NN_CACHE

    contextual_nn_dir = base_dir / "analysis_results" / "contextual_nearest_neighbors"

    # Map vision encoder names to folder format
    vision_map = {
        'openai_clip-vit-large-patch14-336': 'vit-l-14-336',
        'google_siglip-so400m-patch14-384': 'siglip',
        'facebook_dinov2-large': 'dinov2-large-336',
    }
    vision_short = vision_map.get(vision, vision)

    # Folder pattern: train_mlp-only_pixmo_cap_resize_{llm}_{vision}_step12000-unsharded
    pattern = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_short}_*step12000-unsharded"
    matching_folders = [f for f in contextual_nn_dir.glob(pattern) if not f.name.endswith('_lite10')]

    if not matching_folders:
        return None

    nn_folder = matching_folders[0]
    # Use the file matching the target layer (layer 16 → visual16_allLayers.json)
    # This matches the fixed LLM judge behavior
    nn_file = nn_folder / f"contextual_neighbors_visual{layer}_allLayers.json"

    if not nn_file.exists():
        # Fallback to visual0 if layer-specific file doesn't exist
        nn_file = nn_folder / "contextual_neighbors_visual0_allLayers.json"
        if not nn_file.exists():
            return None

    cache_key = str(nn_file)
    if cache_key not in _CONTEXTUAL_NN_CACHE:
        with open(nn_file) as f:
            _CONTEXTUAL_NN_CACHE[cache_key] = json.load(f)

    return _CONTEXTUAL_NN_CACHE[cache_key]


def get_phrase_for_patch(nn_data, img_idx, patch_row, patch_col, layer):
    """
    Get the top-1 contextual NN phrase for a specific patch.

    IMPORTANT: LLM judge stores the top-left corner of a 3x3 bbox, but uses the
    CENTER patch for NN lookup. We must apply the same +1 offset to match.
    See run_llm_judge.py lines 696-700:
        center_row = patch_row + bbox_size // 2  # bbox_size=3, so +1
        center_col = patch_col + bbox_size // 2
    """
    if not nn_data:
        return None, None, None

    # Convert from bbox top-left (stored in LLM judge results) to center patch
    center_row = patch_row + 1  # bbox_size=3, so offset is 3//2 = 1
    center_col = patch_col + 1

    for result in nn_data.get('results', []):
        if result.get('image_idx') != img_idx:
            continue
        for chunk in result.get('chunks', []):
            for patch in chunk.get('patches', []):
                if patch.get('patch_row') == center_row and patch.get('patch_col') == center_col:
                    neighbors = patch.get('nearest_contextual_neighbors', [])
                    if neighbors:
                        top = neighbors[0]
                        return top.get('caption', ''), top.get('token_str', ''), top.get('position', 0)
    return None, None, None


def load_random_comparison_phrase(token, llm, layer, original_sentence=None):
    """
    Load a random other phrase from the contextual embeddings corpus that uses the same token.
    Returns (caption, token_position) tuple, or (None, None) if not found.
    """
    global _TOKEN_DATA_CACHE

    llm_to_folder = {
        'llama3-8b': 'meta-llama_Meta-Llama-3-8B',
        'olmo-7b': 'allenai_OLMo-7B-1024-preview',
        'qwen2-7b': 'Qwen_Qwen2-7B',
    }

    folder = llm_to_folder.get(llm)
    if not folder:
        return None, None

    base_path = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/molmo_data/contextual_llm_embeddings_vg")

    # Available layers in corpus (layer 0 doesn't exist)
    available_layers = [1, 2, 4, 8, 16, 24, 30, 31]

    # Find best layer
    layers_to_try = [layer] if layer in available_layers else [min(available_layers, key=lambda x: abs(x - layer))]

    token_json = None
    for try_layer in layers_to_try:
        candidate = base_path / folder / f"layer_{try_layer}" / "token_embeddings.json"
        if candidate.exists():
            token_json = candidate
            break

    if token_json is None:
        return None, None

    # Load from cache or file
    cache_key = str(token_json)
    if cache_key not in _TOKEN_DATA_CACHE:
        with open(token_json) as f:
            _TOKEN_DATA_CACHE[cache_key] = json.load(f)
    token_data = _TOKEN_DATA_CACHE[cache_key]

    # Try to find the token (with leading space as in tokenizer)
    token_key = f" {token.strip()}" if not token.startswith(' ') else token

    # Also try without leading space
    candidates = []
    matched_key = None
    for key in [token_key, token.strip(), token]:
        if key in token_data:
            candidates = token_data[key]
            matched_key = key
            break

    if not candidates:
        # Try case-insensitive search
        token_lower = token.strip().lower()
        for key, entries in token_data.items():
            if key.strip().lower() == token_lower:
                candidates = entries
                matched_key = key
                break

    if not candidates:
        return None, None

    # Filter out the original sentence if provided (we want a DIFFERENT example)
    if original_sentence:
        original_lower = original_sentence.lower().strip()
        filtered = [e for e in candidates if e.get('caption', '').lower().strip() != original_lower]
        if filtered:
            candidates = filtered

    # Pick a random one
    entry = random.choice(candidates)
    return entry.get('caption', ''), entry.get('position', 0)


def create_annotation_image(full_image, crop, sentence, token, model_info, output_path,
                           img_idx, patch_row, patch_col, llm,
                           comparison_phrase=None, comparison_token_pos=None, token_pos=None):
    """Create annotation image: full image with bbox on left, crop + sentences on right."""

    # Resize full image to fit nicely
    max_img_height = 350
    img_w, img_h = full_image.size
    scale = max_img_height / img_h
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    full_resized = full_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Draw bbox on full image
    full_with_bbox = draw_bbox_on_image(full_resized, patch_row, patch_col)

    # Resize crop
    crop_size = 120
    crop_resized = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)

    # Create canvas
    margin = 15
    text_width = 380
    right_panel_width = crop_size + text_width + 2 * margin
    total_width = new_w + right_panel_width + 3 * margin
    total_height = max(new_h + 2 * margin, 400)

    canvas = Image.new('RGB', (total_width, total_height), color='white')

    # Paste full image on left
    canvas.paste(full_with_bbox, (margin, margin))

    # Paste crop on right side
    crop_x = new_w + 2 * margin
    canvas.paste(crop_resized, (crop_x, margin))

    draw = ImageDraw.Draw(canvas)

    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        title_font = body_font = small_font = ImageFont.load_default()

    text_x = crop_x + crop_size + margin
    text_y = margin

    # Word wrap helper
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word) if current else word
            bbox = draw.textbbox((0, 0), test.replace("**", ""), font=font)
            if bbox[2] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def draw_highlighted_text(text, start_y, label, label_color='black', pre_highlighted=False):
        """Draw text with **highlighted** portions in yellow."""
        draw.text((text_x, start_y), label, fill=label_color, font=title_font)
        start_y += 20

        # If text is already highlighted (has ** markers), use as-is; otherwise highlight with token
        if pre_highlighted or '**' in text:
            highlighted = text
        else:
            highlighted = highlight_token_in_sentence(text, token)
        lines = wrap_text(highlighted, body_font, text_width - 10)

        for line in lines:
            x = text_x
            parts = re.split(r'(\*\*[^*]+\*\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    word = part[2:-2]
                    bbox = draw.textbbox((0, 0), word, font=title_font)
                    word_width = bbox[2] - bbox[0]
                    draw.rectangle([x-2, start_y-2, x + word_width + 2, start_y + 14], fill='yellow')
                    draw.text((x, start_y), word, fill='black', font=title_font)
                    x += word_width + 4
                else:
                    draw.text((x, start_y), part, fill='black', font=body_font)
                    if part:
                        bbox = draw.textbbox((0, 0), part, font=body_font)
                        x += bbox[2] - bbox[0]
            start_y += 18
        return start_y + 5

    # Draw V-Lens phrase (top-1 contextual NN match with highlighted token)
    if token_pos is not None:
        vlens_highlighted = highlight_by_token_position(sentence, token_pos, llm, expected_token=token)
    else:
        vlens_highlighted = highlight_token_in_sentence(sentence, token)
    text_y = draw_highlighted_text(vlens_highlighted, text_y, "V-Lens top phrase:", 'darkgreen', pre_highlighted=True)

    # Draw comparison phrase if available (use token position for correct highlighting)
    if comparison_phrase:
        text_y += 10
        # Highlight using actual tokenizer position, with fallback to string matching
        if comparison_token_pos is not None:
            comparison_highlighted = highlight_by_token_position(comparison_phrase, comparison_token_pos, llm, expected_token=token)
        else:
            comparison_highlighted = highlight_token_in_sentence(comparison_phrase, token)
        text_y = draw_highlighted_text(comparison_highlighted, text_y, "Random corpus example:", 'darkblue', pre_highlighted=True)

    # Draw model info at bottom
    text_y = max(text_y, crop_size + margin + 20)
    draw.text((crop_x, text_y), f"Model: {model_info}", fill='gray', font=small_font)
    text_y += 15
    draw.text((crop_x, text_y), f"Image {img_idx}, patch ({patch_row}, {patch_col})", fill='gray', font=small_font)

    canvas.save(output_path)


def main():
    random.seed(42)

    base_dir = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo")
    output_dir = base_dir / "analysis_results" / "phrase_annotation_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PixMoCap for images
    dataset = PixMoCap(split="validation", mode="captions")

    # Find all LLM judge result files (word-level, which has more coverage)
    llm_judge_dir = base_dir / "analysis_results" / "llm_judge_contextual_nn"
    contextual_nn_dir = base_dir / "analysis_results" / "contextual_nearest_neighbors"

    # Collect all interpretable examples
    all_examples = []

    # Process each model/layer combination
    for result_folder in sorted(llm_judge_dir.glob("llm_judge_*_gpt5_cropped")):
        if '/ablations/' in str(result_folder):
            continue

        results_file = result_folder / "results_validation.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            results = json.load(f)

        llm = results.get('llm', '')
        vision = results.get('vision_encoder', '')

        # Parse layer from folder name (e.g., contextual16)
        folder_name = result_folder.name
        layer_match = re.search(r'contextual(\d+)', folder_name)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))

        # Find examples where TOP-1 word is interpretable (in any of 3 categories)
        for result in results.get('results', []):
            words = result.get('words', [])
            if not words:
                continue

            top_word = words[0]
            gpt_resp = result.get('gpt_response', {})
            concrete = gpt_resp.get('concrete_words', [])
            abstract = gpt_resp.get('abstract_words', [])
            global_w = gpt_resp.get('global_words', [])

            # Filter: top-1 word must be in one of the 3 interpretable categories
            if top_word not in concrete and top_word not in abstract and top_word not in global_w:
                continue

            img_idx = result['image_idx']
            patch_row = result['patch_row']
            patch_col = result['patch_col']

            all_examples.append({
                'img_idx': img_idx,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'llm': llm,
                'vision': vision,
                'layer': layer,
                'model_info': f"{llm} + {vision}, L{layer}",
                'better_than_no_context': "",
                'better_than_random_context': "",
            })

    print(f"Found {len(all_examples)} interpretable examples total")

    # Group by model for diverse selection
    by_model = defaultdict(list)
    for ex in all_examples:
        key = (ex['llm'], ex['vision'])
        by_model[key].append(ex)

    # Shuffle each group
    for key in by_model:
        random.shuffle(by_model[key])

    # Select only 60 valid examples (validate lazily during selection)
    num_examples = 60
    selected = []
    attempts = 0
    max_attempts = 500

    while len(selected) < num_examples and attempts < max_attempts:
        # Round-robin across models
        for key in list(by_model.keys()):
            if not by_model[key] or len(selected) >= num_examples:
                continue

            ex = by_model[key].pop()
            attempts += 1

            # Load full phrase from contextual NN
            nn_data = load_contextual_nn_data(ex['llm'], ex['vision'], ex['layer'], base_dir)
            phrase, token_str, token_pos = get_phrase_for_patch(
                nn_data, ex['img_idx'], ex['patch_row'], ex['patch_col'], ex['layer']
            )

            if not phrase:
                continue

            # Skip if token is just punctuation (not a real word)
            token_clean = token_str.strip() if token_str else ''
            if not token_clean or not any(c.isalnum() for c in token_clean):
                continue

            # Get comparison phrase (random corpus example with same token)
            comparison_phrase, comparison_token_pos = load_random_comparison_phrase(
                token_str, ex['llm'], ex['layer'], original_sentence=phrase
            )

            if not comparison_phrase:
                continue

            ex['phrase'] = phrase
            ex['token'] = token_str
            ex['token_pos'] = token_pos
            ex['_comparison_phrase'] = comparison_phrase
            ex['_comparison_token_pos'] = comparison_token_pos
            selected.append(ex)

        # Remove empty model groups
        by_model = {k: v for k, v in by_model.items() if v}

    print(f"Selected {len(selected)} examples for annotation")

    # Create visualizations
    for i, ex in enumerate(tqdm(selected, desc="Creating visualizations")):
        try:
            # Load image
            example = dataset.get(ex['img_idx'], random)
            image_path = example['image']
            if not Path(image_path).exists():
                print(f"Image not found: {image_path}")
                continue
            image = Image.open(image_path).convert('RGB')

            # Create crop
            crop = crop_around_patch(image, ex['patch_row'], ex['patch_col'])

            # Use pre-validated comparison phrase
            comparison_phrase = ex['_comparison_phrase']
            comparison_token_pos = ex['_comparison_token_pos']

            # Create visualization with full phrases
            output_path = output_dir / f"example_{i:03d}_img{ex['img_idx']}_{ex['llm']}_{ex['vision']}_L{ex['layer']}.png"
            create_annotation_image(
                image, crop, ex['phrase'], ex['token'], ex['model_info'],
                output_path, ex['img_idx'], ex['patch_row'], ex['patch_col'],
                llm=ex['llm'],
                comparison_phrase=comparison_phrase,
                comparison_token_pos=comparison_token_pos,
                token_pos=ex['token_pos']
            )
        except Exception as e:
            print(f"Error processing example {i}: {e}")

    # Save metadata
    metadata = {
        'num_examples': len(selected),
        'examples': selected
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Examples saved to: {output_dir}")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
