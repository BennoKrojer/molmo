#!/usr/bin/env python3
"""
Generate random (non-cherry-picked) method comparison examples for appendix.
VERSION 2: PNG images + LaTeX tables (proper formatting)

Creates 20 examples (4 per layer × 5 layers) comparing:
- EmbeddingLens (nearest neighbors in input embedding matrix)
- LogitLens (LM head predictions)
- LatentLens (contextual nearest neighbors) - Ours

Uses existing code from:
- create_phrase_example_pdfs.py: preprocess_image(), find_token_in_phrase()
- method_comparison_table_v3.tex: LaTeX formatting patterns
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
import sys
BASE_DIR = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts" / "analysis"))

from olmo.data.pixmo_datasets import PixMoCap


# ============================================================================
# REUSE EXISTING CODE FROM create_phrase_example_pdfs.py
# ============================================================================

# Vision encoder configs - EXACT copy from create_phrase_example_pdfs.py:22-26
VISION_CONFIGS = {
    'vit-l-14-336': {'size': 336, 'method': 'resize_pad', 'grid': 24},
    'siglip': {'size': 384, 'method': 'squash', 'grid': 27},
    'dinov2-large-336': {'size': 336, 'method': 'squash', 'grid': 24},
    'qwen2-vl-native': {'size': 448, 'method': 'squash', 'grid': 28},  # Qwen2-VL
}


def preprocess_image(image, vision_encoder, patch_row, patch_col):
    """
    Apply vision encoder preprocessing and draw bbox.
    EXACT copy from create_phrase_example_pdfs.py:29-88
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

        # Create padded square with light gray background (as in original)
        # Black padding for CLIP/ViT (consistent with rest of paper)
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

    # Draw bbox - EXACT copy from create_phrase_example_pdfs.py:77-86
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
    """
    Find token in phrase, expand to word boundaries.
    EXACT copy from create_phrase_example_pdfs.py:91-104
    Returns (before, matched_word, after).
    """
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


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
@dataclass
class ModelConfig:
    """Configuration for a single model."""
    checkpoint_name: str
    llm: str
    vision_encoder: str
    grid_size: int
    last_layer: int
    display_name: str
    is_qwen2vl: bool = False


MODELS = [
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded',
        llm='olmo-7b', vision_encoder='vit-l-14-336', grid_size=24, last_layer=31,
        display_name='OLMo+CLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded',
        llm='olmo-7b', vision_encoder='siglip', grid_size=27, last_layer=31,
        display_name='OLMo+SigLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded',
        llm='olmo-7b', vision_encoder='dinov2-large-336', grid_size=24, last_layer=31,
        display_name='OLMo+DINOv2'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded',
        llm='llama3-8b', vision_encoder='vit-l-14-336', grid_size=24, last_layer=31,
        display_name='LLaMA3+CLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded',
        llm='llama3-8b', vision_encoder='siglip', grid_size=27, last_layer=31,
        display_name='LLaMA3+SigLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded',
        llm='llama3-8b', vision_encoder='dinov2-large-336', grid_size=24, last_layer=31,
        display_name='LLaMA3+DINOv2'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10_step12000-unsharded',
        llm='qwen2-7b', vision_encoder='vit-l-14-336', grid_size=24, last_layer=27,
        display_name='Qwen2+CLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded',
        llm='qwen2-7b', vision_encoder='siglip', grid_size=27, last_layer=27,
        display_name='Qwen2+SigLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded',
        llm='qwen2-7b', vision_encoder='dinov2-large-336', grid_size=24, last_layer=27,
        display_name='Qwen2+DINOv2'
    ),
    ModelConfig(
        checkpoint_name='qwen2_vl/Qwen_Qwen2-VL-7B-Instruct',
        llm='qwen2-vl', vision_encoder='qwen2-vl-native', grid_size=28, last_layer=27,
        display_name='Qwen2-VL',
        is_qwen2vl=True
    ),
]


# ============================================================================
# DATA LOADING (same as before)
# ============================================================================
_NN_CACHE = {}
_LOGIT_CACHE = {}
_CONTEXTUAL_CACHE = {}


def load_embedding_lens(model: ModelConfig, layer: int) -> Optional[dict]:
    """Load EmbeddingLens (nearest neighbors) data."""
    if model.is_qwen2vl:
        path = BASE_DIR / "analysis_results" / "nearest_neighbors" / "qwen2_vl" / \
               "Qwen_Qwen2-VL-7B-Instruct" / f"nearest_neighbors_layer{layer}_topk5.json"
    else:
        path = BASE_DIR / "analysis_results" / "nearest_neighbors" / model.checkpoint_name / \
               f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{layer}.json"

    cache_key = str(path)
    if cache_key not in _NN_CACHE:
        if not path.exists():
            print(f"Warning: Missing EmbeddingLens data: {path}")
            return None
        with open(path) as f:
            _NN_CACHE[cache_key] = json.load(f)
    return _NN_CACHE[cache_key]


def load_logit_lens(model: ModelConfig, layer: int) -> Optional[dict]:
    """Load LogitLens data."""
    if model.is_qwen2vl:
        path = BASE_DIR / "analysis_results" / "logit_lens" / "qwen2_vl" / \
               "Qwen_Qwen2-VL-7B-Instruct" / f"logit_lens_layer{layer}_topk5.json"
    else:
        path = BASE_DIR / "analysis_results" / "logit_lens" / model.checkpoint_name / \
               f"logit_lens_layer{layer}_topk5_multi-gpu.json"

    cache_key = str(path)
    if cache_key not in _LOGIT_CACHE:
        if not path.exists():
            print(f"Warning: Missing LogitLens data: {path}")
            return None
        with open(path) as f:
            _LOGIT_CACHE[cache_key] = json.load(f)
    return _LOGIT_CACHE[cache_key]


def load_latent_lens(model: ModelConfig, layer: int) -> Optional[dict]:
    """Load LatentLens (contextual NN) data."""
    if model.is_qwen2vl:
        path = BASE_DIR / "analysis_results" / "contextual_nearest_neighbors" / "qwen2_vl" / \
               "Qwen_Qwen2-VL-7B-Instruct" / f"contextual_neighbors_visual{layer}_allLayers.json"
    else:
        path = BASE_DIR / "analysis_results" / "contextual_nearest_neighbors" / model.checkpoint_name / \
               f"contextual_neighbors_visual{layer}_allLayers.json"

    cache_key = str(path)
    if cache_key not in _CONTEXTUAL_CACHE:
        if not path.exists():
            print(f"Warning: Missing LatentLens data: {path}")
            return None
        with open(path) as f:
            _CONTEXTUAL_CACHE[cache_key] = json.load(f)
    return _CONTEXTUAL_CACHE[cache_key]


def get_patch_data(data: dict, image_idx: int, patch_row: int, patch_col: int,
                   data_type: str, is_qwen2vl: bool = False) -> List[Tuple]:
    """Extract top-3 predictions for a patch from loaded data."""
    results = []

    if data_type == 'embedding':
        if is_qwen2vl:
            for img in data.get('results', []):
                if img.get('image_idx') == image_idx:
                    for patch in img.get('patches', []):
                        if patch.get('patch_row') == patch_row and patch.get('patch_col') == patch_col:
                            for nn in patch.get('nearest_neighbors', [])[:3]:
                                results.append((nn.get('token', '?'), nn.get('similarity', 0)))
                            break
                    break
        else:
            for img in data.get('splits', {}).get('validation', {}).get('images', []):
                if img.get('image_idx') == image_idx:
                    for chunk in img.get('chunks', []):
                        for patch in chunk.get('patches', []):
                            if patch.get('patch_row') == patch_row and patch.get('patch_col') == patch_col:
                                for nn in patch.get('nearest_neighbors', [])[:3]:
                                    results.append((nn.get('token', '?'), nn.get('similarity', 0)))
                                break
                    break

    elif data_type == 'logit':
        for img in data.get('results', []):
            if img.get('image_idx') == image_idx:
                if is_qwen2vl:
                    for patch in img.get('patches', []):
                        if patch.get('patch_row') == patch_row and patch.get('patch_col') == patch_col:
                            for pred in patch.get('top_predictions', [])[:3]:
                                results.append((pred.get('token', '?'), pred.get('logit', 0)))
                            break
                else:
                    for chunk in img.get('chunks', []):
                        for patch in chunk.get('patches', []):
                            if patch.get('patch_row') == patch_row and patch.get('patch_col') == patch_col:
                                for pred in patch.get('top_predictions', [])[:3]:
                                    results.append((pred.get('token', '?'), pred.get('logit', 0)))
                                break
                break

    elif data_type == 'contextual':
        for img in data.get('results', []):
            if img.get('image_idx') == image_idx:
                if is_qwen2vl:
                    patches_source = img.get('patches', [])
                    if not patches_source:
                        for chunk in img.get('chunks', []):
                            patches_source.extend(chunk.get('patches', []))
                else:
                    patches_source = []
                    for chunk in img.get('chunks', []):
                        patches_source.extend(chunk.get('patches', []))

                for patch in patches_source:
                    if patch.get('patch_row') == patch_row and patch.get('patch_col') == patch_col:
                        for nn in patch.get('nearest_contextual_neighbors', [])[:3]:
                            token = nn.get('token_str', '?')
                            caption = nn.get('caption', '')
                            sim = nn.get('similarity', 0)
                            results.append((token, sim, caption))
                        break
                break

    return results


# ============================================================================
# LATEX FORMATTING (matching method_comparison_table_v3.tex)
# ============================================================================

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
        ('<', r'\textless{}'),
        ('>', r'\textgreater{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_latentlens_phrase(token: str, phrase: str) -> str:
    """
    Format LatentLens phrase with yellow-highlighted word.
    Uses \colorbox{yellow!50}{\textbf{word}} as in method_comparison_table_v3.tex
    """
    before, matched, after = find_token_in_phrase(phrase, token)

    # Truncate if too long
    max_len = 45
    full = before + matched + after
    if len(full) > max_len:
        # Try to keep the matched word visible
        if len(matched) > max_len - 10:
            matched = matched[:max_len-13] + "..."
            before, after = "", ""
        else:
            # Truncate around the match
            available = max_len - len(matched) - 3
            before_len = min(len(before), available // 2)
            after_len = min(len(after), available - before_len)
            if before_len < len(before):
                before = "..." + before[-(before_len-3):] if before_len > 3 else ""
            if after_len < len(after):
                after = after[:after_len-3] + "..." if after_len > 3 else ""

    # Escape and format
    before_esc = escape_latex(before)
    matched_esc = escape_latex(matched)
    after_esc = escape_latex(after)

    if matched_esc:
        return f'``{before_esc}\\colorbox{{yellow!50}}{{\\textbf{{{matched_esc}}}}}{after_esc}\'\''
    else:
        return f'``{escape_latex(phrase[:max_len])}\'\''


def format_token_for_latex(token: str, color: str) -> str:
    """
    Format a token with colored background.
    Uses \mtag{color}{token} pattern from method_comparison_table_v3.tex
    """
    token = token.strip()
    if not token:
        return r"\mtag{" + color + r"}{?}"

    # Check for non-printable/CJK characters
    has_problematic = False
    cleaned = []
    for c in token:
        if ord(c) < 32:  # Control chars
            continue
        elif ord(c) >= 0x3000:  # CJK range
            has_problematic = True
        elif ord(c) >= 0x0400 and ord(c) < 0x0500:  # Cyrillic
            has_problematic = True
        else:
            cleaned.append(c)

    if has_problematic and len(cleaned) < 2:
        # Mostly non-Latin, use placeholder
        return r"\mtag{" + color + r"}{[?]}"

    token_clean = ''.join(cleaned).strip()
    if not token_clean:
        return r"\mtag{" + color + r"}{[?]}"

    # Truncate if too long
    if len(token_clean) > 12:
        token_clean = token_clean[:10] + ".."

    # Escape for LaTeX
    token_esc = escape_latex(token_clean)

    return r"\mtag{" + color + r"}{" + token_esc + r"}"


# ============================================================================
# MAIN
# ============================================================================
def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading PixMoCap dataset...")
    dataset = PixMoCap(split="validation", mode="captions")

    output_dir = BASE_DIR / "paper" / "figures" / "random_method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer configurations
    LAYERS = [0, 8, 16, 24, 'final']
    LAYER_NAMES = {
        0: "Layer 0 (input)",
        8: "Layer 8 (early-mid)",
        16: "Layer 16 (middle)",
        24: "Layer 24 (late)",
        'final': "Final layer"
    }

    examples_per_layer = 4
    all_examples = []
    latex_content = []

    # LaTeX preamble for the generated file
    latex_content.append(r"% Auto-generated by create_random_method_comparison_v2.py")
    latex_content.append(r"% Uses \mtag and \colorbox formatting from method_comparison_table_v3.tex")
    latex_content.append(r"")
    latex_content.append(r"% Define mtag if not already defined")
    latex_content.append(r"\providecommand{\mtag}[2]{\colorbox{#1}{\rule{0pt}{1ex}#2}}")
    latex_content.append(r"\providecommand{\tagsp}{\hspace{3pt}}")
    latex_content.append(r"")

    for layer_key in LAYERS:
        print(f"\n=== Processing {LAYER_NAMES[layer_key]} ===")

        layer_examples = []
        example_num = 0
        attempts = 0
        max_attempts = 50

        while example_num < examples_per_layer and attempts < max_attempts:
            attempts += 1

            # Randomly select a model
            model = random.choice(MODELS)

            # Determine actual layer number
            if layer_key == 'final':
                layer = model.last_layer
            else:
                layer = layer_key

            # Load data for all three methods
            emb_data = load_embedding_lens(model, layer)
            logit_data = load_logit_lens(model, layer)
            ctx_data = load_latent_lens(model, layer)

            if not all([emb_data, logit_data, ctx_data]):
                continue

            # Find available images
            available_images = [r.get('image_idx') for r in ctx_data.get('results', [])]
            if not available_images:
                continue

            # Randomly select an image
            img_idx = random.choice(available_images)

            # Randomly select a patch (avoiding edges)
            margin = 2
            max_row = model.grid_size - margin - 1
            max_col = model.grid_size - margin - 1
            patch_row = random.randint(margin, max_row)
            patch_col = random.randint(margin, max_col)

            # Get patch data from all three methods
            emb_results = get_patch_data(emb_data, img_idx, patch_row, patch_col,
                                         'embedding', model.is_qwen2vl)
            logit_results = get_patch_data(logit_data, img_idx, patch_row, patch_col,
                                           'logit', model.is_qwen2vl)
            ctx_results = get_patch_data(ctx_data, img_idx, patch_row, patch_col,
                                         'contextual', model.is_qwen2vl)

            if not ctx_results:
                continue

            print(f"  Example {example_num}: {model.display_name}, layer {layer}, img {img_idx}, patch ({patch_row},{patch_col})")

            # Load and preprocess image using EXISTING code
            example_data = dataset.get(img_idx, random)
            image_path = example_data['image']
            if not Path(image_path).exists():
                print(f"    Skipping - image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')
            img_array, _ = preprocess_image(image, model.vision_encoder, patch_row, patch_col)

            # Save PNG
            layer_str = str(layer) if layer_key != 'final' else "final"
            png_name = f"layer{layer_str}_ex{example_num}.png"
            png_path = output_dir / png_name
            Image.fromarray(img_array).save(png_path)
            print(f"    Saved: {png_name}")

            # Store example data for LaTeX
            layer_examples.append({
                'example_num': example_num,
                'model': model.display_name,
                'layer': layer,
                'layer_key': layer_key,
                'png_name': png_name,
                'emb_results': emb_results,
                'logit_results': logit_results,
                'ctx_results': ctx_results,
            })

            example_num += 1

        all_examples.extend(layer_examples)

        # Generate LaTeX for this layer
        latex_content.append(r"% " + "=" * 60)
        latex_content.append(f"% {LAYER_NAMES[layer_key]}")
        latex_content.append(r"% " + "=" * 60)
        latex_content.append(r"\begin{figure}[H]")
        latex_content.append(r"\centering")
        latex_content.append(r"\footnotesize")

        # 2x2 grid using minipages
        for row in range(2):
            latex_content.append(r"\begin{minipage}{0.48\textwidth}")
            latex_content.append(r"\centering")

            for col in range(2):
                idx = row * 2 + col
                if idx < len(layer_examples):
                    ex = layer_examples[idx]

                    # Image
                    latex_content.append(r"\includegraphics[width=0.9\textwidth]{figures/random_method_comparison/" + ex['png_name'] + r"}")
                    latex_content.append(r"")
                    latex_content.append(r"\vspace{0.3em}")
                    latex_content.append(r"")

                    # Model/Layer label
                    latex_content.append(r"{\scriptsize \textbf{" + escape_latex(ex['model']) + r"}, L" + str(ex['layer']) + r"}")
                    latex_content.append(r"")
                    latex_content.append(r"\vspace{0.3em}")
                    latex_content.append(r"")

                    # LatentLens - top 3 with yellow highlight
                    # All results left-aligned for consistency
                    latex_content.append(r"\begin{flushleft}")

                    # LatentLens - top 3 phrases with yellow highlighting
                    latex_content.append(r"{\scriptsize \textcolor{green!50!black}{\textbf{LatentLens:}}}")
                    latex_content.append(r"\vspace{0.1em}")
                    latex_content.append(r"")
                    latex_content.append(r"\scriptsize")
                    for i, item in enumerate(ex['ctx_results'][:3]):
                        if len(item) >= 3:
                            token, sim, caption = item[0], item[1], item[2]
                            formatted = format_latentlens_phrase(token, caption)
                            latex_content.append(f"({i+1}) {formatted}" + r"\\")
                    latex_content.append(r"")
                    latex_content.append(r"\vspace{0.3em}")

                    # EmbeddingLens - top 3 with red tags
                    emb_tokens = [format_token_for_latex(t[0], 'red!20') for t in ex['emb_results'][:3]]
                    latex_content.append(r"{\scriptsize \textcolor{red!70!black}{\textbf{EmbeddingLens:}} " + r"\tagsp".join(emb_tokens) + r"}")
                    latex_content.append(r"")
                    latex_content.append(r"\vspace{0.2em}")
                    latex_content.append(r"")

                    # LogitLens - top 3 with blue tags
                    logit_tokens = [format_token_for_latex(t[0], 'blue!20') for t in ex['logit_results'][:3]]
                    latex_content.append(r"{\scriptsize \textcolor{blue!70!black}{\textbf{LogitLens:}} " + r"\tagsp".join(logit_tokens) + r"}")

                    latex_content.append(r"\end{flushleft}")

                if col == 0 and idx + 1 < len(layer_examples):
                    latex_content.append(r"\end{minipage}")
                    latex_content.append(r"\hfill")
                    latex_content.append(r"\begin{minipage}{0.48\textwidth}")
                    latex_content.append(r"\centering")

            latex_content.append(r"\end{minipage}")
            if row == 0:
                latex_content.append(r"")
                latex_content.append(r"\vspace{1em}")
                latex_content.append(r"")

        # Caption
        if layer_key == 'final':
            cap_text = r"Four randomly sampled visual tokens at the final layer (31 for OLMo/LLaMA, 27 for Qwen2)."
        else:
            cap_text = f"Four randomly sampled visual tokens at layer {layer_key}."

        latex_content.append(r"\caption{\textbf{" + LAYER_NAMES[layer_key] + r":} " + cap_text + r"}")
        latex_content.append(r"\label{fig:random_layer" + str(layer_key) + r"}")
        latex_content.append(r"\end{figure}")
        latex_content.append(r"")

    # Save LaTeX file
    latex_path = output_dir / "random_comparison_content.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_content))

    # Save metadata
    metadata = {
        'seed': 42,
        'num_examples': len(all_examples),
        'examples_per_layer': examples_per_layer,
        'layers': [0, 8, 16, 24, 'final'],
        'num_models': len(MODELS),
        'examples': [{k: v for k, v in ex.items() if k not in ['emb_results', 'logit_results', 'ctx_results']}
                     for ex in all_examples]
    }
    metadata_path = BASE_DIR / "analysis_results" / "random_method_comparison" / "metadata_v2.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Created {len(all_examples)} examples")
    print(f"PNGs saved to: {output_dir}")
    print(f"LaTeX saved to: {latex_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nTo use in appendix.tex, replace Section P content with:")
    print(f"  \\input{{figures/random_method_comparison/random_comparison_content.tex}}")


if __name__ == "__main__":
    main()
