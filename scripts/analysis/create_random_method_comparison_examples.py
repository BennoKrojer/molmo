#!/usr/bin/env python3
"""
Generate random (non-cherry-picked) method comparison examples for appendix.

Creates 20 examples (4 per layer × 5 layers) comparing:
- EmbeddingLens (nearest neighbors in input embedding matrix)
- LogitLens (LM head predictions)
- LatentLens (contextual nearest neighbors) - Ours

Each example randomly samples from 10 models (9 trained + Qwen2-VL).
Uses fixed seed (42) for reproducibility.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
import sys
BASE_DIR = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo")
sys.path.insert(0, str(BASE_DIR))
from olmo.data.pixmo_datasets import PixMoCap


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
        display_name='OLMo + CLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded',
        llm='olmo-7b', vision_encoder='siglip', grid_size=27, last_layer=31,
        display_name='OLMo + SigLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded',
        llm='olmo-7b', vision_encoder='dinov2-large-336', grid_size=24, last_layer=31,
        display_name='OLMo + DINOv2'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded',
        llm='llama3-8b', vision_encoder='vit-l-14-336', grid_size=24, last_layer=31,
        display_name='LLaMA3 + CLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded',
        llm='llama3-8b', vision_encoder='siglip', grid_size=27, last_layer=31,
        display_name='LLaMA3 + SigLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded',
        llm='llama3-8b', vision_encoder='dinov2-large-336', grid_size=24, last_layer=31,
        display_name='LLaMA3 + DINOv2'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10_step12000-unsharded',
        llm='qwen2-7b', vision_encoder='vit-l-14-336', grid_size=24, last_layer=27,
        display_name='Qwen2 + CLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded',
        llm='qwen2-7b', vision_encoder='siglip', grid_size=27, last_layer=27,
        display_name='Qwen2 + SigLIP'
    ),
    ModelConfig(
        checkpoint_name='train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded',
        llm='qwen2-7b', vision_encoder='dinov2-large-336', grid_size=24, last_layer=27,
        display_name='Qwen2 + DINOv2'
    ),
    ModelConfig(
        checkpoint_name='qwen2_vl/Qwen_Qwen2-VL-7B-Instruct',
        llm='qwen2-vl', vision_encoder='qwen2-vl-native', grid_size=28, last_layer=27,
        display_name='Qwen2-VL',
        is_qwen2vl=True
    ),
]


# ============================================================================
# DATA LOADING
# ============================================================================
# Caches to avoid reloading same files
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
                   data_type: str, is_qwen2vl: bool = False) -> List[Tuple[str, float]]:
    """
    Extract top-3 predictions for a patch from loaded data.

    Returns list of (token, score) tuples.
    """
    results = []

    if data_type == 'embedding':
        # EmbeddingLens: different structure for Qwen2-VL vs trained models
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
            # Trained models: splits.validation.images[].chunks[].patches[]
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
        # LogitLens: results[].chunks[].patches[] for trained, results[].patches[] for Qwen2-VL
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
        # LatentLens: results[].chunks[].patches[]
        for img in data.get('results', []):
            if img.get('image_idx') == image_idx:
                if is_qwen2vl:
                    # Qwen2-VL may have patches directly
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
                            # LatentLens has phrase context - show token + context snippet
                            token = nn.get('token_str', '?')
                            caption = nn.get('caption', '')
                            sim = nn.get('similarity', 0)
                            results.append((token, sim, caption))
                        break
                break

    return results


# ============================================================================
# IMAGE PROCESSING (adapted from create_phrase_example_pdfs.py)
# ============================================================================
VISION_CONFIGS = {
    'vit-l-14-336': {'size': 336, 'method': 'resize_pad', 'grid': 24},
    'siglip': {'size': 384, 'method': 'squash', 'grid': 27},
    'dinov2-large-336': {'size': 336, 'method': 'squash', 'grid': 24},
    'qwen2-vl-native': {'size': 448, 'method': 'squash', 'grid': 28},
}


def preprocess_and_draw_bbox(image: Image.Image, model: ModelConfig,
                             patch_row: int, patch_col: int) -> np.ndarray:
    """
    Preprocess image as vision encoder sees it and draw red bbox on patch.
    Adapted from create_phrase_example_pdfs.py lines 29-88.
    """
    config = VISION_CONFIGS.get(model.vision_encoder, VISION_CONFIGS['vit-l-14-336'])
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

        # Create padded square with light gray background
        padded = Image.new('RGB', (target_size, target_size), (240, 240, 240))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        padded.paste(resized, (offset_x, offset_y))

        # Adjust patch coordinates for padding
        cell_orig_w = img_w / grid_size
        cell_orig_h = img_h / grid_size
        cell_new = target_size / grid_size

        patch_x_orig = patch_col * cell_orig_w
        patch_y_orig = patch_row * cell_orig_h
        patch_x_new = patch_x_orig * scale + offset_x
        patch_y_new = patch_y_orig * scale + offset_y

        new_patch_col = patch_x_new / cell_new
        new_patch_row = patch_y_new / cell_new

        processed = padded
    else:  # squash
        # SigLIP/DINOv2/Qwen2-VL: direct squash to square
        processed = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
        new_patch_col = patch_col
        new_patch_row = patch_row

    # Draw bbox (adapted from extract_phrase_annotation_examples.py lines 182-199)
    draw = ImageDraw.Draw(processed)
    cell_size = target_size / grid_size
    x1 = int(new_patch_col * cell_size)
    y1 = int(new_patch_row * cell_size)
    x2 = int((new_patch_col + 1) * cell_size)
    y2 = int((new_patch_row + 1) * cell_size)

    # Draw thick red rectangle (3 pixels)
    for i in range(3):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='red')

    return np.array(processed)


# ============================================================================
# PDF GENERATION
# ============================================================================
def highlight_token_in_phrase(phrase: str, token: str) -> str:
    """Find and bold the token in phrase."""
    if not phrase or not token:
        return phrase or ""

    token_clean = token.strip()
    idx = phrase.lower().find(token_clean.lower())
    if idx == -1:
        return phrase

    # Expand to word boundaries
    start, end = idx, idx + len(token_clean)
    while start > 0 and (phrase[start-1].isalnum() or phrase[start-1] == '-'):
        start -= 1
    while end < len(phrase) and (phrase[end].isalnum() or phrase[end] == '-'):
        end += 1

    return phrase[:start] + "**" + phrase[start:end] + "**" + phrase[end:]


def create_example_pdf(image_array: np.ndarray, model: ModelConfig, layer: int,
                       embedding_lens: List, logit_lens: List, latent_lens: List,
                       output_path: Path, example_num: int):
    """
    Create a PDF for one example showing image + top-3 from all three methods.

    Layout:
    +-------------------------------------------+
    | Model + Layer title                        |
    |                                           |
    |        [Image with red bbox]              |
    |                                           |
    | LatentLens (ours):                        |
    |   1. "phrase with **word**"               |
    |   2. "another **phrase**"                 |
    |   3. "third **example**"                  |
    |                                           |
    | EmbeddingLens:  tok1  tok2  tok3          |
    | LogitLens:      tok1  tok2  tok3          |
    +-------------------------------------------+
    """
    fig = plt.figure(figsize=(3.2, 4.0), facecolor='white')

    # Title
    title = f"{model.display_name}, Layer {layer}"
    fig.text(0.5, 0.97, title, fontsize=9, fontweight='bold', ha='center', va='top',
             fontfamily='sans-serif')

    # Image
    ax_img = fig.add_axes([0.08, 0.52, 0.84, 0.42])
    ax_img.imshow(image_array)
    ax_img.axis('off')

    # Text area
    ax_txt = fig.add_axes([0.05, 0.02, 0.90, 0.48])
    ax_txt.axis('off')
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    y = 0.95

    # LatentLens (ours) - show phrases with highlighted tokens
    ax_txt.text(0, y, "LatentLens (ours):", fontsize=7, fontweight='bold',
                color='#2E7D32', transform=ax_txt.transAxes, va='top')
    y -= 0.08

    for i, item in enumerate(latent_lens[:3]):
        if len(item) >= 3:
            token, sim, caption = item[0], item[1], item[2]
            # Truncate caption if too long and highlight token
            if len(caption) > 45:
                caption = caption[:42] + "..."
            highlighted = highlight_token_in_phrase(caption, token)
            # Replace **word** with just the word (we'll draw it bold)
            display_text = f"{i+1}. \"{caption}\""
            ax_txt.text(0.02, y, display_text, fontsize=5.5, style='italic',
                       transform=ax_txt.transAxes, va='top', fontfamily='serif',
                       wrap=True)
        else:
            ax_txt.text(0.02, y, f"{i+1}. (no data)", fontsize=5.5,
                       transform=ax_txt.transAxes, va='top', color='gray')
        y -= 0.10

    y -= 0.02

    # EmbeddingLens - just tokens
    tokens_emb = [t[0].strip() if t else '?' for t in embedding_lens[:3]]
    tokens_emb_str = "  ".join([f"'{t}'" if t else '?' for t in tokens_emb])
    ax_txt.text(0, y, "EmbeddingLens:", fontsize=6, fontweight='bold',
                color='#C62828', transform=ax_txt.transAxes, va='top')
    ax_txt.text(0.28, y, tokens_emb_str, fontsize=5.5,
                transform=ax_txt.transAxes, va='top', fontfamily='monospace')
    y -= 0.10

    # LogitLens - just tokens
    tokens_logit = [t[0].strip() if t else '?' for t in logit_lens[:3]]
    tokens_logit_str = "  ".join([f"'{t}'" if t else '?' for t in tokens_logit])
    ax_txt.text(0, y, "LogitLens:", fontsize=6, fontweight='bold',
                color='#1565C0', transform=ax_txt.transAxes, va='top')
    ax_txt.text(0.28, y, tokens_logit_str, fontsize=5.5,
                transform=ax_txt.transAxes, va='top', fontfamily='monospace')

    # Border
    border = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor='none', edgecolor='#CCCCCC',
        linewidth=0.5, transform=fig.transFigure
    )
    fig.patches.append(border)

    plt.savefig(output_path, format='pdf', dpi=200, facecolor='white',
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading PixMoCap dataset...")
    dataset = PixMoCap(split="validation", mode="captions")

    output_dir = BASE_DIR / "paper" / "figures" / "random_method_comparison"
    metadata_dir = BASE_DIR / "analysis_results" / "random_method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Layers to sample: [0, 8, 16, 24, N-1]
    # N-1 is 31 for OLMo/LLaMA, 27 for Qwen2
    # We'll use layer 31 for models that have it, 27 otherwise
    LAYER_GROUPS = {
        0: "Layer 0 (input)",
        8: "Layer 8 (early-mid)",
        16: "Layer 16 (middle)",
        24: "Layer 24 (late)",
        'final': "Final layer"  # Will be 31 or 27 depending on model
    }

    examples_per_layer = 4
    all_examples = []

    for layer_key in [0, 8, 16, 24, 'final']:
        print(f"\n=== Processing layer: {layer_key} ===")

        example_num = 0
        attempts = 0
        max_attempts = 50  # Prevent infinite loops

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

            # Find available images (those present in contextual data which has fewer images)
            available_images = []
            for result in ctx_data.get('results', []):
                available_images.append(result.get('image_idx'))

            if not available_images:
                continue

            # Randomly select an image
            img_idx = random.choice(available_images)

            # Randomly select a patch (avoiding edges which may be padding)
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

            print(f"  Example {example_num}: {model.display_name}, layer {layer}")
            print(f"    Image {img_idx}, patch ({patch_row}, {patch_col})")

            # Load and preprocess image
            example_data = dataset.get(img_idx, random)
            image_path = example_data['image']
            if not Path(image_path).exists():
                print(f"    Skipping - image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')
            img_array = preprocess_and_draw_bbox(image, model, patch_row, patch_col)

            # Create PDF - use "final" for consistency across models with different layer counts
            layer_str = str(layer) if layer_key != 'final' else "final"
            pdf_name = f"layer{layer_str}_example{example_num}.pdf"
            pdf_path = output_dir / pdf_name

            create_example_pdf(img_array, model, layer,
                             emb_results, logit_results, ctx_results,
                             pdf_path, example_num)

            print(f"    Created: {pdf_path.name}")

            # Record metadata
            all_examples.append({
                'layer': layer,
                'layer_key': str(layer_key),
                'example_num': example_num,
                'model': model.display_name,
                'checkpoint': model.checkpoint_name,
                'image_idx': img_idx,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'pdf_name': pdf_name,
                'embedding_lens_top3': [(t[0], float(t[1])) for t in emb_results[:3]] if emb_results else [],
                'logit_lens_top3': [(t[0], float(t[1])) for t in logit_results[:3]] if logit_results else [],
                'latent_lens_top3': [(t[0], float(t[1]), t[2]) for t in ctx_results[:3]] if ctx_results else [],
            })

            example_num += 1  # Successfully created this example

    # Save metadata
    metadata = {
        'seed': 42,
        'num_examples': len(all_examples),
        'examples_per_layer': examples_per_layer,
        'layers': [0, 8, 16, 24, 'final'],
        'num_models': len(MODELS),
        'examples': all_examples
    }

    metadata_path = metadata_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Created {len(all_examples)} example PDFs")
    print(f"PDFs saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
