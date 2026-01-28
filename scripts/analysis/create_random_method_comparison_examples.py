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


def draw_bbox_on_original(image: Image.Image, model: ModelConfig,
                          patch_row: int, patch_col: int) -> np.ndarray:
    """
    Draw bbox on ORIGINAL image (not preprocessed) for cleaner display.
    Maps patch coordinates from grid to original image dimensions.
    """
    config = VISION_CONFIGS.get(model.vision_encoder, VISION_CONFIGS['vit-l-14-336'])
    grid_size = config['grid']
    method = config['method']

    img_w, img_h = image.size

    # Make a copy to draw on
    display_img = image.copy()

    if method == 'resize_pad':
        # CLIP: patches correspond to padded square, need to map back
        # The model sees a padded square - find where original image sits
        target_size = config['size']
        scale = min(target_size / img_w, target_size / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2

        # Patch coords in padded space
        cell_size = target_size / grid_size
        patch_x = patch_col * cell_size
        patch_y = patch_row * cell_size

        # Map back to original image coords
        x1_orig = (patch_x - offset_x) / scale
        y1_orig = (patch_y - offset_y) / scale
        x2_orig = ((patch_col + 1) * cell_size - offset_x) / scale
        y2_orig = ((patch_row + 1) * cell_size - offset_y) / scale

        # Clamp to image bounds and ensure valid rectangle
        x1 = max(0, int(x1_orig))
        y1 = max(0, int(y1_orig))
        x2 = min(img_w, int(x2_orig))
        y2 = min(img_h, int(y2_orig))

        # Ensure valid rectangle (x2 > x1, y2 > y1)
        if x2 <= x1:
            x2 = min(img_w, x1 + 20)
        if y2 <= y1:
            y2 = min(img_h, y1 + 20)
    else:
        # Squash: direct mapping
        cell_w = img_w / grid_size
        cell_h = img_h / grid_size
        x1 = int(patch_col * cell_w)
        y1 = int(patch_row * cell_h)
        x2 = int((patch_col + 1) * cell_w)
        y2 = int((patch_row + 1) * cell_h)

    # Draw bbox with white outline for contrast, then red
    draw = ImageDraw.Draw(display_img)
    thickness = max(4, min(img_w, img_h) // 60)

    # White outline (outer)
    for i in range(thickness + 2, thickness + 5):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='#FFFFFF')

    # Red rectangle (inner)
    for i in range(thickness):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='#FF0000')

    return np.array(display_img)


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


def clean_token(token: str) -> str:
    """Clean token for display - replace non-printable chars."""
    if not token:
        return "[?]"
    token = token.strip()
    # Check if token has non-ASCII or control characters
    has_cjk = False
    cleaned = []
    for c in token:
        if ord(c) < 32:  # Control characters
            continue
        elif ord(c) > 127:  # Non-ASCII
            # Keep common accented chars, mark CJK/special
            if ord(c) < 0x3000:  # Likely accented Latin
                cleaned.append(c)
            else:
                has_cjk = True
        else:
            cleaned.append(c)
    result = ''.join(cleaned).strip()
    if not result:
        return "[CJK]" if has_cjk else "[?]"
    if has_cjk and len(result) < 3:
        result = result + "[+]"  # Indicate there was more
    return result


def create_example_pdf(image_array: np.ndarray, model: ModelConfig, layer: int,
                       embedding_lens: List, logit_lens: List, latent_lens: List,
                       output_path: Path, example_num: int):
    """
    Create a compact PDF for one example showing image + top-3 from all three methods.
    Optimized for 2x2 grid display in paper appendix.
    """
    # Compact figure - will be displayed at ~4.5cm height in 2x2 grid
    fig = plt.figure(figsize=(2.8, 3.2), facecolor='white')

    # Title - compact, bold
    title = f"{model.display_name}, L{layer}"
    fig.text(0.5, 0.97, title, fontsize=10, fontweight='bold', ha='center', va='top')

    # Image - takes up ~55% of figure height
    ax_img = fig.add_axes([0.04, 0.42, 0.92, 0.52])
    ax_img.imshow(image_array)
    ax_img.axis('off')

    # Text area - compact, fills remaining space
    ax_txt = fig.add_axes([0.04, 0.02, 0.92, 0.38])
    ax_txt.axis('off')
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    y = 0.98

    # LatentLens (ours) - show top phrase only (most informative)
    ax_txt.text(0, y, "LatentLens:", fontsize=8, fontweight='bold',
                color='#2E7D32', va='top')
    y -= 0.18

    if latent_lens and len(latent_lens[0]) >= 3:
        caption = latent_lens[0][2]
        # Smarter truncation - keep full words
        if len(caption) > 38:
            words = caption[:38].rsplit(' ', 1)
            caption = words[0] + "..." if len(words) > 1 else caption[:35] + "..."
        ax_txt.text(0.02, y, f'"{caption}"', fontsize=7.5, style='italic', va='top')
    y -= 0.22

    # EmbeddingLens - clean tokens
    tokens_emb = [clean_token(t[0]) if t else '?' for t in embedding_lens[:3]]
    # Truncate long tokens
    tokens_emb = [t[:12] + '..' if len(t) > 14 else t for t in tokens_emb]
    ax_txt.text(0, y, "EmbeddingLens:", fontsize=8, fontweight='bold',
                color='#C62828', va='top')
    y -= 0.18
    ax_txt.text(0.02, y, "  ".join(tokens_emb), fontsize=7.5,
                va='top', fontfamily='monospace')
    y -= 0.22

    # LogitLens - clean tokens
    tokens_logit = [clean_token(t[0]) if t else '?' for t in logit_lens[:3]]
    tokens_logit = [t[:12] + '..' if len(t) > 14 else t for t in tokens_logit]
    ax_txt.text(0, y, "LogitLens:", fontsize=8, fontweight='bold',
                color='#1565C0', va='top')
    y -= 0.18
    ax_txt.text(0.02, y, "  ".join(tokens_logit), fontsize=7.5,
                va='top', fontfamily='monospace')

    plt.savefig(output_path, format='pdf', dpi=200, facecolor='white',
                bbox_inches='tight', pad_inches=0.01)
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
            img_array = draw_bbox_on_original(image, model, patch_row, patch_col)

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
