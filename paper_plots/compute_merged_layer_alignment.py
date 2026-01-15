#!/usr/bin/env python3
"""
Compute layer alignment data with LLM layer 0 included.

Merges static NN results (Input Embedding Matrix = layer 0) with
contextual NN results (LatentLens = layers 1+).

For each patch: take 5 static NNs + 5 contextual NNs, sort by similarity,
take top-5, count which layer each came from.
"""

import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
STATIC_NN_DIR = REPO_ROOT / "analysis_results" / "nearest_neighbors"
CONTEXTUAL_NN_DIR = REPO_ROOT / "analysis_results" / "contextual_nearest_neighbors"

# Models to process
MODELS = [
    ("llama3-8b", "dinov2-large-336"),
    ("llama3-8b", "siglip"),
    ("llama3-8b", "vit-l-14-336"),
    ("olmo-7b", "dinov2-large-336"),
    ("olmo-7b", "siglip"),
    ("olmo-7b", "vit-l-14-336"),
    ("qwen2-7b", "dinov2-large-336"),
    ("qwen2-7b", "siglip"),
    ("qwen2-7b", "vit-l-14-336"),
]

# Vision layers per LLM type
VISION_LAYERS_DEFAULT = [0, 1, 2, 4, 8, 16, 24, 30, 31]
VISION_LAYERS_QWEN = [0, 1, 2, 4, 8, 16, 24, 26, 27]

NUM_IMAGES = 100  # Use first 100 images


def get_vision_layers(llm):
    return VISION_LAYERS_QWEN if 'qwen' in llm else VISION_LAYERS_DEFAULT


def find_static_nn_dir(llm, encoder):
    """Find static NN directory for a model."""
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        pattern = f"*{llm}_{encoder}_seed10_step12000-unsharded"
    else:
        pattern = f"*{llm}_{encoder}_step12000-unsharded"

    matches = list(STATIC_NN_DIR.glob(pattern))
    matches = [m for m in matches if 'lite' not in str(m) and m.is_dir()]
    return matches[0] if matches else None


def find_contextual_nn_dir(llm, encoder):
    """Find contextual NN directory for a model."""
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        pattern = f"*{llm}_{encoder}_seed10_step12000-unsharded"
    else:
        pattern = f"*{llm}_{encoder}_step12000-unsharded"

    matches = list(CONTEXTUAL_NN_DIR.glob(pattern))
    matches = [m for m in matches if 'lite' not in str(m) and m.is_dir()]
    return matches[0] if matches else None


def load_static_nn_for_vision_layer(static_dir, vision_layer):
    """
    Load static NN results for a specific vision layer.
    Returns: {image_idx: {(chunk_idx, patch_idx): [nn1, nn2, ...]}}
    """
    # Static NN files are named by the vision/LLM layer
    static_file = static_dir / f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{vision_layer}.json"
    if not static_file.exists():
        return None

    with open(static_file) as f:
        data = json.load(f)

    results = {}
    images = data.get('splits', {}).get('validation', {}).get('images', [])

    for img in images[:NUM_IMAGES]:
        image_idx = img['image_idx']
        results[image_idx] = {}

        for chunk_idx, chunk in enumerate(img.get('chunks', [])):
            for patch in chunk.get('patches', []):
                patch_idx = patch['patch_idx']
                # Add layer=0 to each NN
                nns = []
                for nn in patch.get('nearest_neighbors', []):
                    nns.append({
                        'similarity': nn['similarity'],
                        'layer': 0,  # Static NN = layer 0
                        'token': nn.get('token', '')
                    })
                results[image_idx][(chunk_idx, patch_idx)] = nns

    return results


def load_contextual_nn_for_vision_layer(contextual_dir, vision_layer):
    """
    Load contextual NN results for a specific vision layer.
    Returns: {image_idx: {(chunk_idx, patch_idx): [nn1, nn2, ...]}}
    """
    ctx_file = contextual_dir / f"contextual_neighbors_visual{vision_layer}_allLayers.json"
    if not ctx_file.exists():
        return None

    with open(ctx_file) as f:
        data = json.load(f)

    results = {}

    for r in data.get('results', [])[:NUM_IMAGES]:
        image_idx = r['image_idx']
        results[image_idx] = {}

        for chunk_idx, chunk in enumerate(r.get('chunks', [])):
            for patch in chunk.get('patches', []):
                patch_idx = patch['patch_idx']
                nns = []
                for nn in patch.get('nearest_contextual_neighbors', []):
                    nns.append({
                        'similarity': nn['similarity'],
                        'layer': nn['contextual_layer'],
                        'token': nn.get('token_str', '')
                    })
                results[image_idx][(chunk_idx, patch_idx)] = nns

    return results


def merge_and_count_layers(static_results, contextual_results):
    """
    Merge static and contextual NNs, take top-5, count layer distribution.
    Returns: {layer: count}
    """
    layer_counts = defaultdict(int)

    # Find overlapping images
    common_images = set(static_results.keys()) & set(contextual_results.keys())

    for image_idx in common_images:
        static_patches = static_results[image_idx]
        ctx_patches = contextual_results[image_idx]

        # Find overlapping patches
        common_patches = set(static_patches.keys()) & set(ctx_patches.keys())

        for patch_key in common_patches:
            static_nns = static_patches[patch_key]
            ctx_nns = ctx_patches[patch_key]

            # Merge all NNs
            all_nns = static_nns + ctx_nns

            # Sort by similarity (descending) and take top 5
            all_nns.sort(key=lambda x: x['similarity'], reverse=True)
            top5 = all_nns[:5]

            # Count layers
            for nn in top5:
                layer_counts[nn['layer']] += 1

    return dict(layer_counts)


def compute_model_alignment(llm, encoder):
    """Compute layer alignment for one model."""
    print(f"\n{'='*60}")
    print(f"Processing {llm} + {encoder}")
    print('='*60)

    static_dir = find_static_nn_dir(llm, encoder)
    contextual_dir = find_contextual_nn_dir(llm, encoder)

    if not static_dir:
        print(f"  ERROR: Static NN directory not found")
        return None
    if not contextual_dir:
        print(f"  ERROR: Contextual NN directory not found")
        return None

    print(f"  Static NN dir: {static_dir.name}")
    print(f"  Contextual NN dir: {contextual_dir.name}")

    vision_layers = get_vision_layers(llm)
    alignment_data = {}

    for vl in vision_layers:
        print(f"\n  Vision layer {vl}:", end=" ", flush=True)

        static_results = load_static_nn_for_vision_layer(static_dir, vl)
        if static_results is None:
            print("static NN not found")
            continue

        ctx_results = load_contextual_nn_for_vision_layer(contextual_dir, vl)
        if ctx_results is None:
            print("contextual NN not found")
            continue

        layer_counts = merge_and_count_layers(static_results, ctx_results)
        alignment_data[str(vl)] = layer_counts

        # Print summary
        total = sum(layer_counts.values())
        layer0_pct = layer_counts.get(0, 0) / total * 100 if total > 0 else 0
        print(f"{total} patches, layer0={layer0_pct:.1f}%")

    return alignment_data


# =============================================================================
# QWEN2-VL (off-the-shelf model, different file structure)
# =============================================================================

QWEN2VL_STATIC_DIR = REPO_ROOT / "analysis_results" / "nearest_neighbors" / "qwen2_vl" / "Qwen_Qwen2-VL-7B-Instruct"
QWEN2VL_CONTEXTUAL_DIR = REPO_ROOT / "analysis_results" / "contextual_nearest_neighbors" / "qwen2_vl" / "Qwen_Qwen2-VL-7B-Instruct"


def load_qwen2vl_static_nn(vision_layer):
    """Load Qwen2-VL static NN for a vision layer."""
    static_file = QWEN2VL_STATIC_DIR / f"nearest_neighbors_layer{vision_layer}_topk5.json"
    if not static_file.exists():
        return None

    with open(static_file) as f:
        data = json.load(f)

    results = {}
    for r in data.get('results', [])[:NUM_IMAGES]:
        image_idx = r['image_idx']
        results[image_idx] = {}

        for patch in r.get('patches', []):
            patch_idx = patch['patch_idx']
            nns = []
            for nn in patch.get('nearest_neighbors', []):
                nns.append({
                    'similarity': nn['similarity'],
                    'layer': 0,
                    'token': nn.get('token', '')
                })
            results[image_idx][patch_idx] = nns

    return results


def load_qwen2vl_contextual_nn(vision_layer):
    """Load Qwen2-VL contextual NN for a vision layer."""
    ctx_file = QWEN2VL_CONTEXTUAL_DIR / f"contextual_neighbors_visual{vision_layer}_allLayers.json"
    if not ctx_file.exists():
        return None

    with open(ctx_file) as f:
        data = json.load(f)

    results = {}
    for r in data.get('results', [])[:NUM_IMAGES]:
        image_idx = r['image_idx']
        results[image_idx] = {}

        # Qwen2-VL has patches directly (no chunks)
        for patch in r.get('patches', []):
            patch_idx = patch['patch_idx']
            nns = []
            for nn in patch.get('nearest_contextual_neighbors', []):
                nns.append({
                    'similarity': nn['similarity'],
                    'layer': nn['contextual_layer'],
                    'token': nn.get('token_str', '')
                })
            results[image_idx][patch_idx] = nns

    return results


def compute_qwen2vl_alignment():
    """Compute layer alignment for Qwen2-VL."""
    print(f"\n{'='*60}")
    print("Processing Qwen2-VL-7B-Instruct")
    print('='*60)

    if not QWEN2VL_STATIC_DIR.exists():
        print(f"  ERROR: Static NN dir not found: {QWEN2VL_STATIC_DIR}")
        return None
    if not QWEN2VL_CONTEXTUAL_DIR.exists():
        print(f"  ERROR: Contextual NN dir not found: {QWEN2VL_CONTEXTUAL_DIR}")
        return None

    vision_layers = VISION_LAYERS_QWEN
    alignment_data = {}

    for vl in vision_layers:
        print(f"\n  Vision layer {vl}:", end=" ", flush=True)

        static_results = load_qwen2vl_static_nn(vl)
        if static_results is None:
            print("static NN not found")
            continue

        ctx_results = load_qwen2vl_contextual_nn(vl)
        if ctx_results is None:
            print("contextual NN not found")
            continue

        # Merge using patch_idx as key (no chunk_idx for Qwen2-VL)
        layer_counts = defaultdict(int)
        common_images = set(static_results.keys()) & set(ctx_results.keys())

        for image_idx in common_images:
            static_patches = static_results[image_idx]
            ctx_patches = ctx_results[image_idx]
            common_patch_ids = set(static_patches.keys()) & set(ctx_patches.keys())

            for patch_idx in common_patch_ids:
                all_nns = static_patches[patch_idx] + ctx_patches[patch_idx]
                all_nns.sort(key=lambda x: x['similarity'], reverse=True)
                for nn in all_nns[:5]:
                    layer_counts[nn['layer']] += 1

        alignment_data[vl] = dict(layer_counts)

        total = sum(layer_counts.values())
        layer0_pct = layer_counts.get(0, 0) / total * 100 if total > 0 else 0
        print(f"{total} patches, layer0={layer0_pct:.1f}%")

    return alignment_data


def main():
    print("Computing merged layer alignment (static + contextual NN)")
    print(f"Using first {NUM_IMAGES} images per model/vision-layer")

    all_alignment = {}

    for llm, encoder in MODELS:
        model_key = f"{llm}+{encoder}"
        alignment = compute_model_alignment(llm, encoder)
        if alignment:
            all_alignment[model_key] = alignment

    # Save to data.json
    data_json_path = Path(__file__).parent / "data.json"

    if data_json_path.exists():
        with open(data_json_path) as f:
            data = json.load(f)
    else:
        data = {}

    data['layer_alignment'] = all_alignment

    # Also compute Qwen2-VL alignment
    qwen2vl_alignment = compute_qwen2vl_alignment()
    if qwen2vl_alignment:
        data['qwen2vl_layer_alignment'] = qwen2vl_alignment

    with open(data_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved layer_alignment to {data_json_path}")
    print(f"Models processed: {len(all_alignment)}")
    if qwen2vl_alignment:
        print("Also updated qwen2vl_layer_alignment")


if __name__ == "__main__":
    main()
