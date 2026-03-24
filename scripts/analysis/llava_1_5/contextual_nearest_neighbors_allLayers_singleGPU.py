#!/usr/bin/env python3
"""
LatentLens (Contextual Nearest Neighbors) for LLaVA-1.5-7B across all layers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/llava_1_5/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --contextual-dir molmo_data/contextual_llm_embeddings_vg/llava-hf_llava-1.5-7b-hf \
        --visual-layer 0,1,2,4,8,16,24,30,31 --num-images 100
"""

import argparse
import gc
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import LlavaForConditionalGeneration, AutoProcessor

from preprocessing import (
    preprocess_image_llava, GRID_H, GRID_W, NUM_VISION_TOKENS, MODEL_NAME,
)
from nearest_neighbors import find_vision_token_range

try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False


def find_available_layers(contextual_dir):
    contextual_path = Path(contextual_dir)
    layers = []
    for layer_dir in contextual_path.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            cache_file = layer_dir / "embeddings_cache.pt"
            if cache_file.exists():
                layer_idx = int(layer_dir.name.split("_")[1])
                layers.append(layer_idx)
    return sorted(layers)


def extract_vision_features_all_layers(model, processor, image, prompt, visual_layers, device):
    """Extract normalized vision features from all requested layers."""
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    vision_start, vision_end, num_vision = find_vision_token_range(inputs['input_ids'])

    if num_vision == 0:
        raise ValueError("No vision tokens found")

    features_by_layer = {}
    for layer_idx in visual_layers:
        if layer_idx >= len(hidden_states):
            continue
        hs = hidden_states[layer_idx]
        vision_features = hs[:, vision_start:vision_end, :].squeeze(0)
        features_by_layer[layer_idx] = F.normalize(vision_features, dim=-1).float()

    metadata = {
        'num_vision_tokens': num_vision,
        'hidden_dim': int(hidden_states[0].shape[-1]),
        'grid_info': {'height': GRID_H, 'width': GRID_W},
    }

    del hidden_states, outputs
    torch.cuda.empty_cache()
    return features_by_layer, metadata


def main():
    parser = argparse.ArgumentParser(description="LatentLens for LLaVA-1.5-7B")
    parser.add_argument("--contextual-dir", type=str, required=True)
    parser.add_argument("--visual-layer", type=str, default="0,1,2,4,8,16,24,30,31")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                       default="analysis_results/contextual_nearest_neighbors/llava_1_5")
    args = parser.parse_args()

    device = torch.device("cuda")
    visual_layers = [int(l.strip()) for l in args.visual_layer.split(",")]
    ctx_layers = find_available_layers(args.contextual_dir)

    if not ctx_layers:
        print(f"ERROR: No contextual caches found in {args.contextual_dir}")
        return

    print("=" * 70)
    print("LLAVA-1.5 LATENTLENS (CONTEXTUAL NN)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Visual layers: {visual_layers}")
    print(f"Contextual layers: {ctx_layers}")
    print(f"Images: {args.num_images}, Top-k: {args.top_k}")
    print()

    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Preload images
    if not HAVE_PIXMOCAP:
        print("ERROR: PixMoCap dataset required")
        return
    dataset = PixMoCap(split=args.split, mode="captions")

    cached_images = []
    for img_idx in range(args.num_images):
        example = dataset.get(img_idx, np.random)
        image = Image.open(example["image"]).convert('RGB')
        image = preprocess_image_llava(image, target_size=336, force_square=True)
        caption = ""
        if "message_list" in example and len(example["message_list"]) > 0:
            caption = example["message_list"][0].get("text", "")
        cached_images.append({'image': image, 'caption': caption})
    print(f"Preloaded {len(cached_images)} images")

    output_dir = Path(args.output_dir) / MODEL_NAME.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage: candidates[visual_layer][image_idx][ctx_layer] = (vals, idxs)
    candidates = {vl: {img: {} for img in range(args.num_images)} for vl in visual_layers}
    shape_info = None
    ctx_metadata_cache = {}
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"

    # Main loop: for each contextual cache
    for ctx_idx, ctx_layer in enumerate(ctx_layers):
        print(f"\nContextual layer {ctx_layer} ({ctx_idx+1}/{len(ctx_layers)})")
        cache_file = Path(args.contextual_dir) / f"layer_{ctx_layer}" / "embeddings_cache.pt"
        cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
        embeddings = cache_data['embeddings'].to(device)
        metadata = cache_data['metadata']
        embeddings_norm = F.normalize(embeddings.float(), dim=-1)
        ctx_metadata_cache[ctx_layer] = metadata
        del cache_data
        print(f"  Cache: {embeddings.shape[0]} embeddings")

        for img_idx in range(args.num_images):
            if img_idx % 20 == 0:
                print(f"  Image {img_idx+1}/{args.num_images}", flush=True)

            features_by_layer, meta = extract_vision_features_all_layers(
                model, processor, cached_images[img_idx]['image'],
                prompt, visual_layers, device
            )
            if shape_info is None:
                shape_info = meta

            for vl in visual_layers:
                if vl not in features_by_layer:
                    continue
                sims = torch.matmul(features_by_layer[vl], embeddings_norm.T)
                top_vals, top_idxs = torch.topk(sims, k=args.top_k, dim=-1)
                candidates[vl][img_idx][ctx_layer] = (top_vals.cpu(), top_idxs.cpu())
                del sims

            del features_by_layer
            gc.collect()
            torch.cuda.empty_cache()

        del embeddings, embeddings_norm
        gc.collect()
        torch.cuda.empty_cache()

    # Build results
    print("\nBuilding results...")
    grid_info = shape_info['grid_info']
    grid_h, grid_w = grid_info['height'], grid_info['width']

    for vl in visual_layers:
        results = []
        for img_idx in range(args.num_images):
            all_vals = torch.stack([candidates[vl][img_idx][cl][0] for cl in ctx_layers])
            all_idxs = torch.stack([candidates[vl][img_idx][cl][1] for cl in ctx_layers])
            num_patches = all_vals.shape[1]

            patches = []
            for patch_idx in range(num_patches):
                patch_vals = all_vals[:, patch_idx, :].flatten()
                patch_idxs = all_idxs[:, patch_idx, :].flatten()
                ctx_ids = torch.arange(len(ctx_layers)).unsqueeze(1).expand(-1, args.top_k).flatten()

                global_vals, global_pos = torch.topk(patch_vals, k=args.top_k)

                nearest = []
                for k in range(args.top_k):
                    pos = global_pos[k].item()
                    ci = ctx_ids[pos].item()
                    emb_idx = patch_idxs[pos].item()
                    cl = ctx_layers[ci]
                    meta = ctx_metadata_cache[cl][emb_idx]
                    nearest.append({
                        'token_str': meta['token_str'],
                        'token_id': meta['token_id'],
                        'caption': meta['caption'],
                        'position': meta['position'],
                        'similarity': global_vals[k].item(),
                        'contextual_layer': cl,
                    })

                row = patch_idx // grid_w
                col = patch_idx % grid_w
                patches.append({
                    "patch_idx": patch_idx,
                    "patch_row": row,
                    "patch_col": col,
                    "nearest_contextual_neighbors": nearest,
                })

            results.append({
                "image_idx": img_idx,
                "ground_truth_caption": cached_images[img_idx]['caption'],
                "num_vision_tokens": num_patches,
                "hidden_dim": shape_info['hidden_dim'],
                "grid_info": grid_info,
                "visual_layer": vl,
                "patches": patches,
            })

        out_file = output_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
        out_data = {
            'model_name': MODEL_NAME,
            'visual_layer': vl,
            'contextual_layers_searched': ctx_layers,
            'num_images': len(results),
            'top_k': args.top_k,
            'results': results,
        }
        with open(out_file, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"  Saved visual layer {vl}: {out_file}")

    print(f"\nDONE! Results: {output_dir}")


if __name__ == "__main__":
    main()
