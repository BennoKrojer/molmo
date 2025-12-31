#!/usr/bin/env python3
"""
Contextual nearest neighbors for Qwen2-VL across all layers (single GPU version).

This is the Qwen2-VL equivalent of scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py

Algorithm:
    1. Load Qwen2-VL model
    2. Preload all images
    3. For each contextual cache:
        Load cache
        For each image:
            Forward pass → get all visual layer features
            Search against cache → store candidates
        Unload cache
    4. Merge candidates → pick global top-k
    5. Save results

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --contextual-dir molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-VL-7B-Instruct \
        --visual-layer 0,1,8,16,24,27 --num-images 100
"""

import argparse
import gc
import json
import math
import os
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Import dataset
try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available. Will use synthetic images.")


IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> token ID in Qwen2-VL


def find_available_layers(contextual_dir):
    """Find all layer directories with caches."""
    contextual_path = Path(contextual_dir)
    layers = []
    for layer_dir in contextual_path.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            cache_file = layer_dir / "embeddings_cache.pt"
            if cache_file.exists():
                layer_idx = int(layer_dir.name.split("_")[1])
                layers.append(layer_idx)
    return sorted(layers)


def find_image_token_positions(input_ids):
    """
    Find positions of vision tokens in the input sequence.
    Returns: (start_idx, end_idx, num_vision_tokens)
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    
    if len(input_ids.shape) == 2:
        input_ids = input_ids[0]
    
    image_positions = np.where(input_ids == IMAGE_PAD_TOKEN_ID)[0]
    
    if len(image_positions) == 0:
        return None, None, 0
    
    start_idx = int(image_positions[0])
    end_idx = int(image_positions[-1]) + 1
    num_vision_tokens = len(image_positions)
    
    return start_idx, end_idx, num_vision_tokens


def patch_idx_to_row_col(patch_idx, num_patches):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        # Non-square: estimate
        grid_size = int(math.ceil(math.sqrt(num_patches)))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def preload_images(dataset, num_images, max_size=1024, force_square=False):
    """Preload images from dataset.
    
    Args:
        force_square: If True, center-crop images to square before processing.
                     This ensures consistent grid dimensions across all images.
    """
    cached_data = []
    
    for img_idx in range(num_images):
        if img_idx % 10 == 0:
            print(f"    {img_idx}/{num_images}...", flush=True)
        
        if dataset is not None:
            example = dataset.get(img_idx, np.random)
            image = Image.open(example["image"]).convert('RGB')
            
            caption = ""
            if "message_list" in example and len(example["message_list"]) > 0:
                caption = example["message_list"][0].get("text", "")
        else:
            # Synthetic image
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
            color = colors[img_idx % len(colors)]
            image = Image.new('RGB', (224, 224), color=color)
            caption = f"A solid {color} image"
        
        # Center-crop to square if requested (for consistent grid dimensions)
        if force_square and image.size[0] != image.size[1]:
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize if too large
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        cached_data.append({
            'image': image,
            'caption': caption
        })
    
    return cached_data


def extract_vision_features_all_layers(model, processor, image, prompt, visual_layers, device):
    """
    Extract vision token features from specified layers of Qwen2-VL.
    
    Returns:
        features_by_layer: dict[layer_idx] -> tensor [num_vision, hidden_dim] (normalized)
        metadata: dict with shape info
    """
    # Prepare inputs
    text_with_image = f"<|image_pad|>{prompt}"
    
    model_inputs = processor(
        images=[image],
        text=text_with_image,
        return_tensors="pt"
    )
    model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in model_inputs.items()}
    
    # Find vision token positions
    input_ids = model_inputs['input_ids']
    vision_start, vision_end, num_vision_tokens = find_image_token_positions(input_ids)
    
    if num_vision_tokens == 0:
        raise ValueError("No vision tokens found in input sequence")
    
    # Run forward pass
    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            outputs = model(**model_inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    # Extract vision tokens from each requested layer
    features_by_layer = {}
    for layer_idx in visual_layers:
        if layer_idx >= len(hidden_states):
            print(f"Warning: layer {layer_idx} >= {len(hidden_states)}, using last layer")
            layer_idx = len(hidden_states) - 1
        
        hs = hidden_states[layer_idx]
        vision_features = hs[:, vision_start:vision_end, :].squeeze(0)  # [num_vision, hidden_dim]
        # Normalize for cosine similarity
        features_by_layer[layer_idx] = torch.nn.functional.normalize(vision_features, dim=-1).float()
    
    # Get grid info
    grid_info = None
    if 'image_grid_thw' in model_inputs:
        grid = model_inputs['image_grid_thw']
        if isinstance(grid, torch.Tensor):
            t, h, w = grid[0].tolist()
            grid_info = {'temporal': int(t), 'height': int(h), 'width': int(w)}
    
    metadata = {
        'num_vision_tokens': num_vision_tokens,
        'vision_start': vision_start,
        'vision_end': vision_end,
        'hidden_dim': int(hidden_states[0].shape[-1]),
        'num_layers': len(hidden_states),
        'grid_info': grid_info
    }
    
    # Clean up
    del hidden_states, outputs
    torch.cuda.empty_cache()
    
    return features_by_layer, metadata


def main():
    parser = argparse.ArgumentParser(description="Contextual nearest neighbors for Qwen2-VL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--contextual-dir", type=str, required=True,
                       help="Directory with contextual embeddings (e.g., molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-VL-7B-Instruct)")
    parser.add_argument("--fixed-resolution", type=int, default=448,
                       help="Fixed image resolution (default: 448 for ~256 tokens like Molmo). Set to 0 for dynamic resolution.")
    parser.add_argument("--force-square", action="store_true",
                       help="Center-crop images to square before processing (ensures consistent 16x16 grid for viewer)")
    parser.add_argument("--visual-layer", type=str, default="0,1,8,16,24,27",
                       help="Visual layers to extract (comma-separated)")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to use")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of nearest neighbors per patch")
    parser.add_argument("--output-dir", type=str, default="analysis_results/contextual_nearest_neighbors",
                       help="Output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    visual_layers = [int(l.strip()) for l in args.visual_layer.split(",")]
    ctx_layers = find_available_layers(args.contextual_dir)
    
    if not ctx_layers:
        print(f"ERROR: No contextual caches found in {args.contextual_dir}")
        print("Make sure the merge completed and caches were built.")
        return
    
    print("=" * 70)
    print("QWEN2-VL CONTEXTUAL NEAREST NEIGHBORS")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Contextual dir: {args.contextual_dir}")
    print(f"Visual layers: {visual_layers}")
    print(f"Contextual layers available: {ctx_layers}")
    print(f"Images: {args.num_images}")
    print(f"Top-k: {args.top_k}")
    print()
    
    # ===== LOAD MODEL =====
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    load_start = time.time()
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set fixed resolution if specified (for consistent token counts across images)
    if args.fixed_resolution > 0:
        fixed_pixels = args.fixed_resolution * args.fixed_resolution
        processor.image_processor.min_pixels = fixed_pixels
        processor.image_processor.max_pixels = fixed_pixels
        # Qwen2-VL: 14x14 patches with 2x2 spatial merger = 28 pixels per token
        expected_tokens = (args.fixed_resolution // 28) ** 2
        print(f"✓ Fixed resolution: {args.fixed_resolution}x{args.fixed_resolution} (~{expected_tokens} vision tokens, {int(math.sqrt(expected_tokens))}x{int(math.sqrt(expected_tokens))} grid)")
    else:
        print(f"✓ Using dynamic resolution (variable token counts)")
    
    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")
    print()
    
    # ===== LOAD DATASET =====
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    if HAVE_PIXMOCAP:
        dataset = PixMoCap(split=args.split, mode="captions")
        print(f"✓ Using PixMoCap {args.split} split")
    else:
        dataset = None
        print("✓ Using synthetic images (PixMoCap not available)")
    print()
    
    # ===== PRELOAD IMAGES =====
    print("=" * 70)
    print("PRELOADING IMAGES")
    print("=" * 70)
    preload_start = time.time()
    
    print(f"  Loading {args.num_images} images...", flush=True)
    cached_images = preload_images(dataset, args.num_images, force_square=args.force_square)
    if args.force_square:
        print(f"  (images center-cropped to square for consistent grid)")
    
    preload_time = time.time() - preload_start
    print(f"✓ Images preloaded in {preload_time:.1f}s ({args.num_images/preload_time:.1f} img/s)")
    print()
    
    # Output setup
    model_name_safe = args.model_name.replace("/", "_")
    output_dir = Path(args.output_dir) / model_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== STORAGE FOR CANDIDATES =====
    # candidates[visual_layer][image_idx][ctx_layer] = (top_vals, top_idxs)
    candidates = {vl: {img: {} for img in range(args.num_images)} for vl in visual_layers}
    shape_info = None
    ctx_metadata_cache = {}
    
    total_start = time.time()
    prompt = "Describe this image in detail."
    
    # ===== MAIN LOOP: For each contextual cache =====
    for ctx_idx, ctx_layer in enumerate(ctx_layers):
        print("=" * 70)
        print(f"CONTEXTUAL LAYER {ctx_layer} ({ctx_idx + 1}/{len(ctx_layers)})")
        print("=" * 70)
        
        # Load cache
        print(f"  Loading cache...", end=" ", flush=True)
        cache_start = time.time()
        cache_file = Path(args.contextual_dir) / f"layer_{ctx_layer}" / "embeddings_cache.pt"
        cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
        embeddings = cache_data['embeddings'].to(device)
        metadata = cache_data['metadata']
        embeddings_norm = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        ctx_metadata_cache[ctx_layer] = metadata
        del cache_data
        print(f"done ({time.time() - cache_start:.1f}s, {embeddings.shape[0]} embeddings)")
        
        # Process all images
        print(f"  Processing {args.num_images} images...")
        img_start = time.time()
        
        for img_idx in range(args.num_images):
            if img_idx % 10 == 0 or img_idx == args.num_images - 1:
                elapsed = time.time() - img_start
                rate = (img_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"    Image {img_idx + 1}/{args.num_images} ({rate:.1f} img/s)", flush=True)
            
            # Extract vision features
            features_by_layer, meta = extract_vision_features_all_layers(
                model, processor, cached_images[img_idx]['image'], prompt, visual_layers, device
            )
            
            if shape_info is None:
                shape_info = meta
            
            # Search against this cache for each visual layer
            for vl in visual_layers:
                if vl not in features_by_layer:
                    continue
                feats = features_by_layer[vl]
                similarity = torch.matmul(feats, embeddings_norm.T)
                top_vals, top_idxs = torch.topk(similarity, k=args.top_k, dim=-1)
                candidates[vl][img_idx][ctx_layer] = (top_vals.cpu(), top_idxs.cpu())
                del similarity
            
            del features_by_layer
            gc.collect()
            torch.cuda.empty_cache()
        
        img_time = time.time() - img_start
        print(f"  ✓ Done: {args.num_images} images in {img_time:.1f}s ({args.num_images/img_time:.1f} img/s)")
        
        # Unload cache
        del embeddings, embeddings_norm
        gc.collect()
        torch.cuda.empty_cache()
    
    search_time = time.time() - total_start
    print()
    print(f"✓ All caches processed in {search_time:.1f}s ({search_time/60:.1f} min)")
    print()
    
    # ===== BUILD RESULTS =====
    print("=" * 70)
    print("BUILDING RESULTS")
    print("=" * 70)
    
    build_start = time.time()
    num_patches = shape_info['num_vision_tokens']
    hidden_dim = shape_info['hidden_dim']
    grid_info = shape_info['grid_info']
    
    # Determine grid size for visualization
    if grid_info:
        grid_h = grid_info['height']
        grid_w = grid_info['width']
    else:
        grid_size = int(math.ceil(math.sqrt(num_patches)))
        grid_h = grid_w = grid_size
    
    all_results = {vl: [] for vl in visual_layers}
    
    for img_idx in range(args.num_images):
        if img_idx % 20 == 0:
            print(f"  Image {img_idx + 1}/{args.num_images}...", flush=True)
        
        for vl in visual_layers:
            # Stack candidates from all contextual layers
            # all_vals: [num_ctx_layers, num_patches, top_k]
            # all_idxs: [num_ctx_layers, num_patches, top_k]
            all_vals = torch.stack([candidates[vl][img_idx][cl][0] for cl in ctx_layers])
            all_idxs = torch.stack([candidates[vl][img_idx][cl][1] for cl in ctx_layers])
            
            # Get actual number of patches for THIS image (varies with resolution)
            img_num_patches = all_vals.shape[1]
            
            patches_results = []
            for patch_idx in range(img_num_patches):
                # Get candidates for this patch across all contextual layers
                patch_vals = all_vals[:, patch_idx, :]  # [num_ctx_layers, top_k]
                patch_idxs = all_idxs[:, patch_idx, :]  # [num_ctx_layers, top_k]
                
                # Flatten and find global top-k
                flat_vals = patch_vals.flatten()
                flat_idxs = patch_idxs.flatten()
                ctx_ids = torch.arange(len(ctx_layers)).unsqueeze(1).expand(-1, args.top_k).flatten()
                
                global_top_vals, global_top_pos = torch.topk(flat_vals, k=args.top_k)
                
                nearest = []
                for k_idx in range(args.top_k):
                    pos = global_top_pos[k_idx].item()
                    sim = global_top_vals[k_idx].item()
                    ctx_idx = ctx_ids[pos].item()
                    emb_idx = flat_idxs[pos].item()
                    
                    ctx_layer = ctx_layers[ctx_idx]
                    meta = ctx_metadata_cache[ctx_layer][emb_idx]
                    nearest.append({
                        'token_str': meta['token_str'],
                        'token_id': meta['token_id'],
                        'caption': meta['caption'],
                        'position': meta['position'],
                        'similarity': sim,
                        'contextual_layer': ctx_layer
                    })
                
                row, col = patch_idx_to_row_col(patch_idx, img_num_patches)
                patches_results.append({
                    "patch_idx": patch_idx,
                    "patch_row": row,
                    "patch_col": col,
                    "nearest_contextual_neighbors": nearest
                })
            
            all_results[vl].append({
                "image_idx": img_idx,
                "ground_truth_caption": cached_images[img_idx]['caption'],
                "num_vision_tokens": img_num_patches,
                "hidden_dim": hidden_dim,
                "grid_info": grid_info,
                "visual_layer": vl,
                "patches": patches_results
            })
    
    build_time = time.time() - build_start
    print(f"✓ Results built in {build_time:.1f}s")
    print()
    
    # ===== SAVE =====
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    total_time = preload_time + search_time + build_time
    
    for vl in visual_layers:
        output_file = output_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
        
        output_data = {
            'model_name': args.model_name,
            'contextual_dir': args.contextual_dir,
            'visual_layer': vl,
            'contextual_layers_used': ctx_layers,
            'split': args.split,
            'num_images': args.num_images,
            'top_k': args.top_k,
            'processing_time_seconds': total_time,
            'results': all_results[vl]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ {output_file.name}")
    
    print()
    print("=" * 70)
    print("✓ DONE!")
    print(f"  Preload: {preload_time:.1f}s | Search: {search_time:.1f}s | Build: {build_time:.1f}s")
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


