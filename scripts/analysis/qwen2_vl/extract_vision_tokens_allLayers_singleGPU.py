#!/usr/bin/env python3
"""
Extract vision tokens across ALL LLM layers in Qwen2-VL (single GPU version).

This script is the Qwen2-VL equivalent of contextual_nearest_neighbors_allLayers_singleGPU.py

Key findings from our investigation:
1. Qwen2-VL uses a "merger" that reduces patches to fewer tokens (e.g., 256 patches → 64 tokens)
2. The <|image_pad|> tokens in input_ids mark vision token positions
3. Positions 0 to (num_image_pad - 1) in hidden states are vision tokens
4. We can extract vision features from all 29 layers (embedding + 28 transformer layers)

This script:
1. Loads Qwen2-VL model
2. Processes images from PixMoCap dataset
3. Extracts vision token features from ALL layers
4. Saves features for later analysis (nearest neighbor search, etc.)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_allLayers_singleGPU.py \
        --num-images 100 --output-dir analysis_results/qwen2_vl/vision_features
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

# Import dataset (optional - will use synthetic images if not available)
try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available. Will use synthetic images.")


IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> token ID in Qwen2-VL


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
        # Non-square grid - assume width > height or use aspect ratio
        grid_size = int(math.sqrt(num_patches))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def extract_vision_features_all_layers(model, processor, image, prompt, device):
    """
    Extract vision token features from ALL layers of Qwen2-VL.
    
    Returns:
        dict with:
        - 'features_by_layer': dict[layer_idx] -> tensor [num_vision, hidden_dim]
        - 'metadata': dict with shape info
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
    
    # Extract vision tokens from each layer
    features_by_layer = {}
    for layer_idx, hs in enumerate(hidden_states):
        # hs shape: [batch, seq_len, hidden_dim]
        vision_features = hs[:, vision_start:vision_end, :].squeeze(0)  # [num_vision, hidden_dim]
        # Normalize for cosine similarity (like in Molmo script)
        features_by_layer[layer_idx] = torch.nn.functional.normalize(vision_features, dim=-1).float()
    
    # Get image grid info for spatial layout
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
    
    return {
        'features_by_layer': features_by_layer,
        'metadata': metadata
    }


def preload_images(dataset, num_images, max_size=1024):
    """
    Preload and preprocess images.
    
    Returns list of dicts with 'image' and 'caption'.
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


def main():
    parser = argparse.ArgumentParser(description="Extract vision tokens across all LLM layers in Qwen2-VL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to use")
    parser.add_argument("--output-dir", type=str, default="analysis_results/qwen2_vl/vision_features",
                       help="Output directory")
    parser.add_argument("--save-features", action="store_true",
                       help="Save full feature tensors (large files)")
    parser.add_argument("--layers", type=str, default="all",
                       help="Layers to save: 'all' or comma-separated list like '0,8,16,24,28'")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    # Parse layers to save
    if args.layers == "all":
        layers_to_save = None  # Will save all
    else:
        layers_to_save = [int(l.strip()) for l in args.layers.split(",")]
    
    print("=" * 70)
    print("QWEN2-VL VISION TOKEN EXTRACTION (ALL LAYERS)")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Images: {args.num_images}")
    print(f"Split: {args.split}")
    print(f"Layers: {args.layers}")
    print(f"Output: {args.output_dir}")
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
    cached_images = preload_images(dataset, args.num_images)
    
    print(f"✓ Images preloaded in {time.time() - preload_start:.1f}s")
    print()
    
    # ===== PROCESS IMAGES =====
    print("=" * 70)
    print("EXTRACTING VISION FEATURES")
    print("=" * 70)
    process_start = time.time()
    
    prompt = "Describe this image in detail."
    all_results = []
    
    # Storage for features if saving full tensors
    if args.save_features:
        features_storage = {}  # layer_idx -> list of tensors
    
    for img_idx in tqdm(range(args.num_images), desc="Processing"):
        image_data = cached_images[img_idx]
        
        result = extract_vision_features_all_layers(
            model, processor, image_data['image'], prompt, device
        )
        
        # Build image result
        image_result = {
            'image_idx': img_idx,
            'caption': image_data['caption'],
            'num_vision_tokens': result['metadata']['num_vision_tokens'],
            'hidden_dim': result['metadata']['hidden_dim'],
            'num_layers': result['metadata']['num_layers'],
            'grid_info': result['metadata']['grid_info'],
            'layer_stats': {}
        }
        
        # Compute per-layer statistics
        for layer_idx, features in result['features_by_layer'].items():
            if layers_to_save is not None and layer_idx not in layers_to_save:
                continue
            
            norms = torch.norm(features, dim=-1)
            image_result['layer_stats'][int(layer_idx)] = {
                'mean_norm': float(norms.mean().item()),
                'std_norm': float(norms.std().item()),
                'shape': [int(x) for x in features.shape]
            }
            
            # Store features if saving full tensors
            if args.save_features:
                if layer_idx not in features_storage:
                    features_storage[layer_idx] = []
                features_storage[layer_idx].append(features.cpu())
        
        all_results.append(image_result)
        
        # Clean up
        del result
        gc.collect()
        torch.cuda.empty_cache()
    
    process_time = time.time() - process_start
    print(f"✓ Processed {args.num_images} images in {process_time:.1f}s ({args.num_images/process_time:.1f} img/s)")
    print()
    
    # ===== SAVE RESULTS =====
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    model_name_safe = args.model_name.replace("/", "_")
    output_dir = output_dir / model_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary_file = output_dir / f"extraction_summary_{args.split}_{args.num_images}imgs.json"
    summary_data = {
        'model_name': args.model_name,
        'split': args.split,
        'num_images': args.num_images,
        'layers_saved': args.layers,
        'processing_time_seconds': process_time,
        'results': all_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"  ✓ Summary: {summary_file}")
    
    # Save full features if requested
    if args.save_features:
        for layer_idx, feature_list in features_storage.items():
            features_file = output_dir / f"features_layer{layer_idx}_{args.split}_{args.num_images}imgs.pt"
            torch.save({
                'layer': layer_idx,
                'features': feature_list,  # List of tensors [num_vision, hidden_dim]
                'model_name': args.model_name,
                'num_images': args.num_images
            }, features_file)
            print(f"  ✓ Layer {layer_idx} features: {features_file}")
    
    print()
    print("=" * 70)
    print("✓ DONE!")
    print(f"  Processing time: {process_time:.1f}s ({process_time/60:.1f} min)")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

