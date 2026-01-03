#!/usr/bin/env python3
"""
Static Nearest Neighbor Analysis for Qwen2-VL Vision Tokens

Finds the nearest vocabulary embeddings for vision tokens at each layer.
This is the "input embedding matrix NN" baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/nearest_neighbors.py \
        --num-images 100 --layers "0,8,16,24,27" --output-dir analysis_results/nearest_neighbors/qwen2_vl
"""

import argparse
import gc
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available.")

IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> token ID in Qwen2-VL


def find_image_token_positions(input_ids):
    """Find positions of vision tokens in the input sequence."""
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
        grid_size = int(math.ceil(math.sqrt(num_patches)))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def decode_token(tokenizer, idx):
    """Decode a token ID to string."""
    token = tokenizer.decode([int(idx)])
    return token.encode('utf-8').decode('utf-8')


def find_nearest_neighbors(vision_features, vocab_embeddings, top_k=5):
    """
    Find top-k nearest vocabulary embeddings for each vision token.
    
    Args:
        vision_features: [num_vision, hidden_dim] normalized features
        vocab_embeddings: [vocab_size, hidden_dim] normalized embeddings
        top_k: number of nearest neighbors
    
    Returns:
        topk_indices: [num_vision, top_k]
        topk_similarities: [num_vision, top_k]
    """
    # Cosine similarity (features are already normalized)
    similarities = torch.mm(vision_features, vocab_embeddings.T)  # [num_vision, vocab_size]
    
    topk_similarities, topk_indices = torch.topk(similarities, k=top_k, dim=-1)
    
    return topk_indices.cpu().numpy(), topk_similarities.cpu().numpy()


def extract_nn(model, processor, image, prompt, device, vocab_embeddings, layers_to_analyze, top_k=5):
    """
    Find nearest vocabulary neighbors for vision tokens at specified layers.
    
    Returns dict with per-layer top-k nearest neighbors for each vision token.
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
        return None
    
    # Get image grid info for spatial layout
    grid_info = None
    if 'image_grid_thw' in model_inputs:
        grid = model_inputs['image_grid_thw']
        if isinstance(grid, torch.Tensor):
            t, h, w = grid[0].tolist()
            grid_info = {'temporal': int(t), 'height': int(h), 'width': int(w)}
    
    # Run forward pass
    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            outputs = model(**model_inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    results_by_layer = {}
    
    for layer_idx in layers_to_analyze:
        if layer_idx >= len(hidden_states):
            continue
        
        # Get hidden state at this layer
        hs = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
        
        # Extract vision tokens
        vision_hs = hs[:, vision_start:vision_end, :].squeeze(0)  # [num_vision, hidden_dim]
        
        # Normalize for cosine similarity
        vision_hs_normed = F.normalize(vision_hs.float(), dim=-1)
        
        # Find nearest neighbors
        topk_indices, topk_similarities = find_nearest_neighbors(
            vision_hs_normed, vocab_embeddings, top_k
        )
        
        # Build results for this layer
        layer_results = []
        for patch_idx in range(num_vision_tokens):
            row, col = patch_idx_to_row_col(patch_idx, num_vision_tokens)
            
            neighbors = []
            for idx, sim in zip(topk_indices[patch_idx], topk_similarities[patch_idx]):
                token_str = decode_token(processor.tokenizer, idx)
                neighbors.append({
                    "token": token_str,
                    "token_id": int(idx),
                    "similarity": float(sim)
                })
            
            layer_results.append({
                "patch_idx": patch_idx,
                "patch_row": row,
                "patch_col": col,
                "nearest_neighbors": neighbors  # Standardized key name
            })
        
        results_by_layer[layer_idx] = layer_results
    
    # Clean up
    del hidden_states, outputs
    torch.cuda.empty_cache()
    
    return {
        'layers': results_by_layer,
        'num_vision_tokens': num_vision_tokens,
        'grid_info': grid_info
    }


def main():
    parser = argparse.ArgumentParser(description="Static NN for Qwen2-VL vision tokens")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output-dir", type=str, default="analysis_results/nearest_neighbors/qwen2_vl")
    parser.add_argument("--layers", type=str, default="0,1,2,4,8,16,24,26,27",
                       help="Comma-separated layer indices")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fixed-resolution", type=int, default=448,
                       help="Fixed image resolution (default: 448 for ~256 tokens like Molmo). Set to 0 for dynamic resolution.")
    parser.add_argument("--force-square", action="store_true", default=True,
                       help="Center-crop images to square before processing (ensures consistent 16x16 grid). Default: True")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    # Parse layers
    layers_to_analyze = [int(l.strip()) for l in args.layers.split(",")]
    
    print("=" * 70)
    print("QWEN2-VL STATIC NEAREST NEIGHBOR ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Images: {args.num_images}")
    print(f"Layers: {layers_to_analyze}")
    print(f"Top-K: {args.top_k}")
    print()
    
    # Load model
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set fixed resolution if specified (for consistent token counts across images)
    # This MUST match the preprocessing used in contextual_nearest_neighbors script!
    if args.fixed_resolution > 0:
        fixed_pixels = args.fixed_resolution * args.fixed_resolution
        processor.image_processor.min_pixels = fixed_pixels
        processor.image_processor.max_pixels = fixed_pixels
        # Qwen2-VL: 14x14 patches with 2x2 spatial merger = 28 pixels per token
        expected_tokens = (args.fixed_resolution // 28) ** 2
        print(f"✓ Fixed resolution: {args.fixed_resolution}x{args.fixed_resolution} (~{expected_tokens} vision tokens, {int(math.sqrt(expected_tokens))}x{int(math.sqrt(expected_tokens))} grid)")
    else:
        print(f"✓ Using dynamic resolution (variable token counts)")
    
    if args.force_square:
        print(f"✓ Force-square: ON (images center-cropped to square for consistent 16x16 grid)")
    else:
        print(f"⚠ Force-square: OFF (variable grids possible for non-square images)")
    
    print("✓ Model loaded")
    
    # Get vocabulary embeddings (input embeddings)
    print("Extracting vocabulary embeddings...")
    embed_tokens = model.model.embed_tokens.weight.data  # [vocab_size, hidden_dim]
    vocab_embeddings = F.normalize(embed_tokens.float(), dim=-1).to(device)
    print(f"✓ Vocabulary embeddings: {vocab_embeddings.shape}")
    
    # Load dataset
    if HAVE_PIXMOCAP:
        dataset = PixMoCap(split=args.split, mode="captions")
        print(f"✓ Using PixMoCap {args.split} split")
    else:
        print("ERROR: PixMoCap dataset required")
        return
    
    # Process images
    print(f"\nProcessing {args.num_images} images...")
    prompt = "Describe this image in detail."
    all_results = []
    
    for img_idx in tqdm(range(args.num_images)):
        example = dataset.get(img_idx, np.random)
        image = Image.open(example["image"]).convert('RGB')
        
        # Center-crop to square if requested (CRITICAL for consistent grid dimensions!)
        # Without this, non-square images produce variable grids (e.g., 13x18 instead of 16x16)
        if args.force_square and image.size[0] != image.size[1]:
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
        
        # Get caption
        caption = ""
        if "message_list" in example and len(example["message_list"]) > 0:
            caption = example["message_list"][0].get("text", "")
        
        result = extract_nn(model, processor, image, prompt, device, 
                           vocab_embeddings, layers_to_analyze, args.top_k)
        
        if result is not None:
            all_results.append({
                "image_idx": img_idx,
                "ground_truth_caption": caption,
                "num_vision_tokens": result['num_vision_tokens'],
                "grid_info": result['grid_info'],
                "layers": {str(k): v for k, v in result['layers'].items()}
            })
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results
    output_dir = Path(args.output_dir)
    model_name_safe = args.model_name.replace("/", "_")
    output_dir = output_dir / model_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-layer files (matching Molmo format)
    for layer_idx in layers_to_analyze:
        layer_results = {
            "model_name": args.model_name,
            "layer": layer_idx,
            "num_images": len(all_results),
            "top_k": args.top_k,
            "results": []
        }
        
        for img_result in all_results:
            layer_key = str(layer_idx)
            if layer_key in img_result["layers"]:
                layer_results["results"].append({
                    "image_idx": img_result["image_idx"],
                    "ground_truth_caption": img_result["ground_truth_caption"],
                    "num_vision_tokens": img_result["num_vision_tokens"],
                    "grid_info": img_result["grid_info"],
                    "patches": img_result["layers"][layer_key]
                })
        
        output_file = output_dir / f"nearest_neighbors_layer{layer_idx}_topk{args.top_k}.json"
        with open(output_file, 'w') as f:
            json.dump(layer_results, f, indent=2)
        print(f"  ✓ Saved layer {layer_idx}: {output_file}")
    
    print()
    print("=" * 70)
    print(f"✓ DONE! Results saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


