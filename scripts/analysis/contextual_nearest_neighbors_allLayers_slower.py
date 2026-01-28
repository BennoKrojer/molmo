#!/usr/bin/env python3
"""
Slower but CORRECT version of allLayers contextual nearest neighbors.

Instead of distributed sharded search, this script:
1. Loads contextual embeddings ONE LAYER AT A TIME
2. Computes top-k for all patches against that layer
3. Aggregates best matches across all layers
4. Frees memory after each layer

This is slower than distributed search but guaranteed correct.

Usage:
    torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors_allLayers_slower.py \
        --ckpt-path <path> --contextual-dir <dir> --visual-layer 24 --num-images 300
"""

import argparse
import gc
import json
import time
import math
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap
from olmo.torch_util import get_local_rank, get_world_size


def load_contextual_embeddings(contextual_dir, layer_idx):
    """Load contextual embeddings from cache for a single layer."""
    cache_file = Path(contextual_dir) / f"layer_{layer_idx}" / "embeddings_cache.pt"
    cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
    return cache_data['embeddings'], cache_data['metadata'], cache_data.get('token_to_indices', {})


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


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def process_single_image(model, preprocessor, dataset, image_idx, prompt, use_n_token_only, 
                         visual_layer, contextual_dir, available_layers, top_k, device):
    """
    Process a single image and find nearest neighbors across ALL contextual layers.
    
    Iterates over each layer cache sequentially to avoid memory issues.
    """
    example_data = dataset.get(image_idx, np.random)
    
    # Extract caption
    caption_text = ""
    if "message_list" in example_data and len(example_data["message_list"]) > 0:
        caption_text = example_data["message_list"][0].get("text", "")
    
    # Preprocess
    example = {"image": example_data["image"], "messages": [prompt]}
    batch = preprocessor(example, rng=np.random)
    
    # Extract visual tokens
    model_start = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
            image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
            
            if visual_layer == 0:
                image_features, _ = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    image_features = image_features[:, :, :use_n_token_only, :]
            else:
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                
                output = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=image_input_idx_tensor,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                hidden_states = output.hidden_states
                
                def gather_visual_features(hs_tensor):
                    B, num_chunks, patches_per_chunk, d_model = image_input_idx_tensor.shape[0], image_input_idx_tensor.shape[1], image_input_idx_tensor.shape[2], hs_tensor.shape[-1]
                    feats = torch.zeros((B, num_chunks, patches_per_chunk, d_model), device=hs_tensor.device, dtype=hs_tensor.dtype)
                    flat_positions = image_input_idx_tensor.view(B, -1)
                    valid_mask = flat_positions >= 0
                    for b in range(B):
                        valid_pos = flat_positions[b][valid_mask[b]]
                        if valid_pos.numel() == 0:
                            continue
                        gathered = hs_tensor[b, valid_pos.long(), :]
                        feats.view(B, -1, d_model)[b, valid_mask[b], :] = gathered
                    return feats
                
                layer_index = min(visual_layer, len(hidden_states) - 1)
                image_features = gather_visual_features(hidden_states[layer_index])
    
    batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
    total_patches = num_chunks * patches_per_chunk
    
    # Normalize visual features and convert to float32 (FSDP may use float16)
    all_patches_norm = torch.nn.functional.normalize(
        image_features.view(-1, hidden_dim), dim=-1
    ).cpu().float()  # Ensure float32 for compatibility with contextual embeddings
    
    model_time = time.time() - model_start
    
    # Free GPU memory
    del images_tensor, image_masks_tensor, image_features
    torch.cuda.empty_cache()
    
    if device.index == 0:
        print(f"  [img {image_idx}] Model forward: {model_time:.1f}s, starting layer loop...", flush=True)
    
    layer_start = time.time()
    
    # For each patch, collect candidates from all layers
    # patch_candidates[patch_idx] = list of (sim, layer, local_idx, metadata)
    patch_candidates = [[] for _ in range(total_patches)]
    
    # Iterate over each layer
    for li, layer_idx in enumerate(available_layers):
        if device.index == 0:  # Only rank 0 logs
            print(f"    [img {image_idx}] Loading layer {layer_idx} ({li+1}/{len(available_layers)})...", flush=True)
        embeddings, metadata, token_to_indices = load_contextual_embeddings(contextual_dir, layer_idx)
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)
        
        # Compute similarity for ALL patches at once: [num_patches, num_embeddings]
        similarity = torch.matmul(all_patches_norm, embeddings_norm.T)
        
        # Get top-k for each patch from THIS layer
        top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)
        
        # Store candidates
        for patch_idx in range(total_patches):
            for k in range(top_k):
                sim = top_values[patch_idx, k].item()
                idx = top_indices[patch_idx, k].item()
                meta = metadata[idx]
                
                patch_candidates[patch_idx].append({
                    'similarity': sim,
                    'contextual_layer': layer_idx,
                    'local_idx': idx,
                    'token_str': meta['token_str'],
                    'token_id': meta['token_id'],
                    'caption': meta['caption'],
                    'position': meta['position'],
                    'token_to_indices': token_to_indices  # Keep for lowest-sim lookup
                })
        
        del embeddings, embeddings_norm, similarity, metadata
        gc.collect()
    
    layer_time = time.time() - layer_start
    if device.index == 0:
        print(f"  [img {image_idx}] Layer loop done: {layer_time:.1f}s total", flush=True)
    
    # Build final results - aggregate top-k across all layers
    chunks_results = []
    for chunk_idx in range(num_chunks):
        chunk_results = {
            "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
            "patches": []
        }
        
        for local_patch_idx in range(patches_per_chunk):
            global_patch_idx = chunk_idx * patches_per_chunk + local_patch_idx
            
            # Sort candidates by similarity and take top-k
            candidates = patch_candidates[global_patch_idx]
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            top_candidates = candidates[:top_k]
            
            # Build nearest neighbors list
            nearest_contextual = []
            for cand in top_candidates:
                nearest_contextual.append({
                    'token_str': cand['token_str'],
                    'token_id': cand['token_id'],
                    'caption': cand['caption'],
                    'position': cand['position'],
                    'similarity': cand['similarity'],
                    'contextual_layer': cand['contextual_layer']
                })
            
            row, col = patch_idx_to_row_col(local_patch_idx, patches_per_chunk)
            
            patch_results = {
                "patch_idx": local_patch_idx,
                "patch_row": row,
                "patch_col": col,
                "nearest_contextual_neighbors": nearest_contextual
            }
            chunk_results["patches"].append(patch_results)
        
        chunks_results.append(chunk_results)
    
    return {
        "image_idx": image_idx,
        "ground_truth_caption": caption_text,
        "feature_shape": [batch_size, num_chunks, patches_per_chunk, hidden_dim],
        "llm_layer_used": visual_layer,
        "chunks": chunks_results
    }


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    world_size = get_world_size()
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--contextual-dir", type=str, required=True)
    parser.add_argument("--visual-layer", type=str, default="24")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="analysis_results/contextual_nearest_neighbors")
    args = parser.parse_args()
    
    visual_layers = [int(layer.strip()) for layer in args.visual_layer.split(",")]
    available_layers = find_available_layers(args.contextual_dir)
    
    if local_rank == 0:
        print("="*70)
        print("allLayers Contextual NN (SLOWER but CORRECT version)")
        print("="*70)
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Contextual dir: {args.contextual_dir}")
        print(f"Visual layers: {visual_layers}")
        print(f"Contextual layers: {available_layers}")
        print(f"Images: {args.num_images}")
        print(f"Processes: {world_size}")
        print()
        print("This version iterates over layer caches sequentially (no sharding)")
        print()
    
    # Load model with FSDP
    if local_rank == 0:
        print("Loading model...")
    
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = None
    model = Molmo(cfg.model)
    
    import os
    checkpoint_file = f"{args.ckpt_path}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    
    if checkpoint_size_gb < 1.0:
        if local_rank == 0:
            print(f"  Stripped checkpoint ({checkpoint_size_gb:.2f} GB) - loading pretrained LLM weights...")
            print("  (This can take 10-15 min on first run or if not cached)")
        model.reset_with_pretrained_weights()
        if local_rank == 0:
            print("  ✓ Pretrained weights loaded")
    else:
        if local_rank == 0:
            print(f"  Full checkpoint ({checkpoint_size_gb:.2f} GB) - skipping pretrained weights")
    
    if local_rank == 0:
        print("  Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    del checkpoint_weights
    gc.collect()
    
    model.eval()
    
    # FSDP wrap
    wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    
    if local_rank == 0:
        print(f"Model loaded with FSDP on {world_size} GPUs\n")
    
    # Create preprocessor
    model_config = ModelConfig.load(resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=True, shuffle_messages=False,
        is_training=False, require_image_features=True
    )
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    
    # Load dataset
    dataset = PixMoCap(split=args.split, mode="captions")
    prompt = "Describe this image in detail."
    
    # Process each visual layer
    for visual_layer in visual_layers:
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"Processing visual layer {visual_layer}")
            print(f"{'='*60}")
        
        # Distribute images across ranks
        images_per_rank = args.num_images // world_size
        start_idx = local_rank * images_per_rank
        end_idx = start_idx + images_per_rank
        if local_rank == world_size - 1:
            end_idx = args.num_images
        
        my_images = list(range(start_idx, end_idx))
        
        if local_rank == 0:
            print(f"Each rank processes ~{images_per_rank} images")
        
        # Process images
        results = []
        for img_idx in tqdm(my_images, desc=f"Rank {local_rank}", disable=(local_rank != 0)):
            result = process_single_image(
                model, preprocessor, dataset, img_idx, prompt, use_n_token_only,
                visual_layer, args.contextual_dir, available_layers, args.top_k, device
            )
            results.append(result)
            gc.collect()
        
        # Gather results
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        
        if local_rank == 0:
            combined = []
            for rank_results in all_results:
                combined.extend(rank_results)
            
            # Sort by image_idx
            combined.sort(key=lambda x: x['image_idx'])
            
            # Save
            ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
            output_dir = Path(args.output_dir) / ckpt_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"contextual_neighbors_visual{visual_layer}_allLayers_slower.json"
            
            output_data = {
                'checkpoint': args.ckpt_path,
                'contextual_dir': args.contextual_dir,
                'visual_layer': visual_layer,
                'contextual_layers_used': available_layers,
                'split': args.split,
                'num_images': args.num_images,
                'num_processes': world_size,
                'top_k': args.top_k,
                'results': combined
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n✓ Saved to {output_file}")
        
        dist.barrier()
    
    if local_rank == 0:
        print("\n" + "="*70)
        print("✓ DONE!")
        print("="*70)


if __name__ == "__main__":
    main()

