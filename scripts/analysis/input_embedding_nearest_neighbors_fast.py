#!/usr/bin/env python3
"""
Fast single-GPU nearest neighbors to input embedding matrix.

WITH IMAGE PRELOADING - load all images directly to GPU once at start.

Algorithm:
    1. Load model (includes input embedding matrix)
    2. Preload all images directly to GPU
    3. For each image:
        Forward pass → get visual features for ALL requested layers at once
        For each layer: search against embedding matrix → store results
    4. Save results

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/input_embedding_nearest_neighbors_fast.py \
        --ckpt-path <path> --llm-layers 0,1,2,4,8,16,24,30,31 --num-images 300
"""

import argparse
import gc
import json
import time
import math
import os
import numpy as np
import torch
from pathlib import Path

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def preload_images(dataset, preprocessor, prompt, num_images, device):
    """Preload and preprocess all images directly to GPU."""
    cached_data = []
    
    for img_idx in range(num_images):
        if img_idx % 10 == 0:
            print(f"    {img_idx}/{num_images}...", flush=True)
        example_data = dataset.get(img_idx, np.random)
        
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            caption_text = example_data["message_list"][0].get("text", "")
        
        example = {"image": example_data["image"], "messages": [prompt]}
        batch = preprocessor(example, rng=np.random)
        
        cached_data.append({
            'images': torch.tensor(batch.get("images")).to(device),
            'image_masks': torch.tensor(batch.get("image_masks")).to(device) if batch.get("image_masks") is not None else None,
            'input_tokens': torch.tensor(batch["input_tokens"]).to(device),
            'image_input_idx': torch.tensor(batch.get("image_input_idx")).to(device) if batch.get("image_input_idx") is not None else None,
            'caption': caption_text
        })
    
    return cached_data


def extract_visual_features_all_layers(model, cached_batch, use_n_token_only, llm_layers, device):
    """
    Extract features from pre-cached batch for ALL requested layers in one forward pass.
    Returns: dict[layer] -> features [num_patches, hidden_dim], and metadata
    """
    features_by_layer = {}
    
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            images = cached_batch['images'].unsqueeze(0)
            image_masks = cached_batch['image_masks'].unsqueeze(0) if cached_batch['image_masks'] is not None else None
            
            need_layer_0 = 0 in llm_layers
            need_llm_layers = any(l > 0 for l in llm_layers)
            
            # Layer 0: vision backbone output (before MLP projection)
            if need_layer_0:
                feats_l0, _ = model.vision_backbone(images, image_masks, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    feats_l0 = feats_l0[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    feats_l0 = feats_l0[:, :, use_n_token_only, :]
                
                B, num_chunks, patches_per_chunk, hidden_dim = feats_l0.shape
                features_by_layer[0] = torch.nn.functional.normalize(feats_l0.view(-1, hidden_dim), dim=-1).float()
                del feats_l0
            
            # LLM layers: one forward pass extracts all hidden states
            if need_llm_layers:
                input_ids = cached_batch['input_tokens'].unsqueeze(0)
                image_input_idx = cached_batch['image_input_idx'].unsqueeze(0) if cached_batch['image_input_idx'] is not None else None
                
                output = model(
                    input_ids=input_ids,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                hidden_states = output.hidden_states
                
                B = image_input_idx.shape[0]
                num_chunks = image_input_idx.shape[1]
                patches_per_chunk = image_input_idx.shape[2]
                hidden_dim = hidden_states[0].shape[-1]
                
                flat_pos = image_input_idx.view(B, -1)
                valid_mask = flat_pos >= 0
                
                for layer in llm_layers:
                    if layer == 0:
                        continue
                    layer_idx = min(layer, len(hidden_states) - 1)
                    hs = hidden_states[layer_idx]
                    
                    feats = torch.zeros((B, num_chunks, patches_per_chunk, hidden_dim), device=hs.device, dtype=hs.dtype)
                    for b in range(B):
                        valid = flat_pos[b][valid_mask[b]]
                        if valid.numel() > 0:
                            feats.view(B, -1, hidden_dim)[b, valid_mask[b], :] = hs[b, valid.long(), :]
                    
                    features_by_layer[layer] = torch.nn.functional.normalize(feats.view(-1, hidden_dim), dim=-1).float()
                
                del hidden_states, output
            
            del images, image_masks
            torch.cuda.empty_cache()
    
    first_layer = llm_layers[0]
    num_patches = features_by_layer[first_layer].shape[0]
    metadata = {
        'num_chunks': num_chunks,
        'patches_per_chunk': patches_per_chunk,
        'hidden_dim': hidden_dim,
        'num_patches': num_patches
    }
    
    return features_by_layer, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--llm-layers", type=str, default="0,1,2,4,8,16,24,30,31",
                        help="Comma-separated list of LLM layers to analyze")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="analysis_results/nearest_neighbors")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    llm_layers = [int(l.strip()) for l in args.llm_layers.split(",")]
    
    print("=" * 70)
    print("INPUT EMBEDDING NEAREST NEIGHBORS (FAST)")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"LLM layers: {llm_layers}")
    print(f"Images: {args.num_images}")
    print(f"Top-k: {args.top_k}")
    print()
    
    # ===== LOAD MODEL =====
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    load_start = time.time()
    
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)
    
    ckpt_file = f"{args.ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024**3)
    
    if ckpt_size_gb < 1.0:
        print(f"  Stripped checkpoint ({ckpt_size_gb:.2f} GB) - loading pretrained...")
        model.reset_with_pretrained_weights()
    
    print(f"  Loading weights...")
    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()
    
    print(f"  Moving to GPU (fp16)...")
    model = model.half().cuda().eval()
    torch.cuda.empty_cache()
    
    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")
    print()
    
    # ===== GET INPUT EMBEDDING MATRIX =====
    print("=" * 70)
    print("PREPARING EMBEDDING MATRIX")
    print("=" * 70)
    
    # Get tokenizer for decoding
    from transformers import AutoTokenizer
    model_config = ModelConfig.load(resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False)
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer.identifier)
    
    # Load cached embeddings (same as original script) - this is the most reliable approach
    model_identifier = model_config.tokenizer.identifier
    if "qwen" in model_identifier.lower():
        cached_embeddings_path = "analysis_results/cached_text_embeddings/Qwen_Qwen2-7B/layer_0_static_vocab.npy"
    elif "dolma" in model_identifier.lower() or "olmo" in model_identifier.lower():
        cached_embeddings_path = "analysis_results/cached_text_embeddings/allenai_OLMo-7B-1024-preview/layer_0_static_vocab.npy"
    elif "llama" in model_identifier.lower():
        cached_embeddings_path = "analysis_results/cached_text_embeddings/meta-llama_Meta-Llama-3-8B/layer_0_static_vocab.npy"
    else:
        cached_embeddings_path = None
    
    if cached_embeddings_path and os.path.exists(cached_embeddings_path):
        print(f"  Loading cached embeddings: {cached_embeddings_path}")
        embedding_matrix = torch.from_numpy(np.load(cached_embeddings_path)).to(device).float()
    else:
        print(f"  Cached embeddings not found, loading from model...")
        # Molmo's Embedding class uses .embedding + .new_embedding, not .weight
        wte = model.transformer.wte
        embedding_matrix = torch.cat([wte.embedding, wte.new_embedding], dim=0).float()
    
    embedding_matrix_norm = torch.nn.functional.normalize(embedding_matrix, dim=-1)
    vocab_size = embedding_matrix.shape[0]
    
    print(f"  Embedding matrix shape: {embedding_matrix.shape}")
    print(f"  Vocab size: {vocab_size}")
    print()
    
    # Create preprocessor and dataset
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(model_config, for_inference=True, shuffle_messages=False, is_training=False, require_image_features=True)
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    
    dataset = PixMoCap(split=args.split, mode="captions")
    prompt = "Describe this image in detail."
    
    # ===== PRELOAD IMAGES =====
    print("=" * 70)
    print("PRELOADING IMAGES TO GPU")
    print("=" * 70)
    preload_start = time.time()
    
    print(f"  Loading {args.num_images} images directly to GPU...", flush=True)
    cached_images = preload_images(dataset, preprocessor, prompt, args.num_images, device)
    
    preload_time = time.time() - preload_start
    print(f"✓ Images preloaded in {preload_time:.1f}s ({args.num_images/preload_time:.1f} img/s)")
    print()
    
    # Output setup
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== MAIN PROCESSING =====
    print("=" * 70)
    print("PROCESSING IMAGES")
    print("=" * 70)
    
    process_start = time.time()
    all_results = {layer: [] for layer in llm_layers}
    shape_info = None
    
    for img_idx in range(args.num_images):
        if img_idx % 10 == 0 or img_idx == args.num_images - 1:
            elapsed = time.time() - process_start
            rate = (img_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Image {img_idx + 1}/{args.num_images} ({rate:.1f} img/s)", flush=True)
        
        # Extract features for ALL layers in one forward pass
        features_by_layer, meta = extract_visual_features_all_layers(
            model, cached_images[img_idx], use_n_token_only, llm_layers, device
        )
        
        if shape_info is None:
            shape_info = meta
        
        num_chunks = meta['num_chunks']
        patches_per_chunk = meta['patches_per_chunk']
        hidden_dim = meta['hidden_dim']
        
        # For each layer, compute nearest neighbors to embedding matrix
        for layer in llm_layers:
            feats = features_by_layer[layer]  # [num_patches, hidden_dim]
            
            # Compute similarity to all embeddings
            similarity = torch.matmul(feats, embedding_matrix_norm.T)  # [num_patches, vocab_size]
            top_vals, top_idxs = torch.topk(similarity, k=args.top_k, dim=-1)
            
            # Build results for this image/layer
            chunks_results = []
            for chunk_idx in range(num_chunks):
                chunk_results = {
                    "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                    "patches": []
                }
                
                for local_patch_idx in range(patches_per_chunk):
                    global_patch_idx = chunk_idx * patches_per_chunk + local_patch_idx
                    
                    nearest = []
                    for k_idx in range(args.top_k):
                        token_id = top_idxs[global_patch_idx, k_idx].item()
                        sim = top_vals[global_patch_idx, k_idx].item()
                        token_str = tokenizer.decode([token_id])
                        
                        # Use "token" not "token_str" to match original format expected by LLM judge
                        nearest.append({
                            'token': token_str,
                            'similarity': sim
                        })
                    
                    row, col = patch_idx_to_row_col(local_patch_idx, patches_per_chunk)
                    chunk_results["patches"].append({
                        "patch_idx": local_patch_idx,
                        "patch_row": row,
                        "patch_col": col,
                        "nearest_neighbors": nearest
                    })
                
                chunks_results.append(chunk_results)
            
            all_results[layer].append({
                "image_idx": img_idx,
                "ground_truth_caption": cached_images[img_idx]['caption'],
                "chunks": chunks_results
            })
            
            del similarity, top_vals, top_idxs
        
        del features_by_layer
    
    process_time = time.time() - process_start
    print(f"✓ Processing done in {process_time:.1f}s ({args.num_images/process_time:.1f} img/s)")
    print()
    
    # ===== SAVE RESULTS =====
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    total_time = preload_time + process_time
    
    for layer in llm_layers:
        # Use same filename pattern as original: nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{N}.json
        output_file = output_dir / f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{layer}.json"
        
        # Match original format structure with splits
        output_data = {
            'checkpoint': args.ckpt_path,
            'prompt': "Describe this image in detail.",
            'dataset': 'PixMoCap',
            'num_processes': 1,
            'preprocessing_mode': None,
            'llm_layer': layer,
            'splits': {
                'train': {
                    'num_images': 0,
                    'images': [],
                    'statistics': {}
                },
                args.split: {
                    'num_images': args.num_images,
                    'images': all_results[layer],
                    'statistics': {}
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ {output_file.name}")
    
    print()
    print("=" * 70)
    print("✓ DONE!")
    print(f"  Preload: {preload_time:.1f}s | Process: {process_time:.1f}s")
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

