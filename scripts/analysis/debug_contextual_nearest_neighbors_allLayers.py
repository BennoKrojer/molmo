#!/usr/bin/env python3
"""
DEBUG SCRIPT: Simple sequential comparison of visual tokens to ALL layer contextual embeddings.

Hardcoded for the specific case we're debugging.

Run: python scripts/analysis/debug_contextual_nearest_neighbors_allLayers.py
"""

import gc
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap

# ============ HARDCODED CONFIG ============
CKPT_PATH = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded"
CONTEXTUAL_DIR = "molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"
VISUAL_LAYER = 24
IMAGE_IDX = 0
TOP_K = 5
SPLIT = "validation"
OUTPUT_FILE = "analysis_results/debug_allLayers_sequential_img0.json"
# ==========================================


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


def main():
    device = torch.device("cuda:0")
    
    print("="*70)
    print("DEBUG: Sequential allLayers Nearest Neighbors (ALL PATCHES)")
    print("="*70)
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Contextual dir: {CONTEXTUAL_DIR}")
    print(f"Visual layer: {VISUAL_LAYER}")
    print(f"Image index: {IMAGE_IDX}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    available_layers = find_available_layers(CONTEXTUAL_DIR)
    print(f"Available contextual layers: {available_layers}")
    print()
    
    # ========== STEP 1: Load model and extract visual tokens ==========
    print("="*70)
    print("STEP 1: Loading model and extracting visual tokens")
    print("="*70)
    
    cfg = TrainConfig.load(f"{CKPT_PATH}/config.yaml")
    cfg.model.init_device = None
    model = Molmo(cfg.model)
    
    import os
    checkpoint_file = f"{CKPT_PATH}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    is_full_checkpoint = checkpoint_size_gb > 1.0
    
    if not is_full_checkpoint:
        print(f"Stripped checkpoint ({checkpoint_size_gb:.2f} GB) - loading pretrained weights...")
        model.reset_with_pretrained_weights()
    
    print("Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    del checkpoint_weights
    gc.collect()
    
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    model_config = ModelConfig.load(resource_path(CKPT_PATH, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=True, shuffle_messages=False,
        is_training=False, require_image_features=True
    )
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    
    print(f"\nLoading image {IMAGE_IDX}...")
    dataset = PixMoCap(split=SPLIT, mode="captions")
    example_data = dataset.get(IMAGE_IDX, np.random)
    
    caption_text = ""
    if "message_list" in example_data and len(example_data["message_list"]) > 0:
        caption_text = example_data["message_list"][0].get("text", "")
    print(f"Caption: {caption_text[:100]}...")
    
    prompt = "Describe this image in detail."
    example = {"image": example_data["image"], "messages": [prompt]}
    batch = preprocessor(example, rng=np.random)
    
    print(f"\nExtracting visual tokens at layer {VISUAL_LAYER}...")
    
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
            image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
            
            if VISUAL_LAYER == 0:
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
                
                layer_index = min(VISUAL_LAYER, len(hidden_states) - 1)
                image_features = gather_visual_features(hidden_states[layer_index])
    
    batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
    total_patches = num_chunks * patches_per_chunk
    print(f"Visual features shape: {image_features.shape} → {total_patches} total patches")
    
    # Flatten to [total_patches, hidden_dim] and normalize
    all_patches_norm = torch.nn.functional.normalize(
        image_features.view(-1, hidden_dim), dim=-1
    ).cpu()
    
    del images_tensor, image_masks_tensor, image_features, model
    torch.cuda.empty_cache()
    gc.collect()
    print("Model unloaded, visual tokens on CPU")
    
    # ========== STEP 2: Process each layer ONCE, get top-k for ALL patches ==========
    print("\n" + "="*70)
    print("STEP 2: Finding top-5 per layer (processing all patches at once)")
    print("="*70)
    
    # For each patch, store list of (similarity, layer, local_idx, metadata)
    # We'll aggregate at the end
    # Shape: [num_patches, num_layers * TOP_K, 4] - but use lists for simplicity
    
    # Better approach: for each patch, keep track of current top-k across all layers
    # Initialize with -inf similarities
    # patch_top_k[patch_idx] = list of (sim, layer, token_str, caption, ...)
    
    patch_top_k = [[] for _ in range(total_patches)]
    
    for layer_idx in tqdm(available_layers, desc="Layers"):
        print(f"\n  Loading layer {layer_idx}...")
        embeddings, metadata, _ = load_contextual_embeddings(CONTEXTUAL_DIR, layer_idx)
        print(f"    {embeddings.shape[0]} embeddings")
        
        # Normalize
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)
        
        # Compute similarity for ALL patches at once: [num_patches, num_embeddings]
        print(f"    Computing similarities...")
        similarity = torch.matmul(all_patches_norm, embeddings_norm.T)
        
        # Get top-k for each patch from THIS layer
        print(f"    Finding top-{TOP_K}...")
        top_values, top_indices = torch.topk(similarity, k=TOP_K, dim=-1)  # [num_patches, TOP_K]
        
        # Add to each patch's candidates
        for patch_idx in range(total_patches):
            for k in range(TOP_K):
                sim = top_values[patch_idx, k].item()
                idx = top_indices[patch_idx, k].item()
                meta = metadata[idx]
                
                patch_top_k[patch_idx].append({
                    'similarity': sim,
                    'contextual_layer': layer_idx,
                    'token_str': meta['token_str'],
                    'token_id': meta['token_id'],
                    'caption': meta['caption'],
                    'position': meta['position']
                })
        
        # Free memory
        del embeddings, embeddings_norm, similarity, top_values, top_indices, metadata
        gc.collect()
    
    # ========== STEP 3: For each patch, sort and keep global top-k ==========
    print("\n" + "="*70)
    print("STEP 3: Aggregating global top-5 for each patch")
    print("="*70)
    
    patch_results = []
    for patch_idx in range(total_patches):
        candidates = patch_top_k[patch_idx]
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        global_top = candidates[:TOP_K]
        
        patch_results.append({
            'patch_idx': patch_idx,
            'nearest_neighbors': global_top
        })
    
    # ========== STEP 4: Save results ==========
    print("\n" + "="*70)
    print("STEP 4: Saving results")
    print("="*70)
    
    output_data = {
        'checkpoint': CKPT_PATH,
        'contextual_dir': CONTEXTUAL_DIR,
        'visual_layer': VISUAL_LAYER,
        'contextual_layers_used': available_layers,
        'image_idx': IMAGE_IDX,
        'ground_truth_caption': caption_text,
        'num_patches': total_patches,
        'top_k': TOP_K,
        'patches': patch_results
    }
    
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved to {OUTPUT_FILE}")
    
    # Print sample results
    print("\n" + "="*70)
    print("Sample results:")
    print("="*70)
    
    for patch_idx in [0, 100, 224, 300, 500]:
        if patch_idx < len(patch_results):
            patch = patch_results[patch_idx]
            print(f"\nPatch {patch_idx}:")
            for i, nn in enumerate(patch['nearest_neighbors'][:3]):
                print(f"  {i+1}. sim={nn['similarity']:.4f}, layer={nn['contextual_layer']}, token='{nn['token_str']}'")
    
    print("\n✓ DONE!")


if __name__ == "__main__":
    main()
