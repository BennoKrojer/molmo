#!/usr/bin/env python3
"""
Compare embeddings loaded by same-layer vs allLayers approach.
Verify that layer 24 embeddings are identical in both cases.
"""

import torch
from pathlib import Path

def load_single_layer_cache(contextual_dir, layer_idx):
    """Load cache for a single layer (same as same-layer script)."""
    cache_file = Path(contextual_dir) / f"layer_{layer_idx}" / "embeddings_cache.pt"
    cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
    return cache_data['embeddings'], cache_data['metadata'], cache_data['token_to_indices']

def main():
    contextual_dir = "molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"
    
    # Method 1: Load ONLY layer 24 (like same-layer script)
    print("="*60)
    print("Method 1: Load layer 24 ONLY (like same-layer script)")
    print("="*60)
    single_emb, single_meta, single_tok2idx = load_single_layer_cache(contextual_dir, 24)
    print(f"Shape: {single_emb.shape}")
    print(f"Metadata count: {len(single_meta)}")
    print(f"Sample embedding norm: {torch.norm(single_emb[0]).item():.4f}")
    
    # Method 2: Load ALL layers and extract layer 24 portion (like allLayers script)
    print("\n" + "="*60)
    print("Method 2: Load ALL layers, extract layer 24 portion")
    print("="*60)
    
    available_layers = [1, 2, 4, 8, 16, 24, 30, 31]
    combined_embeddings_list = []
    layer_boundaries = []
    
    current_idx = 0
    for layer_idx in available_layers:
        emb, meta, _ = load_single_layer_cache(contextual_dir, layer_idx)
        start_idx = current_idx
        end_idx = current_idx + len(meta)
        layer_boundaries.append((layer_idx, start_idx, end_idx))
        combined_embeddings_list.append(emb)
        current_idx = end_idx
    
    combined_embeddings = torch.cat(combined_embeddings_list, dim=0)
    print(f"Combined shape: {combined_embeddings.shape}")
    
    # Find layer 24 boundaries
    layer_24_bounds = [b for b in layer_boundaries if b[0] == 24][0]
    layer_24_start = layer_24_bounds[1]
    layer_24_end = layer_24_bounds[2]
    print(f"Layer 24 in combined: [{layer_24_start}, {layer_24_end})")
    
    # Extract layer 24 portion from combined
    combined_layer24 = combined_embeddings[layer_24_start:layer_24_end]
    print(f"Extracted layer 24 shape: {combined_layer24.shape}")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # Shape match?
    print(f"Shapes match: {single_emb.shape == combined_layer24.shape}")
    
    # Values match?
    if single_emb.shape == combined_layer24.shape:
        diff = torch.abs(single_emb - combined_layer24)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        if max_diff < 1e-6:
            print("\n✓ EMBEDDINGS ARE IDENTICAL")
            print("  Bug is NOT in loading/combining, must be in SEARCH or INDEX MAPPING")
        else:
            print("\n✗ EMBEDDINGS DIFFER")
            print("  Bug is in loading/combining!")
            # Find where they differ
            diff_mask = diff.sum(dim=-1) > 1e-6
            diff_indices = torch.where(diff_mask)[0][:10]
            print(f"  First differing indices: {diff_indices.tolist()}")
    
    # Also verify that the ' clock' token embeddings are identical
    print("\n" + "="*60)
    print("VERIFYING ' clock' TOKEN EMBEDDINGS")
    print("="*60)
    
    # Get ' clock' indices from single layer
    if ' clock' in single_tok2idx:
        clock_indices_single = single_tok2idx[' clock']
        print(f"' clock' indices in single layer: {clock_indices_single[:5]}")
        
        # These same indices in combined (with offset)
        clock_indices_combined = [layer_24_start + idx for idx in clock_indices_single]
        print(f"' clock' indices in combined (with offset): {clock_indices_combined[:5]}")
        
        # Compare embeddings
        for i, (single_idx, combined_idx) in enumerate(zip(clock_indices_single[:3], clock_indices_combined[:3])):
            single_clock_emb = single_emb[single_idx]
            combined_clock_emb = combined_embeddings[combined_idx]
            diff = torch.abs(single_clock_emb - combined_clock_emb).max().item()
            print(f"  Instance {i}: max diff = {diff}")

if __name__ == "__main__":
    main()

