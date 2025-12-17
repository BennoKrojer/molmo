#!/usr/bin/env python3
"""
Debug script to investigate why allLayers search misses higher-similarity matches.

Hypothesis: The " clock" token with 0.449 similarity from layer 24 should be in the
combined embeddings, but the allLayers search only found " crown" from layer 8 with 0.43.
"""

import json
import torch
from pathlib import Path
from collections import defaultdict

def load_single_layer_cache(contextual_dir, layer_idx):
    """Load cache for a single layer."""
    cache_file = Path(contextual_dir) / f"layer_{layer_idx}" / "embeddings_cache.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    
    cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
    return cache_data['embeddings'], cache_data['metadata'], cache_data['token_to_indices']

def main():
    contextual_dir = "molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"
    
    # Load layer 24 embeddings (the one that should contain the " clock" token)
    print("Loading layer 24 cache...")
    emb_24, meta_24, tok2idx_24 = load_single_layer_cache(contextual_dir, 24)
    print(f"  Layer 24: {emb_24.shape[0]} embeddings")
    
    # Find " clock" token instances
    if " clock" in tok2idx_24:
        clock_indices = tok2idx_24[" clock"]
        print(f"\n✓ Found ' clock' token with {len(clock_indices)} instances in layer 24")
        
        # Show first few instances
        for i, idx in enumerate(clock_indices[:3]):
            print(f"  Instance {i}: caption='{meta_24[idx]['caption']}', position={meta_24[idx]['position']}")
    else:
        print("\n✗ ' clock' token NOT FOUND in layer 24!")
        print("  Available tokens starting with 'clock':")
        for tok in tok2idx_24:
            if 'clock' in tok.lower():
                print(f"    '{tok}': {len(tok2idx_24[tok])} instances")
    
    # Now load ALL layers and verify they're combined correctly
    print("\n" + "="*60)
    print("Loading ALL layers...")
    
    available_layers = [1, 2, 4, 8, 16, 24, 30, 31]
    combined_embeddings_list = []
    combined_metadata_list = []
    layer_boundaries = []
    
    current_idx = 0
    for layer_idx in available_layers:
        emb, meta, _ = load_single_layer_cache(contextual_dir, layer_idx)
        
        start_idx = current_idx
        end_idx = current_idx + len(meta)
        layer_boundaries.append((layer_idx, start_idx, end_idx))
        
        combined_embeddings_list.append(emb)
        for m in meta:
            m['contextual_layer'] = layer_idx
        combined_metadata_list.extend(meta)
        
        current_idx = end_idx
        print(f"  Layer {layer_idx}: {len(meta)} embeddings, range [{start_idx}, {end_idx})")
    
    combined_embeddings = torch.cat(combined_embeddings_list, dim=0)
    print(f"\nTotal combined: {combined_embeddings.shape}")
    
    # Find layer 24 boundaries
    layer_24_bounds = [b for b in layer_boundaries if b[0] == 24][0]
    print(f"\nLayer 24 boundaries in combined array: [{layer_24_bounds[1]}, {layer_24_bounds[2]})")
    
    # Verify " clock" is in combined metadata at correct positions
    clock_in_combined = [i for i, m in enumerate(combined_metadata_list) 
                         if m.get('token_str') == ' clock' and m.get('contextual_layer') == 24]
    print(f"\n' clock' from layer 24 found at {len(clock_in_combined)} positions in combined metadata")
    if clock_in_combined:
        print(f"  First few indices: {clock_in_combined[:5]}")
        # Verify these are within layer 24 bounds
        in_bounds = all(layer_24_bounds[1] <= idx < layer_24_bounds[2] for idx in clock_in_combined)
        print(f"  All within layer 24 bounds: {in_bounds}")
    
    # Check if there's any normalization issue
    print("\n" + "="*60)
    print("Checking embedding norms...")
    
    # Sample some embeddings and check norms
    sample_indices = [0, 100000, 500000, 1000000, 1500000, 2000000]
    sample_indices = [i for i in sample_indices if i < combined_embeddings.shape[0]]
    
    for idx in sample_indices:
        norm = torch.norm(combined_embeddings[idx]).item()
        layer = [b[0] for b in layer_boundaries if b[1] <= idx < b[2]][0]
        print(f"  Index {idx} (layer {layer}): norm = {norm:.4f}")
    
    # Check if the embeddings are pre-normalized (norms close to 1)
    print("\nAre embeddings pre-normalized?")
    random_indices = torch.randint(0, combined_embeddings.shape[0], (100,))
    norms = torch.norm(combined_embeddings[random_indices], dim=-1)
    print(f"  Mean norm: {norms.mean().item():.4f}")
    print(f"  Std norm: {norms.std().item():.4f}")
    print(f"  Min/Max: {norms.min().item():.4f} / {norms.max().item():.4f}")

if __name__ == "__main__":
    main()

