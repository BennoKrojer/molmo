#!/usr/bin/env python3
"""
Quick check: Sample 100 random files from each layer to estimate corruption rate.
Fast way to see which layers are affected.
"""

import numpy as np
import ml_dtypes
from pathlib import Path
import random

def check_layer_sample(layer_num, sample_size=100):
    """Check random sample of files from a layer."""
    layer_dir = Path(f"molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B/layer_{layer_num}")
    embeddings_dir = layer_dir / "embeddings"
    
    if not embeddings_dir.exists():
        return None, "Directory not found"
    
    npy_files = list(embeddings_dir.glob("emb_*.npy"))
    if len(npy_files) == 0:
        return None, "No .npy files"
    
    sample = random.sample(npy_files, min(sample_size, len(npy_files)))
    
    corrupted = 0
    for npy_file in sample:
        try:
            emb_raw = np.load(npy_file)
            emb_fp8 = emb_raw.view(ml_dtypes.float8_e4m3fn)
            emb_fp32 = emb_fp8.astype(np.float32)
            if np.isnan(emb_fp32).any():
                corrupted += 1
        except Exception:
            corrupted += 1
    
    return corrupted, f"{corrupted}/{len(sample)} corrupted ({corrupted/len(sample)*100:.1f}%)"

print("Checking Qwen2 VG layers for corruption (100 random samples per layer)...\n")

layers = [1, 2, 4, 8, 16, 24, 26, 27]

for layer_num in layers:
    corrupted_count, result = check_layer_sample(layer_num)
    
    if corrupted_count is None:
        status = "⚠"
        print(f"{status} Layer {layer_num:2d}: {result}")
    elif corrupted_count == 0:
        status = "✅"
        print(f"{status} Layer {layer_num:2d}: CLEAN - {result}")
    else:
        status = "❌"
        print(f"{status} Layer {layer_num:2d}: CORRUPTED - {result}")

print("\nRecommendation:")
print("  - Delete and regenerate only the ❌ corrupted layers")
print("  - Keep ✅ clean layers to save time")

