#!/usr/bin/env python3
"""
Scan ALL .npy files in Qwen2 VG layers and identify which ones have NaN.
Saves list of corrupted files for cleanup.
"""

import numpy as np
import ml_dtypes
from pathlib import Path
from tqdm import tqdm
import json

def check_npy_file(npy_path):
    """Check if .npy file has NaN. Returns True if corrupted."""
    try:
        emb_raw = np.load(npy_path)
        emb_fp8 = emb_raw.view(ml_dtypes.float8_e4m3fn)
        emb_fp32 = emb_fp8.astype(np.float32)
        return np.isnan(emb_fp32).any()
    except Exception:
        return True  # Treat load errors as corrupted

def scan_layer(layer_dir):
    """Scan all .npy files in a layer directory."""
    import time
    
    embeddings_dir = layer_dir / "embeddings"
    
    if not embeddings_dir.exists():
        print(f"❌ Embeddings dir not found: {embeddings_dir}")
        return []
    
    print(f"  Listing files...", flush=True)
    list_start = time.time()
    npy_files = list(embeddings_dir.glob("emb_*.npy"))
    list_time = time.time() - list_start
    print(f"  Found {len(npy_files)} .npy files (listing took {list_time:.1f}s)", flush=True)
    
    # Time first 100 files to estimate total time
    print(f"  Timing first 100 files for ETA...", flush=True)
    timing_start = time.time()
    sample_corrupted = 0
    for i, npy_file in enumerate(npy_files[:100]):
        if check_npy_file(npy_file):
            sample_corrupted += 1
    timing_duration = time.time() - timing_start
    
    rate = 100 / timing_duration
    eta_seconds = len(npy_files) / rate
    eta_minutes = eta_seconds / 60
    eta_hours = eta_minutes / 60
    
    print(f"  Rate: {rate:.1f} files/sec", flush=True)
    print(f"  ETA for this layer: {eta_minutes:.1f} minutes ({eta_hours:.2f} hours)", flush=True)
    print(f"  Sample corruption rate: {sample_corrupted}/100 ({sample_corrupted}%)", flush=True)
    print(f"  Starting full scan...", flush=True)
    
    corrupted = []
    scan_start = time.time()
    for npy_file in tqdm(npy_files, desc=f"  Scanning", ncols=100):
        if check_npy_file(npy_file):
            corrupted.append(str(npy_file.relative_to(layer_dir)))
    
    scan_time = time.time() - scan_start
    actual_rate = len(npy_files) / scan_time
    print(f"  Scan complete in {scan_time/60:.1f} minutes (actual rate: {actual_rate:.1f} files/sec)", flush=True)
    
    return corrupted

def main():
    base_dir = Path("molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B")
    
    # Layers to check (we know 1,2 are clean, 4+ are affected)
    layers_to_check = [4, 8, 16, 24, 26, 27]
    
    print(f"{'='*80}")
    print(f"Scanning Qwen2 VG embeddings for corrupted .npy files")
    print(f"{'='*80}\n")
    
    all_corrupted = {}
    
    for layer_num in layers_to_check:
        layer_dir = base_dir / f"layer_{layer_num}"
        
        if not layer_dir.exists():
            print(f"Layer {layer_num}: Directory not found, skipping")
            continue
        
        print(f"\nLayer {layer_num}:")
        corrupted = scan_layer(layer_dir)
        
        corrupted_pct = len(corrupted) / len(list((layer_dir / "embeddings").glob("emb_*.npy"))) * 100 if corrupted else 0
        print(f"  Corrupted: {len(corrupted)} ({corrupted_pct:.2f}%)")
        
        if corrupted:
            all_corrupted[layer_num] = corrupted
    
    # Save results
    output_file = Path("corrupted_embeddings_list.json")
    with open(output_file, 'w') as f:
        json.dump(all_corrupted, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    total_corrupted = sum(len(files) for files in all_corrupted.values())
    print(f"Total corrupted files: {total_corrupted}")
    print(f"Affected layers: {list(all_corrupted.keys())}")
    print(f"\nCorrupted file list saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the list")
    print(f"  2. Run cleanup script to delete corrupted files")
    print(f"  3. Regenerate missing embeddings")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

