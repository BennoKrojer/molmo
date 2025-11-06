"""
Efficiently precompute embedding caches for contextual embeddings.

This script converts the many individual .npy files into single cached .pt files,
which dramatically speeds up loading in downstream scripts.

Uses multiprocessing to parallelize across different layer directories.

Usage:
    python scripts/analysis/precompute_contextual_caches.py --contextual-base molmo_data/contextual_llm_embeddings --num-workers 8
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from datetime import datetime


def convert_from_stored_dtype(embedding_array):
    """Convert embedding from stored dtype back to float32 for computation."""
    dtype_str = str(embedding_array.dtype)
    
    # Check if it's a void type (raw bytes from float8)
    if dtype_str.startswith('|V') or dtype_str.startswith('V'):
        # This is float8 saved as raw bytes, need to reinterpret
        import ml_dtypes
        # Reinterpret the raw bytes as float8, then convert to float32
        embedding_fp8 = embedding_array.view(ml_dtypes.float8_e4m3fn)
        return embedding_fp8.astype(np.float32)
    elif 'float8' in dtype_str:
        import ml_dtypes
        return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float16:
        return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float32:
        return embedding_array
    else:
        # Try generic conversion
        return embedding_array.astype(np.float32)


def load_single_embedding(args_tuple):
    """Load a single embedding file (for parallel I/O)."""
    embedding_path, token_str, emb_info = args_tuple
    
    try:
        if not embedding_path.exists():
            return None
        
        # Load numpy array with memory mapping
        embedding = np.load(embedding_path, allow_pickle=False, mmap_mode='r')
        
        if not isinstance(embedding, np.ndarray):
            return None
        
        # Convert to float32 and make a copy
        embedding_fp32 = convert_from_stored_dtype(embedding).copy()
        
        # Return embedding and metadata
        return {
            'embedding': embedding_fp32,
            'metadata': {
                'token_str': token_str,
                'token_id': emb_info['token_id'],
                'caption': emb_info['caption'],
                'position': emb_info['position']
            }
        }
    except Exception:
        return None


def build_cache_for_layer(args_tuple):
    """
    Build cache for a single layer directory.
    
    Args:
        args_tuple: (layer_dir, max_per_token, force_rebuild)
    
    Returns:
        (layer_dir, success, message)
    """
    layer_dir, max_per_token, force_rebuild = args_tuple
    layer_dir = Path(layer_dir)
    
    # Define cache file path
    cache_suffix = f"_max{max_per_token}" if max_per_token is not None else ""
    cache_file = layer_dir / f"embeddings_cache{cache_suffix}.pt"
    
    # Check if cache already exists and validate it
    if cache_file.exists() and not force_rebuild:
        try:
            # Quick validation: load and check integrity
            cache_data = torch.load(cache_file, map_location='cpu')
            embeddings = cache_data.get('embeddings')
            metadata = cache_data.get('metadata')
            
            if embeddings is not None and metadata is not None:
                if embeddings.shape[0] == len(metadata):
                    return (str(layer_dir), True, f"Cache already exists (validated: {len(metadata)} embeddings)")
                else:
                    print(f"[{layer_dir.parent.name}/{layer_dir.name}] Cache corrupted (shape mismatch), rebuilding...", flush=True)
            else:
                print(f"[{layer_dir.parent.name}/{layer_dir.name}] Cache incomplete, rebuilding...", flush=True)
        except Exception as e:
            print(f"[{layer_dir.parent.name}/{layer_dir.name}] Cache load failed ({e}), rebuilding...", flush=True)
    
    # Print start message
    layer_name = layer_dir.parent.name + "/" + layer_dir.name
    start_time = time.time()
    print(f"[STARTING] {layer_name} at {datetime.now().strftime('%H:%M:%S')}", flush=True)
    
    # Load token embeddings metadata
    token_embeddings_file = layer_dir / "token_embeddings.json"
    if not token_embeddings_file.exists():
        return (str(layer_dir), False, f"token_embeddings.json not found")
    
    # Load metadata
    json_start = time.time()
    with open(token_embeddings_file, 'r') as f:
        token_dict = json.load(f)
    json_time = time.time() - json_start
    
    print(f"[{layer_name}] Found {len(token_dict)} unique tokens (JSON load: {json_time:.2f}s), loading embeddings with multi-threaded I/O...", flush=True)
    
    # Prepare all file loading tasks
    loading_tasks = []
    for token_str, embeddings_info in token_dict.items():
        # Handle both list and dict formats
        if isinstance(embeddings_info, dict):
            embeddings_info = embeddings_info.get('preferred', []) + embeddings_info.get('fallback', [])
        
        # Limit per token if specified
        if max_per_token is not None:
            embeddings_info = embeddings_info[:max_per_token]
        
        for emb_info in embeddings_info:
            if not isinstance(emb_info, dict):
                continue
            
            embedding_path = layer_dir / emb_info['embedding_path']
            loading_tasks.append((embedding_path, token_str, emb_info))
    
    print(f"[{layer_name}] Prepared {len(loading_tasks)} file loading tasks, starting parallel I/O...", flush=True)
    
    # Load embeddings using thread pool for parallel I/O
    embeddings_batches = []
    metadata_list = []
    failed_count = 0
    current_batch = []
    batch_size = 10000
    
    last_print_count = 0
    last_print_time = time.time()
    print_interval = 1000
    loading_start = time.time()
    
    # Use 32 threads for I/O operations (doesn't consume much CPU, just waits for disk)
    with ThreadPoolExecutor(max_workers=32) as executor:
        for result in executor.map(load_single_embedding, loading_tasks):
            if result is None:
                failed_count += 1
                continue
            
            current_batch.append(result['embedding'])
            metadata_list.append(result['metadata'])
            
            # Stack current batch when it reaches batch_size
            if len(current_batch) >= batch_size:
                embeddings_batches.append(np.stack(current_batch, axis=0))
                current_batch = []
            
            # Print progress
            total_loaded = len(embeddings_batches) * batch_size + len(current_batch)
            if total_loaded >= last_print_count + print_interval:
                current_time = time.time()
                elapsed_total = current_time - loading_start
                elapsed_interval = current_time - last_print_time
                rate_interval = print_interval / elapsed_interval if elapsed_interval > 0 else 0
                rate_total = total_loaded / elapsed_total if elapsed_total > 0 else 0
                percent = int((total_loaded / len(loading_tasks)) * 100)
                print(f"[{layer_name}] {datetime.now().strftime('%H:%M:%S')} | {total_loaded} embs loaded ({percent}%) | rate: {rate_interval:.1f}/s (avg: {rate_total:.1f}/s) | elapsed: {elapsed_total:.1f}s", flush=True)
                last_print_count = total_loaded
                last_print_time = current_time
    
    # Handle any remaining embeddings in current_batch
    if len(current_batch) > 0:
        embeddings_batches.append(np.stack(current_batch, axis=0))
    
    total_embeddings = len(metadata_list)
    if total_embeddings == 0:
        return (str(layer_dir), False, "No embeddings loaded")
    
    loading_time = time.time() - loading_start
    print(f"[{layer_name}] Loading complete: {total_embeddings} embeddings in {loading_time:.1f}s ({total_embeddings/loading_time:.1f} embs/s avg)", flush=True)
    
    # Concatenate all batches into single matrix
    stack_start = time.time()
    if len(embeddings_batches) == 1:
        embeddings_matrix = torch.from_numpy(embeddings_batches[0])
    else:
        embeddings_matrix = torch.from_numpy(np.concatenate(embeddings_batches, axis=0))
    stack_time = time.time() - stack_start
    print(f"[{layer_name}] Concatenation of {len(embeddings_batches)} batches complete in {stack_time:.2f}s", flush=True)
    
    # Build token_to_indices mapping
    mapping_start = time.time()
    token_to_indices = defaultdict(list)
    for idx, meta in enumerate(metadata_list):
        token_to_indices[meta['token_str']].append(idx)
    token_to_indices = dict(token_to_indices)
    mapping_time = time.time() - mapping_start
    print(f"[{layer_name}] Token mapping complete in {mapping_time:.2f}s", flush=True)
    
    # Save to cache (atomically to prevent partial saves)
    save_start = time.time()
    try:
        cache_data = {
            'embeddings': embeddings_matrix,
            'metadata': metadata_list,
            'token_to_indices': token_to_indices
        }
        
        # Save to temporary file first, then atomically rename
        temp_cache_file = cache_file.with_suffix('.pt.tmp')
        torch.save(cache_data, temp_cache_file)
        temp_cache_file.rename(cache_file)  # Atomic on most filesystems
        
        save_time = time.time() - save_start
        
        total_time = time.time() - start_time
        message = f"Cached {total_embeddings} embeddings ({len(token_to_indices)} unique tokens) in {total_time:.1f}s (save: {save_time:.2f}s)"
        if failed_count > 0:
            message += f", {failed_count} failed"
        
        print(f"[COMPLETED] {layer_name} at {datetime.now().strftime('%H:%M:%S')}: {message}", flush=True)
        return (str(layer_dir), True, message)
    except Exception as e:
        # Clean up temp file if save failed
        temp_cache_file = cache_file.with_suffix('.pt.tmp')
        if temp_cache_file.exists():
            temp_cache_file.unlink()
        return (str(layer_dir), False, f"Failed to save cache: {e}")


def find_all_layer_dirs(contextual_base):
    """
    Find all layer directories across all LLM subdirectories.
    
    Returns:
        List of (llm_name, layer_dir_path) tuples
    """
    contextual_base = Path(contextual_base)
    layer_dirs = []
    
    # Find all LLM subdirectories
    for llm_dir in contextual_base.iterdir():
        if not llm_dir.is_dir():
            continue
        
        # Find all layer_* directories
        for layer_dir in llm_dir.glob("layer_*"):
            if layer_dir.is_dir():
                layer_dirs.append((llm_dir.name, layer_dir))
    
    return layer_dirs


def main():
    parser = argparse.ArgumentParser(description="Precompute embedding caches for contextual embeddings")
    parser.add_argument("--contextual-base", type=str, 
                       default="molmo_data/contextual_llm_embeddings",
                       help="Base directory containing all contextual embeddings")
    parser.add_argument("--llm-names", type=str, nargs="+", default=None,
                       help="Specific LLM names to process (default: all)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Specific layer indices to process (default: all)")
    parser.add_argument("--max-per-token", type=int, default=None,
                       help="Maximum embeddings per token (default: no limit)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild even if cache already exists")
    args = parser.parse_args()
    
    # Set number of workers 
    # Note: Default to 4 workers to avoid disk I/O contention when loading 300k+ small files per layer
    # Using too many workers actually slows things down due to disk thrashing
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 4  # Conservative default to avoid disk contention
    
    print(f"{'='*80}")
    print(f"Precomputing Contextual Embedding Caches")
    print(f"{'='*80}\n")
    print(f"⚠️  WARNING: Each layer has ~300,000 tiny .npy files to load")
    print(f"⚠️  This is a ONE-TIME preprocessing step that takes ~4-8 hours")
    print(f"⚠️  After completion, all future runs will load in SECONDS from cache")
    print(f"⚠️  This is much better than spending hours loading on every analysis run!\n")
    print(f"Base directory: {args.contextual_base}")
    print(f"Number of workers: {num_workers}")
    print(f"Force rebuild: {args.force_rebuild}")
    print(f"Max per token: {args.max_per_token}")
    print()
    
    # Find all layer directories
    print("Scanning for layer directories...")
    all_layer_dirs = find_all_layer_dirs(args.contextual_base)
    
    # Filter by LLM name if specified
    if args.llm_names is not None:
        all_layer_dirs = [(llm, path) for llm, path in all_layer_dirs if llm in args.llm_names]
    
    # Filter by layer index if specified
    if args.layers is not None:
        filtered_dirs = []
        for llm, path in all_layer_dirs:
            layer_num = int(path.name.replace("layer_", ""))
            if layer_num in args.layers:
                filtered_dirs.append((llm, path))
        all_layer_dirs = filtered_dirs
    
    if len(all_layer_dirs) == 0:
        print("No layer directories found!")
        return
    
    print(f"Found {len(all_layer_dirs)} layer directories to process:\n")
    
    # Group by LLM for display
    by_llm = defaultdict(list)
    for llm, path in all_layer_dirs:
        layer_num = int(path.name.replace("layer_", ""))
        by_llm[llm].append(layer_num)
    
    for llm, layers in sorted(by_llm.items()):
        print(f"  {llm}: layers {sorted(layers)}")
    print()
    
    # Prepare arguments for parallel processing
    process_args = [
        (path, args.max_per_token, args.force_rebuild)
        for llm, path in all_layer_dirs
    ]
    
    # Process in parallel
    print(f"Processing with {num_workers} workers...")
    print(f"{'='*80}\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    completed_count = 0
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for results as they complete
        results = pool.imap_unordered(build_cache_for_layer, process_args)
        
        for layer_dir, success, message in results:
            completed_count += 1
            layer_name = Path(layer_dir).parent.name + "/" + Path(layer_dir).name
            
            if success:
                if "already exists" in message:
                    skip_count += 1
                    status = "SKIP"
                else:
                    success_count += 1
                    status = "✓"
            else:
                fail_count += 1
                status = "✗"
            
            print(f"[{status}] {layer_name}: {message} [{completed_count}/{len(process_args)} completed] at {datetime.now().strftime('%H:%M:%S')}", flush=True)
    
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total:    {len(process_args)}")
    print(f"Success:  {success_count}")
    print(f"Skipped:  {skip_count}")
    print(f"Failed:   {fail_count}")
    print(f"{'='*80}\n")
    
    if fail_count > 0:
        print("Some caches failed to build. Check the error messages above.")
        sys.exit(1)
    else:
        print("✓ All caches built successfully!")


if __name__ == "__main__":
    main()

