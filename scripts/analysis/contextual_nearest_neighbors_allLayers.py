"""
Find nearest contextual text embeddings for visual tokens.

This script takes visual tokens from a Molmo checkpoint and finds their top-5 nearest neighbors
in the contextual text embedding space (saved by create_contextual_embeddings.py).

Similar to general_and_nearest_neighbors_pixmo_cap_multi-gpu.py, but instead of comparing
visual tokens to static vocabulary embeddings, we compare them to contextualized word
embeddings from real captions. This tells us which real words in real contexts are most
similar to each visual token.

KEY DIFFERENCE: For each visual layer, this script compares vision tokens to contextual
embeddings from ALL layers (not just the same layer). This allows us to see which layer's
contextual embeddings are most similar to each vision token.

This is the multi-GPU version that uses FSDP for model sharding.

Usage:
    torchrun --nproc_per_node=2 scripts/analysis/contextual_nearest_neighbors_allLayers.py --ckpt-path <path> --contextual-dir <dir> --visual-layer 4 --num-images 100
"""

import argparse
import gc
import json
import math
import os
import pickle
import tempfile
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

# Try to import FAISS for fast vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap
from olmo.torch_util import get_local_rank, get_world_size


def convert_from_stored_dtype(embedding_array):
    """Convert embedding from stored dtype back to float32 for computation."""
    dtype_str = str(embedding_array.dtype)
    
    # Check if it's a void type (raw bytes from float8)
    if dtype_str.startswith('|V') or dtype_str.startswith('V'):
        # This is float8 saved as raw bytes, need to reinterpret
        try:
            import ml_dtypes
            # Reinterpret the raw bytes as float8, then convert to float32
            embedding_fp8 = embedding_array.view(ml_dtypes.float8_e4m3fn)
            return embedding_fp8.astype(np.float32)
        except ImportError:
            print("Warning: ml_dtypes not available, cannot load fp8 embeddings")
            raise
    elif 'float8' in dtype_str:
        try:
            import ml_dtypes
            return embedding_array.astype(np.float32)
        except ImportError:
            print("Warning: ml_dtypes not available, may have issues with fp8")
            return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float16:
        return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float32:
        return embedding_array
    else:
        # Try generic conversion
        return embedding_array.astype(np.float32)


def find_available_layers(contextual_dir):
    """
    Find all available layer directories in contextual_dir.
    
    Args:
        contextual_dir: Base directory for contextual embeddings
    
    Returns:
        List of available layer indices (sorted)
    """
    contextual_path = Path(contextual_dir)
    available_layers = []
    
    for layer_dir in contextual_path.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            try:
                layer_idx = int(layer_dir.name.split("_")[1])
                # Check if cache file exists
                cache_file = layer_dir / "embeddings_cache.pt"
                if cache_file.exists():
                    available_layers.append(layer_idx)
            except (ValueError, IndexError):
                continue
    
    return sorted(available_layers)


def load_contextual_embeddings(contextual_dir, layer_idx, max_per_token=None):
    """
    Load contextual text embeddings from disk.
    
    Args:
        contextual_dir: Base directory for contextual embeddings (e.g., molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview)
        layer_idx: Layer index to load
        max_per_token: Maximum embeddings to load per token (for memory management)
    
    Returns:
        embeddings_matrix: torch.Tensor [num_embeddings, hidden_dim]
        embeddings_metadata: List of dicts with caption, position, token info
        token_to_indices: Dict mapping token_str to list of indices in embeddings_matrix
    """
    layer_dir = Path(contextual_dir) / f"layer_{layer_idx}"
    token_embeddings_file = layer_dir / "token_embeddings.json"
    
    if not token_embeddings_file.exists():
        raise FileNotFoundError(f"Token embeddings file not found: {token_embeddings_file}")
    
    # Define cache file path
    cache_suffix = f"_max{max_per_token}" if max_per_token is not None else ""
    cache_file = layer_dir / f"embeddings_cache{cache_suffix}.pt"
    
    # Check if cache exists - REQUIRED!
    if not cache_file.exists():
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"ERROR: Cache file not found: {cache_file}\n"
            f"{'='*80}\n\n"
            f"This script now REQUIRES pre-built cache files to avoid disk contention.\n"
            f"Please run the cache building script first:\n\n"
            f"  python3 scripts/analysis/precompute_contextual_caches.py --num-workers 1\n\n"
            f"This will build caches for all layers. Once caches exist, this script will be fast.\n"
            f"{'='*80}\n"
        )
    
    # Load from cache
    print(f"Loading contextual embeddings from cache: {cache_file}")
    try:
        cache_data = torch.load(cache_file, map_location='cpu')
        embeddings_matrix = cache_data['embeddings']
        metadata_list = cache_data['metadata']
        token_to_indices = cache_data.get('token_to_indices', None)
        
        # Validate cache integrity
        if embeddings_matrix.shape[0] != len(metadata_list):
            raise ValueError(
                f"Cache corrupted! Embeddings shape {embeddings_matrix.shape} "
                f"doesn't match metadata length {len(metadata_list)}. "
                f"Please rebuild cache with --force-rebuild flag."
            )
        
        # Build token_to_indices if not in cache (for backward compatibility)
        if token_to_indices is None:
            print("Building token_to_indices mapping...")
            token_to_indices = defaultdict(list)
            for idx, meta in enumerate(metadata_list):
                token_to_indices[meta['token_str']].append(idx)
            token_to_indices = dict(token_to_indices)
        
        print(f"✓ Loaded {len(metadata_list)} contextual embeddings from cache with shape {embeddings_matrix.shape}")
        return embeddings_matrix, metadata_list, token_to_indices
    except Exception as e:
        raise RuntimeError(
            f"Failed to load cache: {e}\n"
            f"Cache file may be corrupted. Please rebuild with:\n"
            f"  python3 scripts/analysis/precompute_contextual_caches.py --force-rebuild"
        )
    
    # OLD CODE BELOW - Never reached, but keeping for reference
    # Load from individual files (cache miss or corrupted cache)
    print(f"Loading contextual embeddings from {token_embeddings_file}")
    with open(token_embeddings_file, 'r') as f:
        token_dict = json.load(f)
    
    print(f"Found {len(token_dict)} unique tokens")
    
    # Load all embeddings into memory
    embeddings_list = []
    metadata_list = []
    
    for token_str, embeddings_info in tqdm(token_dict.items(), desc="Loading embeddings"):
        # Handle both list and dict formats
        if isinstance(embeddings_info, dict):
            embeddings_info = embeddings_info.get('preferred', []) + embeddings_info.get('fallback', [])
        
        # Limit per token if specified
        if max_per_token is not None:
            embeddings_info = embeddings_info[:max_per_token]
        
        for emb_info in embeddings_info:
            if not isinstance(emb_info, dict):
                continue
            
            # Load embedding
            embedding_path = layer_dir / emb_info['embedding_path']
            if not embedding_path.exists():
                continue
            
            try:
                # Load numpy array - allow_pickle=False for security, use mmap_mode for large files
                embedding = np.load(embedding_path, allow_pickle=False)
                
                # Ensure it's a numpy array
                if not isinstance(embedding, np.ndarray):
                    print(f"Warning: Skipping non-array file: {embedding_path}")
                    continue
                
                # Convert to float32 if needed
                embedding_fp32 = convert_from_stored_dtype(embedding)
                embeddings_list.append(embedding_fp32)
                
                # Store metadata
                metadata_list.append({
                    'token_str': token_str,
                    'token_id': emb_info['token_id'],
                    'caption': emb_info['caption'],
                    'position': emb_info['position']
                })
            except Exception as e:
                print(f"Warning: Failed to load {embedding_path}: {e}")
                continue
    
    # Stack into matrix
    embeddings_matrix = torch.from_numpy(np.stack(embeddings_list, axis=0))
    print(f"Loaded {len(embeddings_list)} contextual embeddings with shape {embeddings_matrix.shape}")
    
    # Build token_to_indices mapping
    print("Building token_to_indices mapping...")
    token_to_indices = defaultdict(list)
    for idx, meta in enumerate(metadata_list):
        token_to_indices[meta['token_str']].append(idx)
    token_to_indices = dict(token_to_indices)
    print(f"Built mapping for {len(token_to_indices)} unique tokens")
    
    # Save to cache for future runs
    print(f"Saving embeddings to cache: {cache_file}")
    try:
        cache_data = {
            'embeddings': embeddings_matrix,
            'metadata': metadata_list,
            'token_to_indices': token_to_indices
        }
        torch.save(cache_data, cache_file)
        print(f"Cache saved successfully!")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return embeddings_matrix, metadata_list, token_to_indices


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def build_faiss_index(embeddings, use_gpu=False, approximate=False, quantized=False, 
                      nlist=4096, nprobe=64, m=64, nbits=8, local_rank=0):
    """
    Build a FAISS index for fast similarity search.
    
    Args:
        embeddings: torch.Tensor [num_embeddings, hidden_dim] - normalized embeddings
        use_gpu: Whether to use GPU for search (WARNING: requires GPU memory for embeddings)
        approximate: If True, use approximate search (IVF). If False, use exact search
        quantized: If True, use Product Quantization (PQ) to compress embeddings (4-16x memory reduction)
        nlist: Number of clusters for IVF (only used if approximate=True)
        nprobe: Number of clusters to search in IVF (only used if approximate=True)
        m: Number of subquantizers for PQ (only used if quantized=True). Must divide embedding dim.
        nbits: Bits per subquantizer for PQ (only used if quantized=True). Default 8 (256 centroids).
        local_rank: Rank for printing (default: 0)
    
    Returns:
        faiss_index: FAISS index ready for search
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
    
    d = embeddings.shape[1]  # Dimension
    
    # Convert to numpy float32 (FAISS requirement)
    # Keep on CPU to avoid GPU OOM - FAISS can search efficiently on CPU
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    
    if quantized:
        # Use Product Quantization (PQ) - compresses embeddings significantly
        # Memory reduction: original_size / (m * 2^nbits / 8)
        # Example: m=64, nbits=8 → ~16x compression
        if d % m != 0:
            raise ValueError(f"Embedding dimension {d} must be divisible by m={m} for PQ")
        
        if approximate:
            # IVF + PQ: Approximate search with compression
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
            index.nprobe = nprobe
            
            if local_rank == 0:
                compression_ratio = (m * (2**nbits)) / (8 * d)
                print(f"Building quantized approximate FAISS index (IVF+PQ):")
                print(f"  Compression: ~{compression_ratio:.1f}x memory reduction")
                print(f"  nlist={nlist}, nprobe={nprobe}, m={m}, nbits={nbits}")
        else:
            # PQ only: Approximate search with compression (PQ is approximate by nature)
            index = faiss.IndexPQ(d, m, nbits)
            
            if local_rank == 0:
                compression_ratio = (m * (2**nbits)) / (8 * d)
                print(f"Building quantized FAISS index (PQ) - NOTE: PQ is approximate (compression introduces small errors):")
                print(f"  Compression: ~{compression_ratio:.1f}x memory reduction")
                print(f"  m={m}, nbits={nbits}")
                print(f"  Accuracy: Typically 98-99% (small approximation due to compression)")
        
        # Train and add
        index.train(embeddings_np)
        index.add(embeddings_np)
        
    elif approximate:
        # Use IVF (Inverted File Index) for approximate search - faster but no compression
        quantizer = faiss.IndexFlatIP(d)  # Inner product (cosine similarity for normalized vectors)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.nprobe = nprobe
        
        if local_rank == 0:
            print(f"Building approximate FAISS index (IVF) with {nlist} clusters, nprobe={nprobe}...")
        
        # Train the index
        index.train(embeddings_np)
        index.add(embeddings_np)
    else:
        # Use exact search - still faster than PyTorch for large batches
        index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity for normalized vectors
        
        if local_rank == 0:
            print(f"Building exact FAISS index (IndexFlatIP)...")
        
        index.add(embeddings_np)
    
    # Move to GPU if requested and available
    # WARNING: This still requires GPU memory, so only use if you have enough VRAM
    if use_gpu and torch.cuda.is_available():
        if local_rank == 0:
            print(f"Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    return index


def search_faiss(index, query_embeddings, top_k, use_gpu=False):
    """
    Search FAISS index for nearest neighbors.
    
    Args:
        index: FAISS index (can be on CPU or GPU)
        query_embeddings: torch.Tensor [num_queries, hidden_dim] - normalized query embeddings
        top_k: Number of neighbors to return
        use_gpu: Whether index is on GPU (if False, searches on CPU even if embeddings are on GPU)
    
    Returns:
        distances: numpy array [num_queries, top_k] - similarity scores
        indices: numpy array [num_queries, top_k] - indices of nearest neighbors
    """
    # Convert to numpy float32 (FAISS requirement)
    # Always convert to CPU numpy - FAISS handles CPU/GPU internally
    if isinstance(query_embeddings, torch.Tensor):
        query_np = query_embeddings.cpu().numpy().astype('float32')
    else:
        query_np = query_embeddings.astype('float32')
    
    # Search (FAISS handles CPU/GPU automatically based on index location)
    distances, indices = index.search(query_np, top_k)
    
    return distances, indices


def distributed_topk_search(query_embeddings, contextual_embeddings_shard, global_offset, top_k, world_size):
    """
    Distributed exact nearest neighbor search across sharded embeddings.
    
    Each GPU holds a shard of the contextual embeddings. This function:
    1. Computes local top-k on this GPU's shard
    2. All-gathers top-k from all GPUs
    3. Merges to find global top-k
    
    Args:
        query_embeddings: torch.Tensor [num_queries, hidden_dim] - normalized query embeddings (on this GPU)
        contextual_embeddings_shard: torch.Tensor [shard_size, hidden_dim] - this GPU's shard of embeddings (normalized)
        global_offset: int - starting index of this shard in the global embedding array
        top_k: int - number of neighbors to return
        world_size: int - number of GPUs
    
    Returns:
        global_top_values: torch.Tensor [num_queries, top_k] - similarity scores
        global_top_indices: torch.Tensor [num_queries, top_k] - global indices
    """
    device = query_embeddings.device
    num_queries = query_embeddings.shape[0]
    
    # Step 1: Local similarity computation
    # [num_queries, shard_size]
    local_similarity = torch.matmul(query_embeddings, contextual_embeddings_shard.T)
    
    # Step 2: Local top-k
    local_k = min(top_k, local_similarity.shape[-1])
    local_values, local_indices = torch.topk(local_similarity, k=local_k, dim=-1)
    
    # Convert local indices to global indices
    local_indices_global = local_indices + global_offset
    
    # Step 3: All-gather from all GPUs
    # Gather top-k values and indices from all ranks
    gathered_values_list = [torch.zeros_like(local_values) for _ in range(world_size)]
    gathered_indices_list = [torch.zeros_like(local_indices_global) for _ in range(world_size)]
    
    dist.all_gather(gathered_values_list, local_values)
    dist.all_gather(gathered_indices_list, local_indices_global)
    
    # Step 4: Concatenate and find global top-k
    # [num_queries, world_size * local_k]
    all_values = torch.cat(gathered_values_list, dim=-1)
    all_indices = torch.cat(gathered_indices_list, dim=-1)
    
    # Final top-k across all shards
    global_top_values, merge_indices = torch.topk(all_values, k=top_k, dim=-1)
    
    # Map merge_indices to actual global indices
    global_top_indices = torch.gather(all_indices, dim=-1, index=merge_indices)
    
    return global_top_values, global_top_indices


def load_all_contextual_embeddings(contextual_dir, available_layers, max_per_token=None):
    """
    Load contextual embeddings from all available layers and combine them.
    
    Args:
        contextual_dir: Base directory for contextual embeddings
        available_layers: List of layer indices to load
        max_per_token: Maximum embeddings to load per token (for memory management)
    
    Returns:
        combined_embeddings: torch.Tensor [total_embeddings, hidden_dim]
        combined_metadata: List of dicts with caption, position, token info, and contextual_layer
        layer_boundaries: List of (layer_idx, start_idx, end_idx) tuples for each layer
        token_to_indices: Dict mapping token_str to list of indices in combined_embeddings
    """
    combined_embeddings_list = []
    combined_metadata_list = []
    layer_boundaries = []
    token_to_indices = defaultdict(list)
    
    current_idx = 0
    
    for layer_idx in available_layers:
        if current_idx == 0:
            print(f"Loading contextual embeddings from all layers...")
        
        print(f"  Loading layer {layer_idx}...")
        embeddings, metadata, token_to_indices_layer = load_contextual_embeddings(
            contextual_dir, layer_idx, max_per_token=max_per_token
        )
        
        # Add layer information to metadata
        for meta in metadata:
            meta['contextual_layer'] = layer_idx
        
        # Update token_to_indices with offset
        for token_str, indices in token_to_indices_layer.items():
            token_to_indices[token_str].extend([current_idx + idx for idx in indices])
        
        # Store boundaries for this layer
        start_idx = current_idx
        end_idx = current_idx + len(metadata)
        layer_boundaries.append((layer_idx, start_idx, end_idx))
        
        combined_embeddings_list.append(embeddings)
        combined_metadata_list.extend(metadata)
        
        current_idx = end_idx
        print(f"    ✓ Loaded {len(metadata)} embeddings (total so far: {current_idx})")
    
    # Concatenate all embeddings
    combined_embeddings = torch.cat(combined_embeddings_list, dim=0)
    token_to_indices = dict(token_to_indices)
    
    print(f"\n✓ Combined {len(combined_metadata_list)} contextual embeddings from {len(available_layers)} layers")
    print(f"  Shape: {combined_embeddings.shape}")
    print(f"  Unique tokens: {len(token_to_indices)}")
    
    return combined_embeddings, combined_metadata_list, layer_boundaries, token_to_indices


def shard_embeddings_for_distributed_search(embeddings, metadata, world_size, local_rank):
    """
    Shard embeddings across GPUs for distributed search.
    
    Each GPU gets every world_size-th embedding starting from local_rank.
    This ensures even distribution.
    
    Args:
        embeddings: torch.Tensor [total_embeddings, hidden_dim]
        metadata: List of metadata dicts
        world_size: Number of GPUs
        local_rank: This GPU's rank
    
    Returns:
        shard_embeddings: torch.Tensor [shard_size, hidden_dim] - this GPU's shard
        shard_indices: List[int] - global indices of embeddings in this shard
        global_offset: int - for contiguous sharding, the starting index
    """
    total_embeddings = embeddings.shape[0]
    
    # Use contiguous sharding for simplicity (each GPU gets a contiguous chunk)
    shard_size = total_embeddings // world_size
    start_idx = local_rank * shard_size
    
    # Last GPU gets the remainder
    if local_rank == world_size - 1:
        end_idx = total_embeddings
    else:
        end_idx = start_idx + shard_size
    
    shard_embeddings = embeddings[start_idx:end_idx]
    global_offset = start_idx
    
    if local_rank == 0:
        print(f"\n📊 Embedding sharding across {world_size} GPUs:")
        print(f"  Total embeddings: {total_embeddings:,}")
        print(f"  Per GPU: ~{shard_size:,}")
        print(f"  Memory per GPU: {shard_embeddings.numel() * 4 / 1e9:.2f} GB")
    
    return shard_embeddings, global_offset


def process_images(model, preprocessor, dataset, num_images, prompt, use_n_token_only, 
                   contextual_embeddings, contextual_metadata, token_to_indices, top_k=5, llm_layer=0,
                   use_faiss=False, faiss_index=None, faiss_cpu=False, 
                   contextual_embeddings_norm_cpu=None, contextual_embeddings_cpu=None,
                   use_distributed_gpu_search=False, embeddings_shard=None, 
                   embeddings_shard_norm=None, global_offset=0):
    """
    Process images and find nearest contextual neighbors for each visual token.
    
    Similar to process_split from general_and_nearest_neighbors_pixmo_cap_multi-gpu.py
    but finds nearest contextual embeddings instead of vocabulary tokens.
    
    For each of the top-k nearest neighbors, also finds the same token with lowest similarity.
    
    Uses distributed processing - each process handles a subset of images.
    
    Search modes:
    1. FAISS (use_faiss=True): Uses FAISS index for search
    2. Distributed GPU (use_distributed_gpu_search=True): Sharded GPU search with all-gather
    3. PyTorch (default): Brute-force matmul on single GPU
    """
    # Distribute images across processes
    # IMPORTANT: All ranks must do the same number of iterations due to FSDP synchronization
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    # Create list of image indices for each rank (round-robin distribution for balance)
    all_indices = list(range(num_images))
    rank_indices = [all_indices[i] for i in range(local_rank, num_images, world_size)]
    
    # All ranks must iterate the same number of times for FSDP sync
    max_iterations = (num_images + world_size - 1) // world_size  # ceil division
    
    if local_rank == 0:
        print(f"Process {local_rank}: Processing {len(rank_indices)} images (indices: {rank_indices[:5]}{'...' if len(rank_indices) > 5 else ''})")
        print(f"All ranks will iterate {max_iterations} times for FSDP sync")
    
    results = []
    device = torch.device(f"cuda:{local_rank}")
    
    # Timing stats
    timing_stats = {'preprocess': 0, 'model_forward': 0, 'faiss_search': 0, 'postprocess': 0}
    first_image_detailed = True
    
    for iteration in tqdm(range(max_iterations), desc=f"Rank {local_rank}"):
        # Get actual image index for this iteration (or use image 0 for padding)
        # Padding iterations are needed to keep all ranks in sync for FSDP/distributed ops
        is_padding = iteration >= len(rank_indices)
        i = rank_indices[iteration] if not is_padding else 0  # Use image 0 as dummy
        
        import time as _time
        t_start = _time.time()
        
        example_data = dataset.get(i, np.random)
        
        # Extract ground truth caption
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            caption_text = message.get("text", "")
        
        # Create example
        example = {
            "image": example_data["image"],
            "messages": [prompt]
        }
        
        # Preprocess
        batch = preprocessor(example, rng=np.random)
        t_preprocess = _time.time()
        timing_stats['preprocess'] += t_preprocess - t_start
        
        # Initialize results
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text,
            "chunks": []
        }
        
        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                
                # Extract features (vision backbone or LLM layer)
                if llm_layer == 0:
                    image_features, _ = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                    if type(use_n_token_only) == int and use_n_token_only != -1:
                        image_features = image_features[:, :, :use_n_token_only, :]
                    elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                        image_features = image_features[:, :, use_n_token_only, :]
                    t_model = _time.time()
                    timing_stats['model_forward'] += t_model - t_preprocess
                else:
                    # Forward through LLM
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
                    
                    # Extract visual tokens from LLM layer
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
                    
                    layer_index = min(llm_layer, len(hidden_states) - 1)
                    image_features = gather_visual_features(hidden_states[layer_index])
                
                image_results["feature_shape"] = list(image_features.shape)
                image_results["llm_layer_used"] = llm_layer
                
                t_model = _time.time()
                timing_stats['model_forward'] += t_model - t_preprocess
                
                # Batch process all patches at once (like general_and_nearest_neighbors_pixmo_cap_multi-gpu.py)
                batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                
                # Reshape to combine batch and chunks dimensions
                image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)
                
                # Normalize for cosine similarity
                image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
                
                if use_distributed_gpu_search and embeddings_shard_norm is not None:
                    # Distributed GPU search: each GPU has a shard, all-gather for global top-k
                    num_patches = image_features_norm.shape[0] * image_features_norm.shape[1]
                    patches_flat = image_features_norm.view(num_patches, -1)
                    
                    t_search_start = _time.time()
                    
                    # Distributed search across all GPU shards
                    top_values_tensor, top_indices_tensor = distributed_topk_search(
                        patches_flat, embeddings_shard_norm, global_offset, top_k, world_size
                    )
                    
                    t_search_end = _time.time()
                    timing_stats['faiss_search'] += t_search_end - t_search_start  # Reuse timing key
                    
                    # Reshape back to [batch*chunks, patches_per_chunk, top_k]
                    top_values = top_values_tensor.cpu().numpy().reshape(
                        image_features_norm.shape[0], image_features_norm.shape[1], top_k
                    )
                    top_indices = top_indices_tensor.cpu().numpy().reshape(
                        image_features_norm.shape[0], image_features_norm.shape[1], top_k
                    )
                    
                    # Print detailed timing for first image
                    if first_image_detailed and local_rank == 0:
                        first_image_detailed = False
                        shard_size = embeddings_shard_norm.shape[0]
                        print(f"\n⏱️  First image timing breakdown (Distributed GPU Search):")
                        print(f"  Preprocess:     {t_preprocess - t_start:.2f}s")
                        print(f"  Model forward:  {t_model - t_preprocess:.2f}s")
                        print(f"  GPU search:     {t_search_end - t_search_start:.2f}s ({num_patches} patches × {world_size} GPUs × {shard_size:,} embeddings each)")
                        total_time = t_search_end - t_start
                        est_total = total_time * len(rank_indices)
                        print(f"  Total per image: {total_time:.2f}s")
                        print(f"  Estimated total: {est_total/60:.1f} min for {len(rank_indices)} images")
                        print()
                    
                    del patches_flat
                    
                elif use_faiss and faiss_index is not None:
                    # Use FAISS for fast search
                    # Reshape to [num_patches, hidden_dim] for FAISS
                    num_patches = image_features_norm.shape[0] * image_features_norm.shape[1]
                    
                    # For CPU FAISS, convert to CPU numpy
                    if faiss_cpu:
                        patches_flat = image_features_norm.cpu().view(num_patches, -1)
                    else:
                        patches_flat = image_features_norm.view(num_patches, -1)
                    
                    # Search FAISS index
                    t_faiss_start = _time.time()
                    top_values_np, top_indices_np = search_faiss(
                        faiss_index, patches_flat, top_k, use_gpu=not faiss_cpu
                    )
                    t_faiss_end = _time.time()
                    timing_stats['faiss_search'] += t_faiss_end - t_faiss_start
                    
                    # Reshape back to [batch*chunks, patches_per_chunk, top_k]
                    top_values = top_values_np.reshape(image_features_norm.shape[0], image_features_norm.shape[1], top_k)
                    top_indices = top_indices_np.reshape(image_features_norm.shape[0], image_features_norm.shape[1], top_k)
                    
                    # Print detailed timing for first image
                    if first_image_detailed and local_rank == 0:
                        first_image_detailed = False
                        print(f"\n⏱️  First image timing breakdown:")
                        print(f"  Preprocess:     {t_preprocess - t_start:.2f}s")
                        print(f"  Model forward:  {t_model - t_preprocess:.2f}s")
                        print(f"  FAISS search:   {t_faiss_end - t_faiss_start:.2f}s ({num_patches} patches × 3.3M embeddings)")
                        total_time = t_faiss_end - t_start
                        est_total = total_time * len(rank_indices)
                        print(f"  Total per image: {total_time:.2f}s")
                        print(f"  Estimated total: {est_total/60:.1f} min for {len(rank_indices)} images")
                        print()
                    
                    # Clear intermediate tensors
                    del patches_flat
                else:
                    # Original PyTorch brute-force approach
                    contextual_norm = torch.nn.functional.normalize(contextual_embeddings, dim=-1)
                    
                    # Compute similarity for ALL patches at once
                    # similarity: [batch*chunks, patches_per_chunk, num_contextual_embeddings]
                    similarity = torch.matmul(image_features_norm, contextual_norm.T)
                    
                    # Get top-k for all patches at once
                    top_values, top_indices = torch.topk(similarity, k=min(top_k, similarity.shape[-1]), dim=-1)
                    
                    # Convert to CPU for processing
                    top_values = top_values.cpu().numpy()
                    top_indices = top_indices.cpu().numpy()
                    
                    # Clear similarity tensors to free GPU memory
                    del similarity, contextual_norm
                
                # Clear normalized features
                del image_features_norm
                
                # Process each chunk
                for chunk_idx in range(num_chunks):
                    chunk_results = {
                        "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                        "patches": []
                    }
                    
                    for patch_idx in range(patches_per_chunk):
                        # Get top-k results for this patch
                        patch_top_values = top_values[chunk_idx, patch_idx]
                        patch_top_indices = top_indices[chunk_idx, patch_idx]
                        
                        # Get the patch embedding for finding lowest similarities
                        patch_embedding = image_features_reshaped[chunk_idx, patch_idx:patch_idx+1, :]  # [1, hidden_dim]
                        patch_embedding_norm = torch.nn.functional.normalize(patch_embedding, dim=-1)
                        
                        # Build nearest neighbors list
                        nearest_contextual = []
                        top_nn_embeddings = []  # Store embeddings for inter-NN comparisons
                        
                        for val, idx in zip(patch_top_values, patch_top_indices):
                            idx = int(idx)
                            token_str = contextual_metadata[idx]['token_str']
                            
                            # Store embedding for later comparison
                            # Handle GPU, CPU, and distributed search cases
                            if use_distributed_gpu_search:
                                # Distributed GPU search - embeddings are on CPU
                                top_nn_embeddings.append(contextual_embeddings[idx])
                            elif use_faiss and faiss_cpu and contextual_embeddings_norm_cpu is not None:
                                top_nn_embeddings.append(contextual_embeddings_norm_cpu[idx])
                            elif use_faiss and not faiss_cpu:
                                # GPU FAISS - embeddings are on GPU
                                top_nn_embeddings.append(contextual_embeddings[idx])
                            else:
                                # PyTorch fallback - embeddings are on GPU
                                top_nn_embeddings.append(contextual_embeddings[idx])
                            
                            # Find the same token with lowest similarity
                            lowest_info = None
                            if token_str in token_to_indices:
                                same_token_indices = token_to_indices[token_str]
                                if len(same_token_indices) > 1:  # Only if there are multiple instances
                                    # Get embeddings for all instances of this token
                                    # Handle GPU, CPU, and distributed search cases
                                    if use_distributed_gpu_search:
                                        # Distributed GPU search - embeddings are on CPU
                                        same_token_embeddings = contextual_embeddings[same_token_indices]
                                        patch_embedding_norm_cpu = patch_embedding_norm.cpu()
                                        same_token_embeddings_norm = torch.nn.functional.normalize(same_token_embeddings, dim=-1)
                                        same_token_sims = torch.matmul(patch_embedding_norm_cpu, same_token_embeddings_norm.T).squeeze(0)
                                    elif use_faiss and faiss_cpu and contextual_embeddings_norm_cpu is not None:
                                        same_token_embeddings = contextual_embeddings_norm_cpu[same_token_indices]
                                        patch_embedding_norm_cpu = patch_embedding_norm.cpu()
                                        same_token_embeddings_norm = torch.nn.functional.normalize(same_token_embeddings, dim=-1)
                                        same_token_sims = torch.matmul(patch_embedding_norm_cpu, same_token_embeddings_norm.T).squeeze(0)
                                    elif use_faiss and not faiss_cpu:
                                        # GPU FAISS - use GPU embeddings
                                        same_token_embeddings = contextual_embeddings[same_token_indices]
                                        same_token_embeddings_norm = torch.nn.functional.normalize(same_token_embeddings, dim=-1)
                                        same_token_sims = torch.matmul(patch_embedding_norm, same_token_embeddings_norm.T).squeeze(0)
                                    else:
                                        # PyTorch fallback - use GPU embeddings
                                        same_token_embeddings = contextual_embeddings[same_token_indices]
                                        same_token_embeddings_norm = torch.nn.functional.normalize(same_token_embeddings, dim=-1)
                                        same_token_sims = torch.matmul(patch_embedding_norm, same_token_embeddings_norm.T).squeeze(0)
                                    
                                    # Find the lowest similarity
                                    min_idx = torch.argmin(same_token_sims).item()
                                    min_sim = same_token_sims[min_idx].item()
                                    min_global_idx = same_token_indices[min_idx]
                                    
                                    lowest_info = {
                                        'token_str': token_str,
                                        'token_id': contextual_metadata[min_global_idx]['token_id'],
                                        'caption': contextual_metadata[min_global_idx]['caption'],
                                        'position': contextual_metadata[min_global_idx]['position'],
                                        'similarity': float(min_sim),
                                        'num_instances': len(same_token_indices)
                                    }
                            
                            # Get contextual layer info if available
                            contextual_layer = contextual_metadata[idx].get('contextual_layer', None)
                            
                            nearest_contextual.append({
                                'token_str': token_str,
                                'token_id': contextual_metadata[idx]['token_id'],
                                'caption': contextual_metadata[idx]['caption'],
                                'position': contextual_metadata[idx]['position'],
                                'similarity': float(val),
                                'contextual_layer': contextual_layer,  # Track which layer this came from
                                'lowest_similarity_same_token': lowest_info
                            })
                        
                        # Compute similarities between 1st NN and other NNs (2nd, 3rd, 4th, 5th)
                        if len(top_nn_embeddings) > 1:
                            first_nn_embedding = top_nn_embeddings[0].unsqueeze(0)  # [1, hidden_dim]
                            first_nn_norm = torch.nn.functional.normalize(first_nn_embedding, dim=-1)
                            
                            # Compute similarities with other NNs
                            other_nns_embeddings = torch.stack(top_nn_embeddings[1:], dim=0)  # [k-1, hidden_dim]
                            other_nns_norm = torch.nn.functional.normalize(other_nns_embeddings, dim=-1)
                            
                            # Cosine similarities: [k-1]
                            inter_nn_sims = torch.matmul(first_nn_norm, other_nns_norm.T).squeeze(0)
                            inter_nn_sims = inter_nn_sims.cpu().numpy()
                            
                            # Add to first NN's data
                            nearest_contextual[0]['similarity_to_other_nns'] = {
                                f'vs_{i+2}': float(inter_nn_sims[i]) 
                                for i in range(len(inter_nn_sims))
                            }
                        
                        # Add row/col info
                        row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                        
                        patch_results = {
                            "patch_idx": patch_idx,
                            "patch_row": row,
                            "patch_col": col,
                            "nearest_contextual_neighbors": nearest_contextual
                        }
                        
                        chunk_results["patches"].append(patch_results)
                    
                    image_results["chunks"].append(chunk_results)
                
                # Clear intermediate tensors
                del images_tensor, image_masks_tensor, image_features
                clear_gpu_memory()
        
        # Only append results for actual images, not padding iterations
        if not is_padding:
            results.append(image_results)
        clear_gpu_memory()
    
    # Gather results from all processes using file-based approach to avoid GPU OOM
    # (all_gather_object uses GPU memory for serialization which can OOM with large results)
    
    # Save this rank's results to a temp file
    temp_dir = tempfile.gettempdir()
    results_file = os.path.join(temp_dir, f"nn_results_rank{local_rank}.pkl")
    print(f"[DEBUG] Rank {local_rank}: Writing {len(results)} results to {results_file}")
    print(f"[DEBUG] Rank {local_rank}: Image indices: {[r.get('image_idx') for r in results[:5]]}...")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Synchronize to ensure all ranks have written their files
    dist.barrier()
    
    # Combine results on main process
    if local_rank == 0:
        combined_results = []
        print(f"\n[DEBUG] Gathering results from {world_size} ranks...")
        for rank in range(world_size):
            rank_file = os.path.join(temp_dir, f"nn_results_rank{rank}.pkl")
            print(f"  [DEBUG] Reading {rank_file}...")
            with open(rank_file, 'rb') as f:
                rank_results = pickle.load(f)
            print(f"  [DEBUG] Rank {rank}: {len(rank_results)} images, indices: {[r.get('image_idx') for r in rank_results[:5]]}...")
            combined_results.extend(rank_results)
            # Clean up temp file
            os.remove(rank_file)
        print(f"[DEBUG] Total combined: {len(combined_results)} images")
        # Signal to other ranks that we're done reading
        dist.barrier()
        return combined_results
    else:
        # Wait for rank 0 to read all files before returning
        dist.barrier()
        return None

def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # Set CUDA device
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    
    parser = argparse.ArgumentParser(description="Find nearest contextual text embeddings for visual tokens (Multi-GPU)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint to analyze")
    parser.add_argument("--contextual-dir", type=str, default="molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview",
                       help="Directory with contextual embeddings (e.g., molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview)")
    parser.add_argument("--contextual-layer", type=str, default=None,
                       help="[DEPRECATED] Not used in allLayers version. All available layers are automatically loaded.")
    parser.add_argument("--visual-layer", type=str, required=True,
                       help="Visual layer(s) to extract. Single layer (e.g., '4') or comma-separated list (e.g., '4,8,16'). 0 = vision backbone, >0 = LLM layer. Vision tokens from these layers will be compared to ALL contextual layers.")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of nearest neighbors to find (default: 5)")
    parser.add_argument("--max-contextual-per-token", type=int, default=None,
                       help="Maximum contextual embeddings to load per token (for memory management)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/contextual_nearest_neighbors",
                       help="Output directory for results")
    parser.add_argument("--use-faiss", action="store_true",
                       help="Use FAISS for fast vector search (requires faiss-gpu or faiss-cpu)")
    parser.add_argument("--faiss-gpu", action="store_true",
                       help="Use GPU for FAISS index (default: CPU to avoid OOM. Only use if you have enough VRAM)")
    parser.add_argument("--faiss-approximate", action="store_true",
                       help="Use approximate FAISS search (IVF) instead of exact search. Much faster for large embedding sets.")
    parser.add_argument("--faiss-quantized", action="store_true",
                       help="Use Product Quantization (PQ) to compress embeddings. Reduces memory by 4-16x. Recommended for OOM issues.")
    parser.add_argument("--faiss-nlist", type=int, default=4096,
                       help="Number of clusters for approximate FAISS search (default: 4096)")
    parser.add_argument("--faiss-nprobe", type=int, default=64,
                       help="Number of clusters to probe in approximate FAISS search (default: 64)")
    parser.add_argument("--faiss-m", type=int, default=64,
                       help="Number of subquantizers for PQ (must divide embedding dim, default: 64)")
    parser.add_argument("--faiss-nbits", type=int, default=8,
                       help="Bits per subquantizer for PQ (default: 8, gives 256 centroids)")
    parser.add_argument("--distributed-gpu-search", action="store_true",
                       help="Use distributed GPU search (sharded across all GPUs). Fast and EXACT. Recommended for multi-GPU setups.")
    args = parser.parse_args()
    
    # Parse visual layers to process
    visual_layers = [int(layer.strip()) for layer in args.visual_layer.split(",")]
    
    # Find all available contextual layers
    available_contextual_layers = find_available_layers(args.contextual_dir)
    
    if len(available_contextual_layers) == 0:
        raise ValueError(
            f"No contextual layers found in {args.contextual_dir}. "
            f"Please ensure contextual embeddings have been created."
        )
    
    if local_rank == 0:
        print(f"{'='*80}")
        print(f"Visual-to-Contextual Nearest Neighbors Analysis (ALL LAYERS)")
        print(f"{'='*80}\n")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Contextual embeddings: {args.contextual_dir}")
        print(f"Visual layers to process: {visual_layers}")
        print(f"Available contextual layers: {available_contextual_layers}")
        print(f"  → Vision tokens from each visual layer will be compared to ALL contextual layers")
        print(f"Dataset split: {args.split}")
        print(f"Number of images: {args.num_images}")
        print(f"Top-k neighbors: {args.top_k}")
        print(f"Running on {world_size} processes")
        if args.use_faiss:
            if FAISS_AVAILABLE:
                search_type = "approximate (IVF)" if args.faiss_approximate else "exact"
                print(f"FAISS search: ENABLED ({search_type})")
            else:
                print(f"FAISS search: REQUESTED but NOT AVAILABLE (will use PyTorch)")
        else:
            print(f"FAISS search: DISABLED (using PyTorch brute-force)")
        print()
    
    # Load model with FSDP
    if local_rank == 0:
        print(f"Loading model from {args.ckpt_path}")
    
    # Load model on CPU first
    # Works with both full checkpoints and stripped (MLP-only) checkpoints
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = None  # Override init_device to avoid meta tensors
    
    model = Molmo(cfg.model)
    
    # Check checkpoint size to determine if it's a full or stripped checkpoint
    # Full checkpoints (with trained LLM/ViT) are ~29GB and contain all weights
    # Stripped checkpoints (connector-only) are ~200MB and only contain connector weights
    import os
    checkpoint_file = f"{args.ckpt_path}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    
    # If checkpoint is > 1GB, it's a full checkpoint with all weights
    is_full_checkpoint = checkpoint_size_gb > 1.0
    
    if not is_full_checkpoint:
        # Small checkpoint - only contains connector weights, need to load pretrained LLM/ViT
        if local_rank == 0:
            print(f"Detected stripped checkpoint ({checkpoint_size_gb:.2f} GB)")
            print("Loading pretrained weights (LLM + ViT)...")
        model.reset_with_pretrained_weights()
        if local_rank == 0:
            print("Pretrained weights loaded")
    else:
        if local_rank == 0:
            print(f"Detected full checkpoint ({checkpoint_size_gb:.2f} GB)")
            print("Skipping pretrained weights loading (checkpoint contains all weights)")
    
    # Load checkpoint weights (all ranks load in parallel for speed)
    if local_rank == 0:
        print("Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    
    # Free checkpoint memory immediately
    num_params = len(checkpoint_weights)
    del checkpoint_weights
    gc.collect()
    
    if local_rank == 0:
        print(f"Loaded {num_params} parameter tensors from checkpoint")
    
    model.eval()
    
    # Wrap model with FSDP for sharding
    if local_rank == 0:
        print("Wrapping model with FSDP for sharding...")
    
    # Get FSDP wrap policy from the model
    wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")
    
    # Wrap model in FSDP
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
        print(f"Model wrapped with FSDP on device: {device}\n")
    
    # Create preprocessor
    if "hf:" in args.ckpt_path:
        model_config = model.config
    else:
        model_config = ModelConfig.load(resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False)
    
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )
    
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    
    # Load dataset
    if local_rank == 0:
        print(f"Loading PixMo-Cap {args.split} split...")
    dataset = PixMoCap(split=args.split, mode="captions")
    if local_rank == 0:
        print()
    
    # Load ALL contextual embeddings from all layers (on all ranks)
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Loading contextual embeddings from ALL layers")
        print(f"{'='*60}\n")
    
    contextual_embeddings, contextual_metadata, layer_boundaries, token_to_indices = load_all_contextual_embeddings(
        args.contextual_dir,
        available_contextual_layers,
        max_per_token=args.max_contextual_per_token
    )
    
    # Build FAISS index if requested (BEFORE moving to GPU to avoid OOM)
    faiss_index = None
    # Default to CPU to avoid GPU OOM (can override with --faiss-gpu if you have enough VRAM)
    use_faiss_cpu = args.use_faiss and (not args.faiss_gpu)  # CPU by default, GPU only if explicitly requested
    
    if args.use_faiss:
        if not FAISS_AVAILABLE:
            raise ImportError(
                f"\n{'='*80}\n"
                f"ERROR: --use-faiss specified but FAISS is not available!\n"
                f"{'='*80}\n\n"
                f"FAISS was explicitly requested but cannot be imported.\n"
                f"Install FAISS with one of:\n"
                f"  pip install faiss-cpu\n"
                f"  pip install faiss-gpu\n\n"
                f"The script will NOT silently fall back to PyTorch.\n"
                f"If you want PyTorch brute-force search, do not use --use-faiss.\n"
                f"{'='*80}\n"
            )
        else:
            if local_rank == 0:
                print(f"\nBuilding FAISS index for {len(contextual_metadata)} embeddings...")
                if use_faiss_cpu:
                    print("  Using CPU-based FAISS to avoid GPU OOM")
                if args.faiss_quantized:
                    print("  Using Product Quantization for memory compression")
            
            # Normalize embeddings for FAISS (required for cosine similarity)
            # Keep on CPU if using CPU-based FAISS
            contextual_embeddings_norm = torch.nn.functional.normalize(contextual_embeddings, dim=-1)
            
            faiss_index = build_faiss_index(
                contextual_embeddings_norm,
                use_gpu=args.faiss_gpu,  # Only use GPU if explicitly requested (defaults to CPU to avoid OOM)
                approximate=args.faiss_approximate,
                quantized=args.faiss_quantized,
                nlist=args.faiss_nlist,
                nprobe=args.faiss_nprobe,
                m=args.faiss_m,
                nbits=args.faiss_nbits,
                local_rank=local_rank
            )
            
            if local_rank == 0:
                print("✓ FAISS index built and ready for fast search\n")
            
            # If using CPU FAISS, we can free GPU memory by not moving embeddings to GPU
            if use_faiss_cpu:
                # Keep embeddings on CPU for FAISS
                # Store normalized version for FAISS searches
                contextual_embeddings_norm_cpu = contextual_embeddings_norm.cpu()
                del contextual_embeddings_norm  # Free GPU memory
                # Keep original embeddings on CPU too (for metadata lookup if needed)
                contextual_embeddings_cpu = contextual_embeddings.cpu()
            else:
                # Move to GPU for GPU-based FAISS
                contextual_embeddings_norm = contextual_embeddings_norm.to(device)
                contextual_embeddings = contextual_embeddings.to(device)
                contextual_embeddings_norm_cpu = None
                contextual_embeddings_cpu = None
    else:
        # No FAISS - decide based on distributed search setting
        contextual_embeddings_norm_cpu = None
        contextual_embeddings_cpu = None
        # Don't move to GPU here - will be handled below based on search method
    
    # Setup distributed GPU search if requested (EXACT search, sharded across GPUs)
    embeddings_shard = None
    embeddings_shard_norm = None
    global_offset = 0
    use_distributed_gpu_search = args.distributed_gpu_search
    
    if use_distributed_gpu_search:
        if local_rank == 0:
            print(f"\n🚀 Setting up DISTRIBUTED GPU SEARCH (exact, sharded across {world_size} GPUs)")
        
        # Keep full embeddings on CPU for metadata lookups (needed for "lowest same token" analysis)
        # Only the shard goes to GPU for fast search
        contextual_embeddings_cpu = contextual_embeddings.cpu()
        
        # Shard embeddings across GPUs (from CPU tensor)
        embeddings_shard, global_offset = shard_embeddings_for_distributed_search(
            contextual_embeddings_cpu, contextual_metadata, world_size, local_rank
        )
        
        # Free the GPU copy if it existed
        del contextual_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        
        # Move ONLY the shard to this GPU and normalize
        embeddings_shard = embeddings_shard.to(device)
        embeddings_shard_norm = torch.nn.functional.normalize(embeddings_shard, dim=-1)
        
        # Free the unnormalized shard to save GPU memory
        del embeddings_shard
        embeddings_shard = None  # Reset to None so the variable still exists
        torch.cuda.empty_cache()
        
        if local_rank == 0:
            print(f"✓ Distributed GPU search ready!")
            print(f"  Each GPU holds {embeddings_shard_norm.shape[0]:,} embeddings ({embeddings_shard_norm.numel() * 4 / 1e9:.2f} GB)")
            print(f"  CPU holds full embeddings for metadata lookups ({contextual_embeddings_cpu.numel() * 4 / 1e9:.2f} GB)")
            print()
        
        # Set contextual_embeddings to CPU copy for process_images lookups
        contextual_embeddings = contextual_embeddings_cpu
    elif not args.use_faiss:
        # PyTorch brute-force on single GPU - move full embeddings to GPU
        contextual_embeddings = contextual_embeddings.to(device)
    
    # Wait for all processes to be ready
    dist.barrier()
    
    # Process each visual layer sequentially
    for layer_idx, visual_layer in enumerate(visual_layers):
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"Processing visual layer {visual_layer} ({layer_idx + 1}/{len(visual_layers)})")
            print(f"  Comparing to contextual embeddings from layers: {available_contextual_layers}")
            if use_distributed_gpu_search:
                print(f"  Search method: DISTRIBUTED GPU (exact, sharded across {world_size} GPUs)")
            elif args.use_faiss and FAISS_AVAILABLE:
                search_type = []
                if args.faiss_approximate:
                    search_type.append("approximate")
                if args.faiss_quantized:
                    search_type.append("quantized")
                if use_faiss_cpu:
                    search_type.append("CPU")
                else:
                    search_type.append("GPU")
                search_method = f"FAISS ({', '.join(search_type)})"
                print(f"  Search method: {search_method}")
            else:
                print(f"  Search method: PyTorch brute-force")
            print(f"{'='*60}\n")
        
        # Process images (distributed across GPUs)
        prompt = "Describe this image in detail."
        if local_rank == 0:
            print(f"Processing {args.num_images} images across {world_size} GPUs...")
        results = process_images(
            model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
            contextual_embeddings if not use_faiss_cpu else contextual_embeddings_cpu, 
            contextual_metadata, token_to_indices,
            top_k=args.top_k, llm_layer=visual_layer,
            use_faiss=args.use_faiss and FAISS_AVAILABLE, 
            faiss_index=faiss_index,
            faiss_cpu=use_faiss_cpu,
            contextual_embeddings_norm_cpu=contextual_embeddings_norm_cpu if 'contextual_embeddings_norm_cpu' in locals() else None,
            contextual_embeddings_cpu=contextual_embeddings_cpu if 'contextual_embeddings_cpu' in locals() else None,
            use_distributed_gpu_search=use_distributed_gpu_search,
            embeddings_shard=embeddings_shard,
            embeddings_shard_norm=embeddings_shard_norm,
            global_offset=global_offset
        )
        
        # Wait for all processes to finish this layer
        dist.barrier()
        
        # Save results (only on main process)
        if local_rank == 0:
            # CRITICAL VALIDATION: Ensure we have results from ALL images before saving
            # This prevents silently saving incomplete results
            actual_count = len(results) if results else 0
            expected_count = args.num_images
            
            if actual_count != expected_count:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"CRITICAL ERROR: Result count mismatch!\n"
                    f"{'='*80}\n"
                    f"Expected {expected_count} images, but only got {actual_count}.\n"
                    f"This indicates a bug in result gathering from distributed processes.\n"
                    f"Will NOT save incomplete results to JSON.\n"
                    f"{'='*80}\n"
                )
            
            # Verify all image indices are present
            actual_indices = sorted([r.get('image_idx') for r in results])
            expected_indices = list(range(expected_count))
            if actual_indices != expected_indices:
                missing = set(expected_indices) - set(actual_indices)
                extra = set(actual_indices) - set(expected_indices)
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"CRITICAL ERROR: Image index mismatch!\n"
                    f"{'='*80}\n"
                    f"Expected indices: 0 to {expected_count - 1}\n"
                    f"Got indices: {actual_indices[:10]}... (showing first 10)\n"
                    f"Missing indices: {sorted(missing)[:20]}... (showing first 20)\n"
                    f"Extra indices: {sorted(extra)[:20]}... (showing first 20)\n"
                    f"This indicates a bug in result gathering from distributed processes.\n"
                    f"Will NOT save incomplete results to JSON.\n"
                    f"{'='*80}\n"
                )
            
            print(f"✓ Validation passed: {actual_count} images with correct indices")
            
            # Sort results by image_idx for consistent ordering in JSON
            results = sorted(results, key=lambda r: r.get('image_idx', 0))
            
            # Setup output directory
            ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
            output_dir = Path(args.output_dir) / ckpt_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            output_file = output_dir / f"contextual_neighbors_visual{visual_layer}_allLayers_multi-gpu.json"
            print(f"\n✓ Saving results to {output_file}...")
            
            # Sort results by image_idx for consistent ordering
            results = sorted(results, key=lambda r: r.get('image_idx', 0))
            
            output_data = {
                'checkpoint': args.ckpt_path,
                'contextual_dir': args.contextual_dir,
                'visual_layer': visual_layer,
                'contextual_layers_used': available_contextual_layers,
                'layer_boundaries': [(layer_idx, start, end) for layer_idx, start, end in layer_boundaries],
                'split': args.split,
                'num_images': args.num_images,
                'num_processes': world_size,
                'top_k': args.top_k,
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Visual layer {visual_layer} complete! Results saved.")
        
        # Wait for main process to finish saving before moving to next layer
        dist.barrier()
        
        if local_rank == 0:
            print(f"✓ Completed visual layer {visual_layer}\n")
    
    # All visual layers processed
    if local_rank == 0:
        print("="*60)
        print(f"✓ All {len(visual_layers)} visual layer(s) processed successfully!")
        print("="*60)
    
    # Clear contextual embeddings to free memory
    del contextual_embeddings, contextual_metadata, token_to_indices
    torch.cuda.empty_cache()
    
    # Wait for all processes to finish
    dist.barrier()


if __name__ == "__main__":
    main()
