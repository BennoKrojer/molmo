"""
Find nearest contextual text embeddings for visual tokens.

This script takes visual tokens from a Molmo checkpoint and finds their top-5 nearest neighbors
in the contextual text embedding space (saved by create_contextual_embeddings.py).

Similar to general_and_nearest_neighbors_pixmo_cap_multi-gpu.py, but instead of comparing
visual tokens to static vocabulary embeddings, we compare them to contextualized word
embeddings from real captions. This tells us which real words in real contexts are most
similar to each visual token.

This is the multi-GPU version that uses FSDP for model sharding.

Usage:
    torchrun --nproc_per_node=2 scripts/analysis/contextual_nearest_neighbors.py --ckpt-path <path> --contextual-dir <dir> --contextual-layer 16 --num-images 100
"""

import argparse
import gc
import json
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


def process_images(model, preprocessor, dataset, num_images, prompt, use_n_token_only, 
                   contextual_embeddings, contextual_metadata, token_to_indices, top_k=5, llm_layer=0):
    """
    Process images and find nearest contextual neighbors for each visual token.
    
    Similar to process_split from general_and_nearest_neighbors_pixmo_cap_multi-gpu.py
    but finds nearest contextual embeddings instead of vocabulary tokens.
    
    For each of the top-k nearest neighbors, also finds the same token with lowest similarity.
    
    Uses distributed processing - each process handles a subset of images.
    """
    # Distribute images across processes
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    images_per_process = num_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    
    # Handle remainder for last process
    if local_rank == world_size - 1:
        end_idx = num_images
    
    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1}")
    
    results = []
    device = torch.device(f"cuda:{local_rank}")
    
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
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
                
                # Batch process all patches at once (like general_and_nearest_neighbors_pixmo_cap_multi-gpu.py)
                batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                
                # Reshape to combine batch and chunks dimensions
                image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)
                
                # Normalize for cosine similarity
                image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
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
                del similarity, image_features_norm, contextual_norm
                
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
                            top_nn_embeddings.append(contextual_embeddings[idx])
                            
                            # Find the same token with lowest similarity
                            lowest_info = None
                            if token_str in token_to_indices:
                                same_token_indices = token_to_indices[token_str]
                                if len(same_token_indices) > 1:  # Only if there are multiple instances
                                    # Get embeddings for all instances of this token
                                    same_token_embeddings = contextual_embeddings[same_token_indices]
                                    same_token_embeddings_norm = torch.nn.functional.normalize(same_token_embeddings, dim=-1)
                                    
                                    # Compute similarities
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
                            
                            nearest_contextual.append({
                                'token_str': token_str,
                                'token_id': contextual_metadata[idx]['token_id'],
                                'caption': contextual_metadata[idx]['caption'],
                                'position': contextual_metadata[idx]['position'],
                                'similarity': float(val),
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
        
        results.append(image_results)
        clear_gpu_memory()
    
    # Gather results from all processes
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    
    # Combine results on main process
    if local_rank == 0:
        combined_results = []
        for process_results in all_results:
            combined_results.extend(process_results)
        return combined_results
    else:
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
    parser.add_argument("--contextual-layer", type=str, required=True,
                       help="Layer index(es) of contextual embeddings to use. Single layer (e.g., '8') or comma-separated list (e.g., '8,16,24')")
    parser.add_argument("--visual-layer", type=str, default="0",
                       help="Visual layer(s) to extract. Single layer (e.g., '8') or comma-separated list (e.g., '8,16,24'). Must match contextual-layer count. 0 = vision backbone, >0 = LLM layer")
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
    args = parser.parse_args()
    
    # Parse contextual layers and visual layers to process
    contextual_layers = [int(layer.strip()) for layer in args.contextual_layer.split(",")]
    visual_layers = [int(layer.strip()) for layer in args.visual_layer.split(",")]
    
    # Validate that visual_layers and contextual_layers match (for matching pairs)
    if len(visual_layers) != len(contextual_layers):
        raise ValueError(
            f"visual_layer and contextual_layer must have the same number of layers. "
            f"Got {len(visual_layers)} visual layers and {len(contextual_layers)} contextual layers."
        )
    
    # Create matching pairs
    layer_pairs = list(zip(visual_layers, contextual_layers))
    
    if local_rank == 0:
        print(f"{'='*80}")
        print(f"Visual-to-Contextual Nearest Neighbors Analysis (Multi-GPU)")
        print(f"{'='*80}\n")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Contextual embeddings: {args.contextual_dir}")
        print(f"Layer pairs to process (visual_layer, contextual_layer): {layer_pairs}")
        print(f"Dataset split: {args.split}")
        print(f"Number of images: {args.num_images}")
        print(f"Top-k neighbors: {args.top_k}")
        print(f"Running on {world_size} processes")
        print(f"Processing layers sequentially to save memory")
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
    
    # Process each matching layer pair sequentially
    for layer_idx, (visual_layer, contextual_layer) in enumerate(layer_pairs):
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"Processing layer pair: visual{visual_layer} vs contextual{contextual_layer} ({layer_idx + 1}/{len(layer_pairs)})")
            print(f"{'='*60}\n")
        
        # Load contextual embeddings for this layer (on all ranks)
        if local_rank == 0:
            print(f"Loading contextual text embeddings for layer {contextual_layer}...")
        contextual_embeddings, contextual_metadata, token_to_indices = load_contextual_embeddings(
            args.contextual_dir,
            contextual_layer,
            max_per_token=args.max_contextual_per_token
        )
        contextual_embeddings = contextual_embeddings.to(device)
        if local_rank == 0:
            print(f"Loaded {len(contextual_metadata)} contextual embeddings")
            print(f"Found {len(token_to_indices)} unique tokens\n")
        
        # Wait for all processes to be ready
        dist.barrier()
        
        # Process images (distributed across GPUs)
        prompt = "Describe this image in detail."
        if local_rank == 0:
            print(f"Processing {args.num_images} images across {world_size} GPUs...")
        results = process_images(
            model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
            contextual_embeddings, contextual_metadata, token_to_indices,
            top_k=args.top_k, llm_layer=visual_layer
        )
        
        # Wait for all processes to finish this layer
        dist.barrier()
        
        # Save results (only on main process)
        if local_rank == 0:
            # Setup output directory
            ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
            output_dir = Path(args.output_dir) / ckpt_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            output_file = output_dir / f"contextual_neighbors_visual{visual_layer}_contextual{contextual_layer}_multi-gpu.json"
            print(f"\n✓ Saving results to {output_file}...")
            
            output_data = {
                'checkpoint': args.ckpt_path,
                'contextual_dir': args.contextual_dir,
                'contextual_layer': contextual_layer,
                'visual_layer': visual_layer,
                'split': args.split,
                'num_images': args.num_images,
                'num_processes': world_size,
                'top_k': args.top_k,
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Layer {contextual_layer} complete! Results saved.")
        
        # Wait for main process to finish saving before moving to next layer
        dist.barrier()
        
        # Clear contextual embeddings to free memory
        del contextual_embeddings, contextual_metadata, token_to_indices
        torch.cuda.empty_cache()
        
        if local_rank == 0:
            print(f"✓ Cleared contextual embeddings for layer {contextual_layer}\n")
    
    # All layers processed
    if local_rank == 0:
        print("="*60)
        print(f"✓ All {len(layer_pairs)} layer pair(s) processed successfully!")
        print("="*60)
    
    # Wait for all processes to finish
    dist.barrier()


if __name__ == "__main__":
    main()
