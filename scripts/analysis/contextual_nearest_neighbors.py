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

from olmo.config import ModelConfig
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
    """
    layer_dir = Path(contextual_dir) / f"layer_{layer_idx}"
    token_embeddings_file = layer_dir / "token_embeddings.json"
    
    if not token_embeddings_file.exists():
        raise FileNotFoundError(f"Token embeddings file not found: {token_embeddings_file}")
    
    # Define cache file path
    cache_suffix = f"_max{max_per_token}" if max_per_token is not None else ""
    cache_file = layer_dir / f"embeddings_cache{cache_suffix}.pt"
    
    # Try to load from cache
    if cache_file.exists():
        print(f"Loading contextual embeddings from cache: {cache_file}")
        try:
            cache_data = torch.load(cache_file, map_location='cpu')
            embeddings_matrix = cache_data['embeddings']
            metadata_list = cache_data['metadata']
            print(f"Loaded {len(metadata_list)} contextual embeddings from cache with shape {embeddings_matrix.shape}")
            return embeddings_matrix, metadata_list
        except Exception as e:
            print(f"Warning: Failed to load cache, will reload from individual files: {e}")
    
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
    
    # Save to cache for future runs
    print(f"Saving embeddings to cache: {cache_file}")
    try:
        cache_data = {
            'embeddings': embeddings_matrix,
            'metadata': metadata_list
        }
        torch.save(cache_data, cache_file)
        print(f"Cache saved successfully!")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return embeddings_matrix, metadata_list


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
                   contextual_embeddings, contextual_metadata, top_k=5, llm_layer=0):
    """
    Process images and find nearest contextual neighbors for each visual token.
    
    Similar to process_split from general_and_nearest_neighbors_pixmo_cap_multi-gpu.py
    but finds nearest contextual embeddings instead of vocabulary tokens.
    
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
                        
                        # Build nearest neighbors list
                        nearest_contextual = []
                        for val, idx in zip(patch_top_values, patch_top_indices):
                            idx = int(idx)
                            nearest_contextual.append({
                                'token_str': contextual_metadata[idx]['token_str'],
                                'token_id': contextual_metadata[idx]['token_id'],
                                'caption': contextual_metadata[idx]['caption'],
                                'position': contextual_metadata[idx]['position'],
                                'similarity': float(val)
                            })
                        
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
    parser.add_argument("--contextual-layer", type=int, required=True,
                       help="Layer index of contextual embeddings to use")
    parser.add_argument("--visual-layer", type=int, default=0,
                       help="Visual layer to extract (0 = vision backbone, >0 = LLM layer)")
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
    
    if local_rank == 0:
        print(f"{'='*80}")
        print(f"Visual-to-Contextual Nearest Neighbors Analysis (Multi-GPU)")
        print(f"{'='*80}\n")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Contextual embeddings: {args.contextual_dir}")
        print(f"Contextual layer: {args.contextual_layer}")
        print(f"Visual layer: {args.visual_layer}")
        print(f"Dataset split: {args.split}")
        print(f"Number of images: {args.num_images}")
        print(f"Top-k neighbors: {args.top_k}")
        print(f"Running on {world_size} processes")
        print()
    
    # Load contextual embeddings (on all ranks)
    if local_rank == 0:
        print("Loading contextual text embeddings...")
    contextual_embeddings, contextual_metadata = load_contextual_embeddings(
        args.contextual_dir,
        args.contextual_layer,
        max_per_token=args.max_contextual_per_token
    )
    contextual_embeddings = contextual_embeddings.to(device)
    if local_rank == 0:
        print(f"Loaded {len(contextual_metadata)} contextual embeddings\n")
    
    # Load Molmo model on CPU first
    if local_rank == 0:
        print(f"Loading Molmo model from {args.ckpt_path}...")
    model = Molmo.from_checkpoint(args.ckpt_path, device="cpu")
    model.eval()
    
    # Wrap model with FSDP for sharding
    if local_rank == 0:
        print("Wrapping model with FSDP for sharding...")
    
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
    
    # Wait for all processes to be ready
    dist.barrier()
    
    # Process images (distributed across GPUs)
    prompt = "Describe this image in detail."
    if local_rank == 0:
        print(f"Processing {args.num_images} images across {world_size} GPUs...")
    results = process_images(
        model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
        contextual_embeddings, contextual_metadata,
        top_k=args.top_k, llm_layer=args.visual_layer
    )
    
    # Wait for all processes to finish
    dist.barrier()
    
    # Save results (only on main process)
    if local_rank == 0:
        # Setup output directory
        ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
        output_dir = Path(args.output_dir) / ckpt_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_file = output_dir / f"contextual_neighbors_visual{args.visual_layer}_contextual{args.contextual_layer}_multi-gpu.json"
        print(f"\nSaving results to {output_file}...")
        
        output_data = {
            'checkpoint': args.ckpt_path,
            'contextual_dir': args.contextual_dir,
            'contextual_layer': args.contextual_layer,
            'visual_layer': args.visual_layer,
            'split': args.split,
            'num_images': args.num_images,
            'num_processes': world_size,
            'top_k': args.top_k,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Print some example results
        print(f"\n{'='*80}")
        print(f"EXAMPLE RESULTS (First image, first 3 patches)")
        print(f"{'='*80}\n")
        
        if results:
            first_image = results[0]
            print(f"Image {first_image['image_idx']}")
            print(f"Ground truth: {first_image['ground_truth_caption'][:100]}...\n")
            
            if first_image['chunks']:
                first_chunk = first_image['chunks'][0]
                for patch in first_chunk['patches'][:3]:
                    print(f"Patch {patch['patch_idx']} (row={patch['patch_row']}, col={patch['patch_col']}):")
                    print("  Top-3 nearest contextual neighbors:")
                    for i, neighbor in enumerate(patch['nearest_contextual_neighbors'][:3], 1):
                        print(f"    {i}. '{neighbor['token_str']}' (sim={neighbor['similarity']:.3f})")
                        print(f"       Caption: {neighbor['caption'][:80]}...")
                        print(f"       Position: {neighbor['position']}")
                    print()
        
        print(f"\n✓ Analysis complete!")
        print(f"Results saved to: {output_dir}")
    
    # Wait for main process to finish saving
    dist.barrier()


if __name__ == "__main__":
    main()
