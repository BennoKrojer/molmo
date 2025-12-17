"""
Find nearest contextual text embeddings for visual tokens in Qwen2-VL.

This script takes visual tokens from Qwen2-VL (off-the-shelf HuggingFace model) and finds their 
top-k nearest neighbors in the contextual text embedding space (saved by create_contextual_embeddings.py).

Similar to contextual_nearest_neighbors.py, but adapted for Qwen2-VL which is loaded directly
from HuggingFace and uses different preprocessing/architecture.

This is the multi-GPU version that uses distributed processing.

Usage:
    torchrun --nproc_per_node=2 scripts/analysis/qwen2_vl/contextual_nearest_neighbors.py \
        --model-name Qwen/Qwen2-VL-7B-Instruct \
        --contextual-dir <dir> \
        --contextual-layer 16 \
        --num-images 100
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
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

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
        contextual_dir: Base directory for contextual embeddings (e.g., molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B)
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


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def extract_visual_tokens_qwen2vl(model, processor, image, prompt, visual_layer=0, device="cuda:0"):
    """
    Extract visual tokens from Qwen2-VL model.
    
    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor for Qwen2-VL
        image: PIL Image or image path
        prompt: Text prompt
        visual_layer: 0 for vision encoder output, >0 for LLM layer output
        device: Device to run on
    
    Returns:
        visual_tokens: torch.Tensor [batch, num_visual_tokens, hidden_dim]
        metadata: Dict with shape info
    """
    # Prepare inputs
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
        # Resize very large images to prevent OOM in vision encoder
        # Qwen2-VL can handle various sizes, but very large images cause memory issues
        max_size = 1024  # Maximum dimension
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Prepare model inputs using processor
    # Qwen2-VL processor requires <|image_pad|> token in text to insert image tokens
    # The processor will automatically expand this to the correct number of tokens based on image_grid_thw
    image_token = "<|image_pad|>"
    prompt_with_image = f"{image_token}{prompt}"  # Image token at the beginning
    
    model_inputs = processor(images=[image], text=prompt_with_image, return_tensors="pt")
    model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in model_inputs.items()}
    
    # Clear any cached memory before forward pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if visual_layer == 0:
        # Extract from LLM embedding layer (layer 0)
        # This gives us vision tokens AFTER they've been processed by the LLM's embedding layer,
        # not raw vision encoder output. This matches the behavior of the Molmo script where
        # visual_layer=0 gets vision tokens that have been embedded into LLM space.
        # Qwen2-VL processes pixel_values internally and includes vision tokens in hidden_states
        # Vision tokens are at the beginning of the sequence, identified by image_grid_thw
        with torch.no_grad():  # Use no_grad instead of inference_mode for better compatibility
            try:
                # Clear cache before forward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Run full model forward to get hidden states from LLM
                # Qwen2-VL's forward accepts pixel_values and output_hidden_states=True
                outputs = model(**model_inputs, output_hidden_states=True)
                
                if len(outputs.hidden_states) == 0:
                    raise RuntimeError("No hidden states returned from model")
                
                # Get the LLM's embedding layer (layer 0) - this is vision tokens in LLM embedding space
                first_hidden = outputs.hidden_states[0]
                
                # Determine number of visual tokens from image_grid_thw
                # image_grid_thw shape: [num_images, 3] where 3 = [temporal, height, width]
                num_image_tokens = None
                if 'image_grid_thw' in model_inputs:
                    image_grid_thw = model_inputs['image_grid_thw']
                    if isinstance(image_grid_thw, torch.Tensor):
                        # Calculate total patches: sum of (t * h * w) for all images
                        # Each image contributes t * h * w visual tokens
                        num_image_tokens = image_grid_thw.sum().item()
                
                if num_image_tokens is None:
                    # Fallback: estimate from pixel_values shape
                    # pixel_values shape is [num_patches, patch_dim] for Qwen2-VL
                    if 'pixel_values' in model_inputs:
                        pixel_values = model_inputs['pixel_values']
                        if pixel_values is not None and len(pixel_values.shape) >= 2:
                            # First dimension is number of patches
                            num_image_tokens = pixel_values.shape[0]
                
                if num_image_tokens is None:
                    # Final fallback: conservative estimate
                    num_image_tokens = 256
                
                # Extract visual tokens from the beginning of the sequence
                if first_hidden.shape[1] >= num_image_tokens:
                    vision_features = first_hidden[:, :num_image_tokens, :]
                else:
                    # If sequence is shorter than expected, use all available tokens
                    print(f"Warning: Sequence length {first_hidden.shape[1]} < expected visual tokens {num_image_tokens}")
                    vision_features = first_hidden
                
                return vision_features, {
                    "feature_shape": list(vision_features.shape),
                    "num_visual_tokens": num_image_tokens
                }
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to extract vision features from Qwen2-VL model: {e}\n"
                    "Make sure pixel_values and image_grid_thw are properly formatted."
                )
    else:
        # Extract from LLM layer
        with torch.no_grad():  # Use no_grad for memory efficiency
            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = model(**model_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            if len(hidden_states) == 0:
                raise RuntimeError("No hidden states returned from model")
            
            # Get the specified layer
            layer_index = min(visual_layer, len(hidden_states) - 1)
            layer_hidden_states = hidden_states[layer_index]
            
            # Extract visual token positions
            # In Qwen2-VL, visual tokens are typically at the beginning of the sequence
            num_image_tokens = None
            
            # Try to get from image_grid_thw (most reliable)
            if 'image_grid_thw' in model_inputs:
                image_grid_thw = model_inputs['image_grid_thw']
                if isinstance(image_grid_thw, torch.Tensor):
                    num_image_tokens = image_grid_thw.sum().item()
            
            # Fallback: try config
            if num_image_tokens is None:
                if hasattr(model.config, 'vision_config') and hasattr(model.config.vision_config, 'num_image_tokens'):
                    num_image_tokens = model.config.vision_config.num_image_tokens
                elif hasattr(model.config, 'num_image_tokens'):
                    num_image_tokens = model.config.num_image_tokens
            
            # Final fallback: estimate
            if num_image_tokens is None:
                # Qwen2-VL-7B typically uses ~256-1024 visual tokens depending on image size
                num_image_tokens = 256  # Conservative default
            
            # Extract visual tokens (first num_image_tokens positions)
            # Make sure we don't exceed sequence length
            actual_num_tokens = min(num_image_tokens, layer_hidden_states.shape[1])
            visual_features = layer_hidden_states[:, :actual_num_tokens, :]
            
            return visual_features, {
                "feature_shape": list(visual_features.shape),
                "llm_layer_used": visual_layer,
                "num_visual_tokens": actual_num_tokens
            }


def process_images(model, processor, dataset, num_images, prompt,
                   contextual_embeddings, contextual_metadata, token_to_indices, 
                   top_k=5, visual_layer=0, device="cuda:0"):
    """
    Process images and find nearest contextual neighbors for each visual token.
    
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
    
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
        example_data = dataset.get(i, np.random)
        
        # Extract ground truth caption
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            caption_text = message.get("text", "")
        
        # Get image path
        image_path = example_data["image"]
        
        # Initialize results
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text,
            "patches": []
        }
        
        # Extract visual tokens
        # Use no_grad for maximum memory efficiency (inference_mode is even better but less compatible)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Clear cache before each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                visual_features, feature_metadata = extract_visual_tokens_qwen2vl(
                    model, processor, image_path, prompt, visual_layer=visual_layer, device=device
                )
                
                image_results.update(feature_metadata)
                
                # Process visual tokens
                # visual_features: [batch, num_visual_tokens, hidden_dim]
                batch_size, num_visual_tokens, hidden_dim = visual_features.shape
                
                # Normalize for cosine similarity
                visual_features_norm = torch.nn.functional.normalize(visual_features, dim=-1)
                contextual_norm = torch.nn.functional.normalize(contextual_embeddings, dim=-1)
                
                # Compute similarity for ALL tokens at once
                # similarity: [batch, num_visual_tokens, num_contextual_embeddings]
                similarity = torch.matmul(visual_features_norm, contextual_norm.T)
                
                # Get top-k for all tokens at once
                top_values, top_indices = torch.topk(similarity, k=min(top_k, similarity.shape[-1]), dim=-1)
                
                # Convert to CPU for processing
                top_values = top_values.cpu().numpy()
                top_indices = top_indices.cpu().numpy()
                
                # Clear similarity tensors to free GPU memory
                del similarity, visual_features_norm, contextual_norm
                
                # Process each visual token
                for token_idx in range(num_visual_tokens):
                    # Get top-k results for this token
                    token_top_values = top_values[0, token_idx]
                    token_top_indices = top_indices[0, token_idx]
                    
                    # Get the token embedding for finding lowest similarities
                    token_embedding = visual_features[0, token_idx:token_idx+1, :]  # [1, hidden_dim]
                    token_embedding_norm = torch.nn.functional.normalize(token_embedding, dim=-1)
                    
                    # Build nearest neighbors list
                    nearest_contextual = []
                    top_nn_embeddings = []  # Store embeddings for inter-NN comparisons
                    
                    for val, idx in zip(token_top_values, token_top_indices):
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
                                same_token_sims = torch.matmul(token_embedding_norm, same_token_embeddings_norm.T).squeeze(0)
                                
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
                    
                    # Add row/col info (if we can determine grid size)
                    # For now, just use token_idx
                    grid_size = int(math.sqrt(num_visual_tokens)) if num_visual_tokens > 0 else 1
                    row = token_idx // grid_size if grid_size > 0 else 0
                    col = token_idx % grid_size if grid_size > 0 else 0
                    
                    patch_results = {
                        "patch_idx": token_idx,
                        "patch_row": row,
                        "patch_col": col,
                        "nearest_contextual_neighbors": nearest_contextual
                    }
                    
                    image_results["patches"].append(patch_results)
                
                # Clear intermediate tensors
                del visual_features
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
    
    parser = argparse.ArgumentParser(description="Find nearest contextual text embeddings for visual tokens in Qwen2-VL (Multi-GPU)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="HuggingFace model name for Qwen2-VL")
    parser.add_argument("--contextual-dir", type=str, required=True,
                       help="Directory with contextual embeddings (e.g., molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B)")
    parser.add_argument("--contextual-layer", type=str, required=True,
                       help="Layer index(es) of contextual embeddings to use. Single layer (e.g., '8') or comma-separated list (e.g., '8,16,24')")
    parser.add_argument("--visual-layer", type=int, default=0,
                       help="Visual layer to extract (0 = vision encoder, >0 = LLM layer)")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of nearest neighbors to find (default: 5)")
    parser.add_argument("--max-contextual-per-token", type=int, default=None,
                       help="Maximum contextual embeddings to load per token (for memory management)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/qwen2_vl/contextual_nearest_neighbors",
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Parse contextual layers to process
    contextual_layers = [int(layer.strip()) for layer in args.contextual_layer.split(",")]
    
    if local_rank == 0:
        print(f"{'='*80}")
        print(f"Qwen2-VL Visual-to-Contextual Nearest Neighbors Analysis (Multi-GPU)")
        print(f"{'='*80}\n")
        print(f"Model: {args.model_name}")
        print(f"Contextual embeddings: {args.contextual_dir}")
        print(f"Contextual layers to process: {contextual_layers}")
        print(f"Visual layer: {args.visual_layer}")
        print(f"Dataset split: {args.split}")
        print(f"Number of images: {args.num_images}")
        print(f"Top-k neighbors: {args.top_k}")
        print(f"Running on {world_size} processes")
        print()
    
    # Load model and processor
    if local_rank == 0:
        print(f"Loading Qwen2-VL model from {args.model_name}...")
    
    # Load model on each rank (may need to use device_map for large models)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{local_rank}" if world_size > 1 else "auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    if local_rank == 0:
        print(f"Model loaded successfully\n")
    
    # Load dataset
    if local_rank == 0:
        print(f"Loading PixMo-Cap {args.split} split...")
    dataset = PixMoCap(split=args.split, mode="captions")
    if local_rank == 0:
        print()
    
    # Process each contextual layer sequentially
    for layer_idx, contextual_layer in enumerate(contextual_layers):
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"Processing contextual layer {contextual_layer} ({layer_idx + 1}/{len(contextual_layers)})")
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
            model, processor, dataset, args.num_images, prompt,
            contextual_embeddings, contextual_metadata, token_to_indices,
            top_k=args.top_k, visual_layer=args.visual_layer, device=device
        )
        
        # Wait for all processes to finish this layer
        dist.barrier()
        
        # Save results (only on main process)
        if local_rank == 0:
            # Setup output directory
            model_name_safe = args.model_name.replace("/", "_")
            output_dir = Path(args.output_dir) / model_name_safe
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            output_file = output_dir / f"contextual_neighbors_visual{args.visual_layer}_contextual{contextual_layer}_multi-gpu.json"
            print(f"\n✓ Saving results to {output_file}...")
            
            output_data = {
                'model_name': args.model_name,
                'contextual_dir': args.contextual_dir,
                'contextual_layer': contextual_layer,
                'visual_layer': args.visual_layer,
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
        print(f"✓ All {len(contextual_layers)} contextual layer(s) processed successfully!")
        print("="*60)
    
    # Wait for all processes to finish
    dist.barrier()


if __name__ == "__main__":
    main()

