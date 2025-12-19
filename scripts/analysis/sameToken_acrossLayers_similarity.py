"""
Compute cosine similarity of vision tokens across layers to layer 0.

This script traces vision tokens through the LLM layers and computes the cosine similarity
between each vision token at layer N and the same token at layer 0 (input layer).

For each layer, we compute:
- Per-patch similarity (each vision token compared to its layer-0 version)
- Average similarity across all patches
- Average similarity across all images

This helps understand how vision tokens evolve through the LLM layers and whether they
stay close to their original representation or drift away.

Single-GPU version (no FSDP overhead).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/sameToken_acrossLayers_similarity.py --ckpt-path <path> --num-images 100
"""

import argparse
import gc
import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def gather_visual_features(hs_tensor, image_input_idx_tensor):
    """
    Extract visual tokens from hidden states using image_input_idx.
    
    Args:
        hs_tensor: Hidden states tensor [B, seq_len, d_model]
        image_input_idx_tensor: Image input indices [B, num_chunks, patches_per_chunk]
    
    Returns:
        feats: Visual features [B, num_chunks, patches_per_chunk, d_model]
    """
    B, num_chunks, patches_per_chunk = image_input_idx_tensor.shape
    d_model = hs_tensor.shape[-1]
    feats = torch.zeros((B, num_chunks, patches_per_chunk, d_model), 
                        device=hs_tensor.device, dtype=hs_tensor.dtype)
    
    flat_positions = image_input_idx_tensor.view(B, -1)
    valid_mask = flat_positions >= 0
    
    for b in range(B):
        valid_pos = flat_positions[b][valid_mask[b]]
        if valid_pos.numel() == 0:
            continue
        gathered = hs_tensor[b, valid_pos.long(), :]
        feats.view(B, -1, d_model)[b, valid_mask[b], :] = gathered
    
    return feats


def process_images(model, preprocessor, dataset, num_images, prompt, use_n_token_only, 
                   max_layers=None, device=None):
    """
    Process images and compute similarity of vision tokens across layers to layer 0.
    
    For each image:
    1. Extract vision tokens at layer 0 (input)
    2. Extract vision tokens at layers 1, 2, 3, ... (through LLM)
    3. Compute cosine similarity between layer N tokens and layer 0 tokens (same position)
    4. Average across patches and images
    """
    if device is None:
        device = torch.device("cuda")
    
    results = []
    
    # Accumulators for averaging across images
    layer_similarities_sum = {}  # layer_idx -> {"same": sum, "baseline": sum}
    layer_similarities_count = {}  # layer_idx -> count of patches
    
    for i in tqdm(range(num_images), desc="Processing images"):
        example_data = dataset.get(i, np.random)
        
        # Create example
        example = {
            "image": example_data["image"],
            "messages": [prompt]
        }
        
        # Preprocess
        batch = preprocessor(example, rng=np.random)
        
        # Initialize results for this image
        image_results = {
            "image_idx": i,
            "layer_similarities": {}
        }
        
        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                
                # Extract layer 0 tokens (vision backbone output)
                image_features_layer0, _ = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    image_features_layer0 = image_features_layer0[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    image_features_layer0 = image_features_layer0[:, :, use_n_token_only, :]
                
                # Normalize layer 0 tokens for cosine similarity
                image_features_layer0_norm = torch.nn.functional.normalize(image_features_layer0, dim=-1)
                
                # Forward through LLM to get hidden states at all layers
                output = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=image_input_idx_tensor,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                hidden_states = output.hidden_states
                
                # Determine which layers to process
                num_layers = len(hidden_states)
                if max_layers is not None:
                    num_layers = min(num_layers, max_layers + 1)  # +1 because we include layer 0
                
                # Process each layer
                for layer_idx in range(1, num_layers):  # Start from 1 (layer 0 already extracted)
                    # Extract visual tokens from this LLM layer
                    layer_hidden_states = hidden_states[layer_idx]
                    image_features_layerN = gather_visual_features(layer_hidden_states, image_input_idx_tensor)
                    
                    # Apply same use_n_token_only filtering as layer 0
                    if type(use_n_token_only) == int and use_n_token_only != -1:
                        image_features_layerN = image_features_layerN[:, :, :use_n_token_only, :]
                    elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                        image_features_layerN = image_features_layerN[:, :, use_n_token_only, :]
                    
                    # Normalize for cosine similarity
                    image_features_layerN_norm = torch.nn.functional.normalize(image_features_layerN, dim=-1)
                    
                    # SAME TOKEN: Compute cosine similarity between same-position tokens
                    # Shape: [B, num_chunks, patches_per_chunk]
                    similarity_same = (image_features_layer0_norm * image_features_layerN_norm).sum(dim=-1)
                    similarity_same_per_patch = similarity_same.view(-1)  # Flatten to [num_patches]
                    
                    # BASELINE: Compute cosine similarity between different tokens
                    # Shuffle layer N tokens so we compare different positions
                    B, num_chunks, patches_per_chunk, hidden_dim = image_features_layerN_norm.shape
                    # Reshape to [B*num_chunks, patches_per_chunk, hidden_dim] for easier shuffling
                    layer0_flat = image_features_layer0_norm.view(B * num_chunks, patches_per_chunk, hidden_dim)
                    layerN_flat = image_features_layerN_norm.view(B * num_chunks, patches_per_chunk, hidden_dim)
                    
                    # Create shuffled indices for each chunk (different shuffle per chunk)
                    baseline_similarities = []
                    for chunk_idx in range(B * num_chunks):
                        # Shuffle the layer N tokens for this chunk
                        shuffled_indices = torch.randperm(patches_per_chunk, device=layerN_flat.device)
                        layerN_shuffled = layerN_flat[chunk_idx, shuffled_indices, :]  # [patches_per_chunk, hidden_dim]
                        
                        # Compare layer 0 tokens to shuffled layer N tokens (different positions)
                        baseline_sim = (layer0_flat[chunk_idx] * layerN_shuffled).sum(dim=-1)  # [patches_per_chunk]
                        baseline_similarities.append(baseline_sim)
                    
                    similarity_baseline_per_patch = torch.cat(baseline_similarities, dim=0)  # [num_patches]
                    
                    # Compute statistics for same-token similarity
                    mean_sim_same = similarity_same_per_patch.mean().item()
                    std_sim_same = similarity_same_per_patch.std().item()
                    min_sim_same = similarity_same_per_patch.min().item()
                    max_sim_same = similarity_same_per_patch.max().item()
                    
                    # Compute statistics for baseline (different-token) similarity
                    mean_sim_baseline = similarity_baseline_per_patch.mean().item()
                    std_sim_baseline = similarity_baseline_per_patch.std().item()
                    min_sim_baseline = similarity_baseline_per_patch.min().item()
                    max_sim_baseline = similarity_baseline_per_patch.max().item()
                    
                    # Store per-image results
                    image_results["layer_similarities"][layer_idx] = {
                        "same_token": {
                            "mean": mean_sim_same,
                            "std": std_sim_same,
                            "min": min_sim_same,
                            "max": max_sim_same,
                        },
                        "baseline_different_token": {
                            "mean": mean_sim_baseline,
                            "std": std_sim_baseline,
                            "min": min_sim_baseline,
                            "max": max_sim_baseline,
                        },
                        "num_patches": len(similarity_same_per_patch)
                    }
                    
                    # Accumulate for global average (same token)
                    if layer_idx not in layer_similarities_sum:
                        layer_similarities_sum[layer_idx] = {"same": 0.0, "baseline": 0.0}
                        layer_similarities_count[layer_idx] = 0
                    
                    layer_similarities_sum[layer_idx]["same"] += mean_sim_same * len(similarity_same_per_patch)
                    layer_similarities_sum[layer_idx]["baseline"] += mean_sim_baseline * len(similarity_baseline_per_patch)
                    layer_similarities_count[layer_idx] += len(similarity_same_per_patch)
                    
                    # Clear intermediate tensors
                    del image_features_layerN, image_features_layerN_norm, similarity_same, similarity_same_per_patch
                    del similarity_baseline_per_patch, layer0_flat, layerN_flat
                
                # Store layer 0 info (reference layer)
                B, num_chunks, patches_per_chunk, hidden_dim = image_features_layer0.shape
                image_results["layer_0_info"] = {
                    "shape": [B, num_chunks, patches_per_chunk, hidden_dim],
                    "num_patches": num_chunks * patches_per_chunk
                }
                
                # Clear intermediate tensors
                del images_tensor, image_masks_tensor, input_ids, image_input_idx_tensor
                del image_features_layer0, image_features_layer0_norm, hidden_states, output
                clear_gpu_memory()
        
        results.append(image_results)
        clear_gpu_memory()
    
    # Compute final averages
    global_layer_similarities = {}
    for layer_idx in layer_similarities_sum:
        total_sum_same = layer_similarities_sum[layer_idx]["same"]
        total_sum_baseline = layer_similarities_sum[layer_idx]["baseline"]
        total_count = layer_similarities_count[layer_idx]
        global_layer_similarities[layer_idx] = {
            "same_token": {
                "mean_similarity": total_sum_same / total_count if total_count > 0 else 0.0,
            },
            "baseline_different_token": {
                "mean_similarity": total_sum_baseline / total_count if total_count > 0 else 0.0,
            },
            "total_patches": total_count
        }
    
    return results, global_layer_similarities


def main():
    parser = argparse.ArgumentParser(description="Compute vision token similarity across layers (Single-GPU)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint to analyze")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--max-layers", type=int, default=None,
                       help="Maximum layer to process (default: all layers)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/sameToken_acrossLayers_similarity",
                       help="Output directory for results")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    print(f"{'='*80}")
    print(f"Vision Token Similarity Across Layers (Single-GPU)")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset split: {args.split}")
    print(f"Number of images: {args.num_images}")
    print(f"Max layers: {args.max_layers if args.max_layers else 'all'}")
    print()
    
    # Load model
    print(f"Loading model from {args.ckpt_path}")
    
    # Load model on CPU first
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    
    model = Molmo(cfg.model)
    
    # Check checkpoint size to determine if it's a full or stripped checkpoint
    checkpoint_file = f"{args.ckpt_path}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    
    is_full_checkpoint = checkpoint_size_gb > 1.0
    
    if not is_full_checkpoint:
        # Small checkpoint - only contains connector weights, need to load pretrained LLM/ViT
        print(f"Detected stripped checkpoint ({checkpoint_size_gb:.2f} GB)")
        print("Loading pretrained weights (LLM + ViT)...")
        model.reset_with_pretrained_weights()
        print("Pretrained weights loaded")
    else:
        print(f"Detected full checkpoint ({checkpoint_size_gb:.2f} GB)")
        print("Skipping pretrained weights loading (checkpoint contains all weights)")
    
    # Load checkpoint weights
    print("Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    
    # Free checkpoint memory immediately
    num_params = len(checkpoint_weights)
    del checkpoint_weights
    gc.collect()
    
    print(f"Loaded {num_params} parameter tensors from checkpoint")
    
    # Move model to GPU (single-GPU, no FSDP)
    print("Moving model to GPU (fp16)...")
    model = model.half().cuda().eval()
    torch.cuda.empty_cache()
    print(f"Model loaded on device: {device}\n")
    
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
    print(f"Loading PixMo-Cap {args.split} split...")
    dataset = PixMoCap(split=args.split, mode="captions")
    print()
    
    # Process images
    prompt = "Describe this image in detail."
    print(f"Processing {args.num_images} images...")
    results, global_similarities = process_images(
        model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
        max_layers=args.max_layers, device=device
    )
    
    # Save results
    # Setup output directory
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_output_file = output_dir / f"similarity_across_layers_detailed.json"
    print(f"\n✓ Saving detailed results to {detailed_output_file}...")
    
    detailed_output_data = {
        'checkpoint': args.ckpt_path,
        'split': args.split,
        'num_images': args.num_images,
        'max_layers': args.max_layers,
        'per_image_results': results
    }
    
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output_data, f, indent=2, ensure_ascii=False)
    
    # Save summary (global averages)
    summary_output_file = output_dir / f"similarity_across_layers_summary.json"
    print(f"✓ Saving summary to {summary_output_file}...")
    
    summary_output_data = {
        'checkpoint': args.ckpt_path,
        'split': args.split,
        'num_images': args.num_images,
        'max_layers': args.max_layers,
        'global_averages': global_similarities
    }
    
    with open(summary_output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: Average Cosine Similarity to Layer 0")
    print(f"{'='*80}")
    print(f"{'Layer':<10} {'Same Token':<20} {'Different Tokens (Same Image)':<20} {'Total Patches':<15}")
    print(f"{'-'*80}")
    for layer_idx in sorted(global_similarities.keys()):
        mean_sim_same = global_similarities[layer_idx]['same_token']['mean_similarity']
        mean_sim_baseline = global_similarities[layer_idx]['baseline_different_token']['mean_similarity']
        total_patches = global_similarities[layer_idx]['total_patches']
        print(f"{layer_idx:<10} {mean_sim_same:<20.6f} {mean_sim_baseline:<20.6f} {total_patches:<15}")
    print(f"{'='*80}\n")
    
    print(f"✓ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

