"""
Compute L2 norm of vision tokens across LLM layers.

This script traces vision tokens through the LLM layers and computes the L2 norm
of each vision token at each layer.

For each layer, we compute:
- Per-patch L2 norm
- Histogram of L2 norms
- Mean, std, min, max

This helps understand how vision token magnitudes change through the LLM layers.

Single-GPU version (no FSDP overhead).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/sameToken_acrossLayers_l2norm.py --ckpt-path <path> --num-images 100
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
                   target_layers=None, device=None, num_bins=50):
    """
    Process images and compute L2 norm of vision tokens across layers.

    For each image:
    1. Extract vision tokens at layer 0 (input)
    2. Extract vision tokens at specified layers (through LLM)
    3. Compute L2 norm of each token
    4. Accumulate histogram data
    """
    if device is None:
        device = torch.device("cuda")

    if target_layers is None:
        target_layers = [0, 4, 8, 24]

    results = []

    # Accumulators for histogram data across all images
    layer_norms_all = {layer: [] for layer in target_layers}

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
            "layer_norms": {}
        }

        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None

                # Extract layer 0 tokens (vision features AFTER connector MLP, in LLM embedding space)
                # Note: return_tokens_before_MLP=True returns (after_MLP, before_MLP), we take the first
                image_features_layer0, _ = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    image_features_layer0 = image_features_layer0[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    image_features_layer0 = image_features_layer0[:, :, use_n_token_only, :]

                # Compute L2 norm at layer 0
                if 0 in target_layers:
                    l2_norm_layer0 = torch.linalg.norm(image_features_layer0.float(), dim=-1)  # [B, num_chunks, patches]
                    l2_flat = l2_norm_layer0.view(-1).cpu().numpy()

                    image_results["layer_norms"][0] = {
                        "mean": float(np.mean(l2_flat)),
                        "std": float(np.std(l2_flat)),
                        "min": float(np.min(l2_flat)),
                        "max": float(np.max(l2_flat)),
                        "num_patches": len(l2_flat)
                    }
                    layer_norms_all[0].extend(l2_flat.tolist())

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
                num_layers = len(hidden_states)

                # Process each target layer (except 0 which we already did)
                for layer_idx in target_layers:
                    if layer_idx == 0:
                        continue
                    if layer_idx >= num_layers:
                        print(f"Warning: Layer {layer_idx} exceeds model layers ({num_layers}), skipping")
                        continue

                    # Extract visual tokens from this LLM layer
                    layer_hidden_states = hidden_states[layer_idx]
                    image_features_layerN = gather_visual_features(layer_hidden_states, image_input_idx_tensor)

                    # Apply same use_n_token_only filtering as layer 0
                    if type(use_n_token_only) == int and use_n_token_only != -1:
                        image_features_layerN = image_features_layerN[:, :, :use_n_token_only, :]
                    elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                        image_features_layerN = image_features_layerN[:, :, use_n_token_only, :]

                    # Compute L2 norm
                    l2_norm = torch.linalg.norm(image_features_layerN.float(), dim=-1)
                    l2_flat = l2_norm.view(-1).cpu().numpy()

                    image_results["layer_norms"][layer_idx] = {
                        "mean": float(np.mean(l2_flat)),
                        "std": float(np.std(l2_flat)),
                        "min": float(np.min(l2_flat)),
                        "max": float(np.max(l2_flat)),
                        "num_patches": len(l2_flat)
                    }
                    layer_norms_all[layer_idx].extend(l2_flat.tolist())

                    del image_features_layerN, l2_norm

                # Store layer 0 info (reference layer)
                B, num_chunks, patches_per_chunk, hidden_dim = image_features_layer0.shape
                image_results["layer_0_info"] = {
                    "shape": [B, num_chunks, patches_per_chunk, hidden_dim],
                    "num_patches": num_chunks * patches_per_chunk
                }

                # Clear intermediate tensors
                del images_tensor, image_masks_tensor, input_ids, image_input_idx_tensor
                del image_features_layer0, hidden_states, output
                clear_gpu_memory()

        results.append(image_results)
        clear_gpu_memory()

    # Compute histogram data for each layer
    histogram_data = {}
    raw_norms = {}  # Store raw values for flexible replotting
    for layer_idx in target_layers:
        if not layer_norms_all[layer_idx]:
            continue

        all_norms = np.array(layer_norms_all[layer_idx])
        counts, bin_edges = np.histogram(all_norms, bins=num_bins)

        histogram_data[layer_idx] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(np.mean(all_norms)),
            "std": float(np.std(all_norms)),
            "min": float(np.min(all_norms)),
            "max": float(np.max(all_norms)),
            "median": float(np.median(all_norms)),
            "n_samples": len(all_norms)
        }

        # Store raw values (as float32 to save space)
        raw_norms[layer_idx] = all_norms.astype(np.float32).tolist()

    return results, histogram_data, raw_norms


def main():
    parser = argparse.ArgumentParser(description="Compute vision token L2 norm across layers (Single-GPU)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint to analyze")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--target-layers", type=str, default="0,4,8,24",
                       help="Comma-separated list of layers to analyze (default: 0,4,8,24)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/sameToken_acrossLayers_l2norm",
                       help="Output directory for results")
    parser.add_argument("--num-bins", type=int, default=50,
                       help="Number of histogram bins")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Parse target layers
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    print(f"{'='*80}")
    print(f"Vision Token L2 Norm Across Layers (Single-GPU)")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset split: {args.split}")
    print(f"Number of images: {args.num_images}")
    print(f"Target layers: {target_layers}")
    print(f"Histogram bins: {args.num_bins}")
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
    results, histogram_data, raw_norms = process_images(
        model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
        target_layers=target_layers, device=device, num_bins=args.num_bins
    )

    # Save results
    # Setup output directory
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_output_file = output_dir / f"l2norm_across_layers_detailed.json"
    print(f"\n✓ Saving detailed results to {detailed_output_file}...")

    detailed_output_data = {
        'checkpoint': args.ckpt_path,
        'split': args.split,
        'num_images': args.num_images,
        'target_layers': target_layers,
        'modality': 'vision',
        'per_image_results': results
    }

    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output_data, f, indent=2, ensure_ascii=False)

    # Save summary (histogram data)
    summary_output_file = output_dir / f"l2norm_across_layers_summary.json"
    print(f"✓ Saving summary to {summary_output_file}...")

    summary_output_data = {
        'checkpoint': args.ckpt_path,
        'split': args.split,
        'num_images': args.num_images,
        'target_layers': target_layers,
        'modality': 'vision',
        'histogram_data': histogram_data
    }

    with open(summary_output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_output_data, f, indent=2, ensure_ascii=False)

    # Save raw norms for flexible replotting
    raw_output_file = output_dir / f"l2norm_raw_values.json"
    print(f"✓ Saving raw values to {raw_output_file}...")

    raw_output_data = {
        'checkpoint': args.ckpt_path,
        'modality': 'vision',
        'target_layers': target_layers,
        'raw_norms': {str(k): v for k, v in raw_norms.items()}  # Convert int keys to str for JSON
    }

    with open(raw_output_file, 'w', encoding='utf-8') as f:
        json.dump(raw_output_data, f)  # No indent to save space

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: L2 Norm Statistics per Layer (Vision Tokens)")
    print(f"{'='*80}")
    print(f"{'Layer':<10} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15} {'Samples':<10}")
    print(f"{'-'*80}")
    for layer_idx in sorted(histogram_data.keys()):
        data = histogram_data[layer_idx]
        print(f"{layer_idx:<10} {data['mean']:<15.4f} {data['std']:<15.4f} {data['min']:<15.4f} {data['max']:<15.4f} {data['n_samples']:<10}")
    print(f"{'='*80}\n")

    print(f"✓ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
