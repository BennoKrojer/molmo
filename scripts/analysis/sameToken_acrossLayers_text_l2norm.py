"""
Compute L2 norm of TEXT tokens across LLM layers.

This script traces text tokens through the LLM layers and computes the L2 norm
of each text token at each layer (layer 0 = raw embedding lookup).

For each layer, we compute:
- Per-token L2 norm
- Histogram of L2 norms
- Mean, std, min, max

This helps understand how text token magnitudes change through the LLM layers
when contextualized by vision tokens.

Single-GPU version (no FSDP overhead).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/sameToken_acrossLayers_text_l2norm.py --ckpt-path <path> --num-images 100
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


def get_text_token_mask(input_ids, image_input_idx, special_token_ids=None):
    """
    Get a mask for text tokens (non-image positions, excluding special tokens).

    Args:
        input_ids: [B, seq_len] - input token IDs
        image_input_idx: [B, num_chunks, patches_per_chunk] - positions of image tokens
        special_token_ids: set of token IDs to exclude (e.g., <im_start>, <im_col>, <im_end>)

    Returns:
        text_mask: [B, seq_len] - True for actual text positions, False for image/special positions
    """
    B, seq_len = input_ids.shape

    # Start with all True (all positions are text)
    text_mask = torch.ones(B, seq_len, dtype=torch.bool, device=input_ids.device)

    # Mark image positions as False
    if image_input_idx is not None:
        # Flatten image_input_idx to [B, num_image_tokens]
        flat_image_idx = image_input_idx.view(B, -1)

        for b in range(B):
            valid_positions = flat_image_idx[b][flat_image_idx[b] >= 0]
            if valid_positions.numel() > 0:
                text_mask[b, valid_positions.long()] = False

    # Filter out special tokens (like <im_start>, <im_col>, <im_end>, BOS, EOS)
    if special_token_ids is not None:
        for b in range(B):
            for pos in range(seq_len):
                token_id = input_ids[b, pos].item()
                if token_id in special_token_ids:
                    text_mask[b, pos] = False

    return text_mask


def get_special_token_ids_to_exclude(tokenizer):
    """
    Get the set of special token IDs that should be excluded from text analysis.
    These are image-related special tokens and BOS/EOS tokens.
    """
    special_ids = set()

    # Get special tokens from tokenizer
    from olmo.tokenizer import (
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_COL_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
    )

    # Try to get token IDs for known special tokens
    special_tokens = [
        DEFAULT_IM_START_TOKEN,  # <im_start>
        DEFAULT_IM_END_TOKEN,    # <im_end>
        DEFAULT_IM_COL_TOKEN,    # <im_col>
        DEFAULT_IMAGE_PATCH_TOKEN,  # <im_patch>
    ]

    for token in special_tokens:
        try:
            token_id = tokenizer.encode(token)
            if len(token_id) == 1:
                special_ids.add(token_id[0])
        except:
            pass

    # Also add BOS and EOS if available
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)

    # Fallback: add known special token IDs from OLMo tokenizer (100257-100281 range)
    # These are typically: BOS=100257, im_start=100278, im_end=100279, im_patch=100280, im_col=100281
    for tid in [100257, 100278, 100279, 100280, 100281]:
        special_ids.add(tid)

    return special_ids


def process_images(model, preprocessor, dataset, num_images,
                   target_layers=None, device=None, special_token_ids=None, num_bins=50):
    """
    Process images and compute L2 norm of text tokens across layers.

    For each image/caption:
    1. Extract text tokens at layer 0 (raw embedding lookup)
    2. Extract text tokens at specified layers (through LLM, contextualized by vision)
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

        # Extract ground truth caption for metadata
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            caption_text = message.get("text", "")

        # Pass the actual message_list from dataset (contains caption + formatting)
        example = {
            "image": example_data["image"],
            "message_list": example_data["message_list"]
        }

        # Preprocess
        batch = preprocessor(example, rng=np.random)

        # Initialize results for this image
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text[:200],  # Truncate for storage
            "layer_norms": {}
        }

        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move to GPU
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None

                # Get text token mask (True for actual text, False for image/special tokens)
                text_mask = get_text_token_mask(input_ids, image_input_idx_tensor, special_token_ids)

                # Get text positions for this batch
                text_positions = text_mask[0].nonzero(as_tuple=True)[0]  # [num_text_tokens]
                num_text_tokens = text_positions.shape[0]

                if num_text_tokens == 0:
                    # No text tokens, skip this image
                    continue

                # Layer 0: Raw text embeddings from embedding lookup
                text_embeddings_layer0 = model.transformer.wte(input_ids)  # [B, seq_len, hidden_dim]
                B, seq_len, hidden_dim = text_embeddings_layer0.shape

                # Extract only text token embeddings at layer 0
                text_features_layer0 = text_embeddings_layer0[0, text_positions, :]  # [num_text_tokens, hidden_dim]

                # Compute L2 norm at layer 0
                if 0 in target_layers:
                    l2_norm_layer0 = torch.linalg.norm(text_features_layer0.float(), dim=-1)  # [num_text_tokens]
                    l2_flat = l2_norm_layer0.cpu().numpy()

                    image_results["layer_norms"][0] = {
                        "mean": float(np.mean(l2_flat)),
                        "std": float(np.std(l2_flat)),
                        "min": float(np.min(l2_flat)),
                        "max": float(np.max(l2_flat)),
                        "num_tokens": len(l2_flat)
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

                    # Extract text tokens from this LLM layer
                    layer_hidden_states = hidden_states[layer_idx]  # [B, seq_len, hidden_dim]
                    text_features_layerN = layer_hidden_states[0, text_positions, :]  # [num_text_tokens, hidden_dim]

                    # Compute L2 norm
                    l2_norm = torch.linalg.norm(text_features_layerN.float(), dim=-1)
                    l2_flat = l2_norm.cpu().numpy()

                    image_results["layer_norms"][layer_idx] = {
                        "mean": float(np.mean(l2_flat)),
                        "std": float(np.std(l2_flat)),
                        "min": float(np.min(l2_flat)),
                        "max": float(np.max(l2_flat)),
                        "num_tokens": len(l2_flat)
                    }
                    layer_norms_all[layer_idx].extend(l2_flat.tolist())

                    del text_features_layerN, l2_norm

                # Store layer 0 info
                image_results["layer_0_info"] = {
                    "num_text_tokens": num_text_tokens,
                    "total_seq_len": seq_len,
                    "hidden_dim": hidden_dim
                }

                # Clear intermediate tensors
                del input_ids, images_tensor, image_masks_tensor, image_input_idx_tensor
                del text_embeddings_layer0, text_features_layer0, hidden_states, output
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
    parser = argparse.ArgumentParser(description="Compute text token L2 norm across layers (Single-GPU)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint to analyze")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--target-layers", type=str, default="0,4,8,24",
                       help="Comma-separated list of layers to analyze (default: 0,4,8,24)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/sameToken_acrossLayers_text_l2norm",
                       help="Output directory for results")
    parser.add_argument("--num-bins", type=int, default=50,
                       help="Number of histogram bins")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Parse target layers
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    print(f"{'='*80}")
    print(f"Text Token L2 Norm Across Layers (Single-GPU)")
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
    # Use for_inference=False and is_training=True to include the caption in the sequence
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=False,  # Include caption in sequence
        shuffle_messages=False,
        is_training=True,  # Training mode includes the full caption
        require_image_features=True
    )

    # Get special token IDs to exclude from text analysis
    special_token_ids = get_special_token_ids_to_exclude(preprocessor.tokenizer)
    print(f"Special token IDs to exclude: {sorted(special_token_ids)}")

    # Load dataset
    print(f"Loading PixMo-Cap {args.split} split...")
    dataset = PixMoCap(split=args.split, mode="captions")
    print()

    # Process images
    print(f"Processing {args.num_images} images...")
    results, histogram_data, raw_norms = process_images(
        model, preprocessor, dataset, args.num_images,
        target_layers=target_layers, device=device,
        special_token_ids=special_token_ids, num_bins=args.num_bins
    )

    # Save results
    # Setup output directory
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_output_file = output_dir / f"text_l2norm_across_layers_detailed.json"
    print(f"\n✓ Saving detailed results to {detailed_output_file}...")

    detailed_output_data = {
        'checkpoint': args.ckpt_path,
        'split': args.split,
        'num_images': args.num_images,
        'target_layers': target_layers,
        'modality': 'text',
        'per_image_results': results
    }

    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output_data, f, indent=2, ensure_ascii=False)

    # Save summary (histogram data)
    summary_output_file = output_dir / f"text_l2norm_across_layers_summary.json"
    print(f"✓ Saving summary to {summary_output_file}...")

    summary_output_data = {
        'checkpoint': args.ckpt_path,
        'split': args.split,
        'num_images': args.num_images,
        'target_layers': target_layers,
        'modality': 'text',
        'histogram_data': histogram_data
    }

    with open(summary_output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_output_data, f, indent=2, ensure_ascii=False)

    # Save raw norms for flexible replotting
    raw_output_file = output_dir / f"text_l2norm_raw_values.json"
    print(f"✓ Saving raw values to {raw_output_file}...")

    raw_output_data = {
        'checkpoint': args.ckpt_path,
        'modality': 'text',
        'target_layers': target_layers,
        'raw_norms': {str(k): v for k, v in raw_norms.items()}  # Convert int keys to str for JSON
    }

    with open(raw_output_file, 'w', encoding='utf-8') as f:
        json.dump(raw_output_data, f)  # No indent to save space

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: L2 Norm Statistics per Layer (Text Tokens)")
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
