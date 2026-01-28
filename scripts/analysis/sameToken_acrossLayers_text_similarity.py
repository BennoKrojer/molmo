"""
Compute cosine similarity of TEXT tokens across layers to layer 0.

This script traces text tokens through the LLM layers and computes the cosine similarity
between each text token at layer N and the same token at layer 0 (raw embedding lookup).

For each layer, we compute:
- Per-token similarity (each text token compared to its layer-0 version)
- Average similarity across all tokens
- Average similarity across all images/captions

This helps understand how text tokens evolve through the LLM layers when contextualized
by vision tokens, and whether they stay close to their original representation or drift away.

Single-GPU version (no FSDP overhead).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/sameToken_acrossLayers_text_similarity.py --ckpt-path <path> --num-images 100
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


def debug_token_layout(input_ids, image_input_idx, tokenizer, caption_text, image_idx=0, special_token_ids=None):
    """Debug helper to understand the token layout."""
    print(f"\n{'='*80}")
    print(f"DEBUG: Image {image_idx} Token Layout")
    print(f"{'='*80}")
    print(f"Caption (first 200 chars): {caption_text[:200]}...")
    print(f"Total sequence length: {input_ids.shape[1]}")
    
    if special_token_ids:
        print(f"Special token IDs to filter: {sorted(special_token_ids)}")
    
    # Flatten image_input_idx to see all image positions
    if image_input_idx is not None:
        flat_img_idx = image_input_idx.view(-1)
        valid_img_positions = flat_img_idx[flat_img_idx >= 0].cpu().numpy()
        print(f"Number of image token positions: {len(valid_img_positions)}")
        print(f"Image positions range: {valid_img_positions.min()} to {valid_img_positions.max()}")
        print(f"First 10 image positions: {valid_img_positions[:10]}")
    else:
        valid_img_positions = []
        print("No image_input_idx provided")
    
    # Decode some tokens to see what they are
    input_ids_cpu = input_ids[0].cpu().numpy()
    print(f"\nFirst 20 tokens (with positions):")
    for pos in range(min(20, len(input_ids_cpu))):
        token_id = input_ids_cpu[pos]
        is_img = pos in valid_img_positions
        is_special = special_token_ids and token_id in special_token_ids
        token_str = tokenizer.decode([token_id]) if token_id >= 0 else "<pad>"
        marker = " [IMG]" if is_img else (" [SPECIAL]" if is_special else "")
        print(f"  pos {pos:4d}: id={token_id:6d} -> '{token_str}'{marker}")
    
    # Find text regions (non-image, non-special tokens)
    all_positions = set(range(len(input_ids_cpu)))
    img_positions_set = set(valid_img_positions)
    
    # Also filter out special tokens
    special_positions = set()
    if special_token_ids:
        for pos in range(len(input_ids_cpu)):
            if input_ids_cpu[pos] in special_token_ids:
                special_positions.add(pos)
    
    text_positions = sorted(all_positions - img_positions_set - special_positions)
    
    print(f"\nFiltered out {len(special_positions)} special token positions")
    print(f"Text token positions (first 30): {text_positions[:30]}")
    print(f"Total actual text tokens (after filtering): {len(text_positions)}")
    
    # Decode text tokens
    if len(text_positions) > 0:
        print(f"\nFirst 20 actual text tokens:")
        for i, pos in enumerate(text_positions[:20]):
            token_id = input_ids_cpu[pos]
            token_str = tokenizer.decode([token_id]) if token_id >= 0 else "<pad>"
            print(f"  pos {pos:4d}: id={token_id:6d} -> '{token_str}'")
        
        # Decode the full text (to verify it's the caption)
        text_token_ids = [input_ids_cpu[pos] for pos in text_positions if input_ids_cpu[pos] >= 0]
        full_text = tokenizer.decode(text_token_ids)
        print(f"\nDecoded caption text (first 400 chars): {full_text[:400]}...")
    
    print(f"{'='*80}\n")
    return text_positions


def process_images(model, preprocessor, dataset, num_images, 
                   max_layers=None, device=None, debug=True, special_token_ids=None):
    """
    Process images and compute similarity of text tokens across layers to layer 0.
    
    For each image/caption:
    1. Extract text tokens at layer 0 (raw embedding lookup)
    2. Extract text tokens at layers 1, 2, 3, ... (through LLM, contextualized by vision)
    3. Compute cosine similarity between layer N tokens and layer 0 tokens (same position)
    4. Average across tokens and images
    """
    if device is None:
        device = torch.device("cuda")
    
    # Get tokenizer for debugging
    tokenizer = preprocessor.tokenizer
    
    results = []
    
    # Accumulators for averaging across images
    layer_similarities_sum = {}  # layer_idx -> {"same": sum, "baseline": sum}
    layer_similarities_count = {}  # layer_idx -> count of tokens
    
    for i in tqdm(range(num_images), desc="Processing images"):
        example_data = dataset.get(i, np.random)
        
        # Extract ground truth caption for metadata
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            caption_text = message.get("text", "")
        
        # Pass the actual message_list from dataset (contains caption + formatting)
        # This is what training uses - the formatter handles converting message_list to tokens
        example = {
            "image": example_data["image"],
            "message_list": example_data["message_list"]  # Contains the actual caption!
        }
        
        # Preprocess
        batch = preprocessor(example, rng=np.random)
        
        # Initialize results for this image
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text,
            "layer_similarities": {}
        }
        
        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move to GPU
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                
                # DEBUG: Print token layout for first image
                if debug and i == 0:
                    debug_text_positions = debug_token_layout(
                        input_ids, image_input_idx_tensor, tokenizer, caption_text, 
                        image_idx=i, special_token_ids=special_token_ids
                    )
                
                # Get text token mask (True for actual text, False for image/special tokens)
                text_mask = get_text_token_mask(input_ids, image_input_idx_tensor, special_token_ids)
                
                # DEBUG: Verify text_mask matches debug calculation
                if debug and i == 0:
                    our_text_positions = text_mask[0].nonzero(as_tuple=True)[0].cpu().tolist()
                    print(f"DEBUG: text_mask gives {len(our_text_positions)} text positions")
                    print(f"DEBUG: text_mask positions (first 30): {our_text_positions[:30]}")
                    print(f"DEBUG: Expected from debug_token_layout: {len(debug_text_positions)} positions")
                    if our_text_positions != debug_text_positions:
                        print("WARNING: Mismatch between text_mask and debug calculation!")
                        print(f"  text_mask: {our_text_positions[:10]}")
                        print(f"  debug:     {debug_text_positions[:10]}")
                    else:
                        print("DEBUG: ✓ text_mask matches debug calculation")
                
                # Layer 0: Raw text embeddings from embedding lookup
                # Access the embedding layer directly
                text_embeddings_layer0 = model.transformer.wte(input_ids)  # [B, seq_len, hidden_dim]
                
                # Extract only text token embeddings at layer 0
                # text_mask is [B, seq_len], we need to expand for gathering
                B, seq_len, hidden_dim = text_embeddings_layer0.shape
                
                # Get text positions for this batch
                text_positions = text_mask[0].nonzero(as_tuple=True)[0]  # [num_text_tokens]
                num_text_tokens = text_positions.shape[0]
                
                if num_text_tokens == 0:
                    # No text tokens, skip this image
                    continue
                
                # Extract text embeddings at layer 0
                text_features_layer0 = text_embeddings_layer0[0, text_positions, :]  # [num_text_tokens, hidden_dim]
                text_features_layer0_norm = torch.nn.functional.normalize(text_features_layer0, dim=-1)
                
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
                    num_layers = min(num_layers, max_layers + 1)
                
                # Process each layer
                for layer_idx in range(1, num_layers):  # Start from 1 (layer 0 already extracted)
                    # Extract text tokens from this LLM layer
                    layer_hidden_states = hidden_states[layer_idx]  # [B, seq_len, hidden_dim]
                    
                    # Extract only text token embeddings
                    text_features_layerN = layer_hidden_states[0, text_positions, :]  # [num_text_tokens, hidden_dim]
                    
                    # Normalize for cosine similarity
                    text_features_layerN_norm = torch.nn.functional.normalize(text_features_layerN, dim=-1)
                    
                    # SAME TOKEN: Compute cosine similarity between same-position tokens
                    similarity_same = (text_features_layer0_norm * text_features_layerN_norm).sum(dim=-1)
                    
                    # BASELINE: Compare different tokens (shuffle layer N)
                    shuffled_indices = torch.randperm(num_text_tokens, device=device)
                    text_features_layerN_shuffled = text_features_layerN_norm[shuffled_indices, :]
                    similarity_baseline = (text_features_layer0_norm * text_features_layerN_shuffled).sum(dim=-1)
                    
                    # Compute statistics for same-token similarity
                    mean_sim_same = similarity_same.mean().item()
                    std_sim_same = similarity_same.std().item()
                    min_sim_same = similarity_same.min().item()
                    max_sim_same = similarity_same.max().item()
                    
                    # Compute statistics for baseline (different-token) similarity
                    mean_sim_baseline = similarity_baseline.mean().item()
                    std_sim_baseline = similarity_baseline.std().item()
                    min_sim_baseline = similarity_baseline.min().item()
                    max_sim_baseline = similarity_baseline.max().item()
                    
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
                        "num_text_tokens": num_text_tokens
                    }
                    
                    # Accumulate for global average
                    if layer_idx not in layer_similarities_sum:
                        layer_similarities_sum[layer_idx] = {"same": 0.0, "baseline": 0.0}
                        layer_similarities_count[layer_idx] = 0
                    
                    layer_similarities_sum[layer_idx]["same"] += mean_sim_same * num_text_tokens
                    layer_similarities_sum[layer_idx]["baseline"] += mean_sim_baseline * num_text_tokens
                    layer_similarities_count[layer_idx] += num_text_tokens
                    
                    # Clear intermediate tensors
                    del text_features_layerN, text_features_layerN_norm, similarity_same, similarity_baseline
                
                # Store layer 0 info (reference layer)
                image_results["layer_0_info"] = {
                    "shape": [B, num_text_tokens, hidden_dim],
                    "num_text_tokens": num_text_tokens,
                    "total_seq_len": seq_len
                }
                
                # Clear intermediate tensors
                del input_ids, images_tensor, image_masks_tensor, image_input_idx_tensor
                del text_embeddings_layer0, text_features_layer0, text_features_layer0_norm
                del hidden_states, output
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
            "total_text_tokens": total_count
        }
    
    return results, global_layer_similarities


def main():
    parser = argparse.ArgumentParser(description="Compute text token similarity across layers (Single-GPU)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint to analyze")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--max-layers", type=int, default=None,
                       help="Maximum layer to process (default: all layers)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/sameToken_acrossLayers_text_similarity",
                       help="Output directory for results")
    parser.add_argument("--no-debug", action="store_true",
                       help="Disable debug output (for batch runs)")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    print(f"{'='*80}")
    print(f"Text Token Similarity Across Layers (Single-GPU)")
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
    # IMPORTANT: Use for_inference=False and is_training=True to include the caption in the sequence
    # With for_inference=True, only the image+prompt are included (caption is what model generates)
    # With for_inference=False, the full sequence including caption is included (like training)
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=False,  # Include caption in sequence (not just prompt)
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
    results, global_similarities = process_images(
        model, preprocessor, dataset, args.num_images,
        max_layers=args.max_layers, device=device, 
        debug=(not args.no_debug),
        special_token_ids=special_token_ids
    )
    
    # Save results
    # Setup output directory
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_output_file = output_dir / f"text_similarity_across_layers_detailed.json"
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
    summary_output_file = output_dir / f"text_similarity_across_layers_summary.json"
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
    print(f"Summary: Average Cosine Similarity of Text Tokens to Layer 0")
    print(f"{'='*80}")
    print(f"{'Layer':<10} {'Same Token':<20} {'Different Tokens (Same Caption)':<30} {'Total Tokens':<15}")
    print(f"{'-'*80}")
    for layer_idx in sorted(global_similarities.keys()):
        mean_sim_same = global_similarities[layer_idx]['same_token']['mean_similarity']
        mean_sim_baseline = global_similarities[layer_idx]['baseline_different_token']['mean_similarity']
        total_tokens = global_similarities[layer_idx]['total_text_tokens']
        print(f"{layer_idx:<10} {mean_sim_same:<20.6f} {mean_sim_baseline:<30.6f} {total_tokens:<15}")
    print(f"{'='*80}\n")
    
    print(f"✓ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

