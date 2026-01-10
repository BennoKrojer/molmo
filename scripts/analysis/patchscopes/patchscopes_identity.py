#!/usr/bin/env python3
"""
Patchscopes Identity Prompt Analysis for Visual Tokens

Implements the token identity Patchscope from:
"Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models"
(Ghandeharioun et al., ICML 2024)

The identity prompt uses few-shot demonstrations to decode token identity:
    "cat->cat; 1135->1135; hello->hello; ?"

For visual tokens:
1. Run image through model to get visual hidden states at layer l
2. Patch each visual token's hidden state into identity prompt at "?" position
3. Continue forward pass from layer l to get predicted tokens

This implements l→l patching (same source and target layer), as recommended in the paper.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/patchscopes/patchscopes_identity.py \
        --ckpt-path <path> --num-images 100 --layers 0,1,2,4,8,16,24
"""

import argparse
import gc
import json
import math
import os
import time
import numpy as np
import torch
from pathlib import Path

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


# Identity prompt template (from Patchscopes paper)
# Paper format: "tok1->tok1; tok2->tok2; ... ; X" where X is the placeholder
# The model learns to repeat tokens, so after patching X with hidden state,
# it should output what the hidden state encodes (the next token prediction)
IDENTITY_PROMPT = "cat->cat; dog->dog; hello->hello; X"


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def decode_token(tokenizer, idx):
    """Decode a token to string."""
    token = tokenizer.decode([int(idx)])
    return token.encode('utf-8').decode('utf-8')


class PatchscopesHook:
    """
    Hook to patch hidden states at a specific layer and position.

    Uses a pre-hook to modify the input to a transformer block.
    This allows us to inject a foreign hidden state (from visual tokens)
    into the forward pass of the identity prompt.
    """

    def __init__(self, patch_position, patch_hidden_states):
        """
        Args:
            patch_position: Position in sequence to patch (the "?" token)
            patch_hidden_states: Tensor [batch_size, hidden_dim] to inject
        """
        self.patch_position = patch_position
        self.patch_hidden_states = patch_hidden_states
        self.handle = None

    def hook_fn(self, module, args):
        """
        Pre-hook that modifies input hidden states.

        In OLMo/Molmo, the first argument to transformer blocks is the
        hidden states tensor of shape [batch, seq_len, hidden_dim].
        """
        # Get hidden states from args
        if isinstance(args, tuple) and len(args) > 0:
            hidden_states = args[0]
        else:
            return args

        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()

        # Patch at the specified position for all items in batch
        hidden_states[:, self.patch_position, :] = self.patch_hidden_states

        # Return modified args tuple
        return (hidden_states,) + args[1:]

    def register(self, module):
        """Register as a pre-hook on the given module."""
        self.handle = module.register_forward_pre_hook(self.hook_fn)

    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_transformer_blocks(model):
    """Get the transformer blocks from the model."""
    if hasattr(model, 'module'):
        # FSDP wrapped
        return model.module.transformer.blocks
    else:
        return model.transformer.blocks


def run_patchscopes_batch(model, tokenizer, visual_hidden_states, layer_idx, device,
                          batch_size=16, top_k=5):
    """
    Run Patchscopes identity decoding for a batch of visual tokens.

    Args:
        model: The Molmo model
        tokenizer: Tokenizer
        visual_hidden_states: [num_patches, hidden_dim] visual token hidden states
        layer_idx: Which layer to patch at (l→l patching)
        device: CUDA device
        batch_size: Processing batch size
        top_k: Number of top predictions to return

    Returns:
        top_indices: [num_patches, top_k] predicted token indices
        top_values: [num_patches, top_k] logit values
    """
    num_patches = visual_hidden_states.shape[0]

    # Tokenize identity prompt
    identity_tokens = tokenizer.encode(IDENTITY_PROMPT)
    seq_len = len(identity_tokens)
    # Paper format: patch at last position (the "X" placeholder)
    # Then read logits at last position to get prediction
    patch_position = seq_len - 1  # Last position (the "X")

    # Get transformer blocks
    blocks = get_transformer_blocks(model)

    all_top_indices = []
    all_top_values = []

    # Process in batches
    for start_idx in range(0, num_patches, batch_size):
        end_idx = min(start_idx + batch_size, num_patches)
        current_batch_size = end_idx - start_idx

        # Create batch of identity prompt tokens
        batch_tokens = torch.tensor(identity_tokens, device=device).unsqueeze(0)
        batch_tokens = batch_tokens.expand(current_batch_size, -1)  # [batch, seq_len]

        # Get visual hidden states for this batch
        batch_visual_hs = visual_hidden_states[start_idx:end_idx]  # [batch, hidden_dim]

        # Create and register hook on target block
        hook = PatchscopesHook(patch_position, batch_visual_hs)
        target_block = blocks[layer_idx]
        hook.register(target_block)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Run forward pass (hook will patch at layer_idx)
                output = model(
                    input_ids=batch_tokens,
                    output_hidden_states=False,
                )
                logits = output.logits  # [batch, seq_len, vocab_size]

        # Remove hook immediately
        hook.remove()

        # Get logits at LAST position (predicts what comes after "->")
        patch_logits = logits[:, -1, :]  # [batch, vocab_size]

        # Get top-k predictions
        top_vals, top_idxs = torch.topk(patch_logits, k=top_k, dim=-1)

        all_top_indices.append(top_idxs.cpu())
        all_top_values.append(top_vals.cpu())

        # Clean up
        del output, logits, patch_logits, batch_tokens
        torch.cuda.empty_cache()

    return torch.cat(all_top_indices, dim=0), torch.cat(all_top_values, dim=0)


def extract_visual_hidden_states(hidden_states, image_input_idx):
    """
    Extract visual token hidden states from full sequence hidden states.

    Args:
        hidden_states: [B, seq_len, hidden_dim]
        image_input_idx: [B, num_chunks, patches_per_chunk] positions of visual tokens

    Returns:
        visual_hs: [B, num_chunks, patches_per_chunk, hidden_dim]
    """
    B = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[-1]
    num_chunks = image_input_idx.shape[1]
    patches_per_chunk = image_input_idx.shape[2]

    visual_hs = torch.zeros(
        (B, num_chunks, patches_per_chunk, hidden_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )

    flat_positions = image_input_idx.view(B, -1)
    valid_mask = flat_positions >= 0

    for b in range(B):
        valid_pos = flat_positions[b][valid_mask[b]]
        if valid_pos.numel() > 0:
            visual_hs.view(B, -1, hidden_dim)[b, valid_mask[b], :] = \
                hidden_states[b, valid_pos.long(), :]

    return visual_hs


def process_image_patchscopes(model, tokenizer, batch_data, layer_idx, device,
                               top_k=5, batch_size=16):
    """
    Process a single image with Patchscopes at a specific layer.

    Args:
        model: Molmo model
        tokenizer: Tokenizer
        batch_data: Dict with preprocessed image data
        layer_idx: Layer to analyze
        device: CUDA device
        top_k: Number of top predictions
        batch_size: Batch size for Patchscopes

    Returns:
        layer_results: Dict with predictions for all patches
    """
    # Move to device
    input_ids = batch_data['input_tokens'].unsqueeze(0).to(device)
    images = batch_data['images'].unsqueeze(0).to(device)
    image_masks = batch_data['image_masks'].unsqueeze(0).to(device) if batch_data['image_masks'] is not None else None
    image_input_idx = batch_data['image_input_idx'].unsqueeze(0).to(device) if batch_data['image_input_idx'] is not None else None

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            # Forward pass to get hidden states
            output = model(
                input_ids=input_ids,
                images=images,
                image_masks=image_masks,
                image_input_idx=image_input_idx,
                output_hidden_states=True,
                last_logits_only=False,
            )

            hidden_states = output.hidden_states

            # Get layer hidden states
            if layer_idx >= len(hidden_states):
                return None

            hs = hidden_states[layer_idx]  # [B, seq_len, hidden_dim]

            # Extract visual token hidden states
            visual_hs = extract_visual_hidden_states(hs, image_input_idx)

            # Get shape info
            B, num_chunks, patches_per_chunk, hidden_dim = visual_hs.shape

            # Flatten for batch processing
            visual_hs_flat = visual_hs.view(-1, hidden_dim)  # [num_patches, hidden_dim]

            # Clean up before Patchscopes (save memory)
            del hidden_states, output, hs
            torch.cuda.empty_cache()

    # Run Patchscopes
    top_indices, top_values = run_patchscopes_batch(
        model, tokenizer, visual_hs_flat, layer_idx, device,
        batch_size=batch_size, top_k=top_k
    )

    # Build results in LogitLens-compatible format
    chunks_results = []
    patch_counter = 0

    for chunk_idx in range(num_chunks):
        chunk_results = {
            "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
            "patches": []
        }

        for local_patch_idx in range(patches_per_chunk):
            row, col = patch_idx_to_row_col(local_patch_idx, patches_per_chunk)

            top_predictions = []
            for k in range(top_k):
                token_idx = top_indices[patch_counter, k].item()
                logit_val = top_values[patch_counter, k].item()
                token_str = decode_token(tokenizer, token_idx)
                top_predictions.append({
                    "token": token_str,
                    "token_id": token_idx,
                    "logit": logit_val
                })

            chunk_results["patches"].append({
                "patch_idx": local_patch_idx,
                "patch_row": row,
                "patch_col": col,
                "top_predictions": top_predictions
            })

            patch_counter += 1

        chunks_results.append(chunk_results)

    # Clean up
    del visual_hs, visual_hs_flat, images, image_masks, input_ids, image_input_idx
    torch.cuda.empty_cache()

    return {
        "layer_idx": layer_idx,
        "chunks": chunks_results
    }


def main():
    parser = argparse.ArgumentParser(description="Patchscopes Identity Analysis for Visual Tokens")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to Molmo checkpoint")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to process")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top predictions per patch")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (default: standard set)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for Patchscopes forward passes")
    parser.add_argument("--output-dir", type=str, default="analysis_results/patchscopes",
                       help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda")

    print("=" * 70)
    print("PATCHSCOPES IDENTITY ANALYSIS FOR VISUAL TOKENS")
    print("=" * 70)
    print(f"Paper: Ghandeharioun et al., ICML 2024")
    print(f"Identity prompt: \"{IDENTITY_PROMPT}\"")
    print(f"Method: l→l patching (same source and target layer)")
    print()
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Images: {args.num_images}")
    print(f"Split: {args.split}")
    print(f"Top-k: {args.top_k}")
    print(f"Batch size: {args.batch_size}")
    print()

    # ===== LOAD MODEL =====
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    load_start = time.time()

    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)

    ckpt_file = f"{args.ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024**3)

    if ckpt_size_gb < 1.0:
        print(f"  Stripped checkpoint ({ckpt_size_gb:.2f} GB) - loading pretrained...")
        model.reset_with_pretrained_weights()

    print(f"  Loading checkpoint weights...")
    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    print(f"  Moving to GPU (fp16)...")
    model = model.half().cuda().eval()
    torch.cuda.empty_cache()

    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")
    print()

    # ===== SETUP =====
    # Create preprocessor
    model_config = ModelConfig.load(
        resource_path(args.ckpt_path, "config.yaml"),
        key="model",
        validate_paths=False
    )
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    tokenizer = preprocessor.tokenizer

    # Determine layers
    num_layers = len(get_transformer_blocks(model))

    if args.layers:
        llm_layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        # Default: standard layer set from project
        # Layers: 0, 1, 2, 4, 8, 16, 24, N-2, N-1
        llm_layers = [0, 1, 2, 4, 8, 16, 24]
        llm_layers.extend([num_layers - 2, num_layers - 1])
        llm_layers = sorted(set(l for l in llm_layers if l < num_layers))

    print(f"Model has {num_layers} transformer layers")
    print(f"Analyzing layers: {llm_layers}")
    print()

    # Tokenize identity prompt and show info
    identity_tokens = tokenizer.encode(IDENTITY_PROMPT)
    print(f"Identity prompt tokens: {identity_tokens}")
    print(f"Identity prompt length: {len(identity_tokens)} tokens")
    print(f"Patch position (last token): {len(identity_tokens) - 1}")
    print()

    # Load dataset
    dataset = PixMoCap(split=args.split, mode="captions")
    prompt = "Describe this image in detail."

    # Output setup
    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== PROCESS IMAGES =====
    print("=" * 70)
    print("PROCESSING IMAGES")
    print("=" * 70)

    # Storage: results[layer_idx] = list of image results
    all_results = {layer_idx: [] for layer_idx in llm_layers}

    total_start = time.time()

    for img_idx in range(args.num_images):
        img_start = time.time()

        # Load and preprocess image
        example_data = dataset.get(img_idx, np.random)

        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            caption_text = example_data["message_list"][0].get("text", "")

        example = {"image": example_data["image"], "messages": [prompt]}
        batch = preprocessor(example, rng=np.random)

        # Prepare batch data
        batch_data = {
            'input_tokens': torch.tensor(batch["input_tokens"]),
            'images': torch.tensor(batch.get("images")),
            'image_masks': torch.tensor(batch.get("image_masks")) if batch.get("image_masks") is not None else None,
            'image_input_idx': torch.tensor(batch.get("image_input_idx")) if batch.get("image_input_idx") is not None else None,
        }

        # Process each layer
        for layer_idx in llm_layers:
            layer_result = process_image_patchscopes(
                model, tokenizer, batch_data, layer_idx, device,
                top_k=args.top_k, batch_size=args.batch_size
            )

            if layer_result is not None:
                all_results[layer_idx].append({
                    "image_idx": img_idx,
                    "ground_truth_caption": caption_text,
                    "chunks": layer_result["chunks"]
                })

        img_time = time.time() - img_start
        if (img_idx + 1) % 10 == 0 or img_idx == 0:
            elapsed = time.time() - total_start
            rate = (img_idx + 1) / elapsed
            eta = (args.num_images - img_idx - 1) / rate if rate > 0 else 0
            print(f"  Image {img_idx + 1}/{args.num_images} ({img_time:.1f}s) | "
                  f"Rate: {rate:.2f} img/s | ETA: {eta/60:.1f} min")

        # Clean up
        del batch_data
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print()
    print(f"✓ Processed {args.num_images} images in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average: {total_time/args.num_images:.2f}s per image")
    print()

    # ===== SAVE RESULTS =====
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    for layer_idx in llm_layers:
        output_file = output_dir / f"patchscopes_identity_layer{layer_idx}_topk{args.top_k}.json"

        output_data = {
            'checkpoint': args.ckpt_path,
            'method': 'patchscopes_identity',
            'identity_prompt': IDENTITY_PROMPT,
            'patching_mode': 'l_to_l',  # Same source and target layer
            'split': args.split,
            'num_images': args.num_images,
            'top_k': args.top_k,
            'layer_idx': layer_idx,
            'num_layers_total': num_layers,
            'processing_time_seconds': total_time,
            'results': all_results[layer_idx]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Layer {layer_idx}: {output_file.name}")

    print()
    print(f"✓ Results saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
