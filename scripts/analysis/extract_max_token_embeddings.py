#!/usr/bin/env python3
"""
Extract embedding values of the max L2 norm vision token for each model.

Creates histograms of individual embedding dimensions to understand if high L2 norm
is driven by a few large values or uniformly larger values.

Builds on the structure of sameToken_acrossLayers_l2norm.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/extract_max_token_embeddings.py \
        --ckpt-path molmo_data/checkpoints/.../step12000-unsharded \
        --num-images 10
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def gather_visual_features(hs_tensor, image_input_idx_tensor):
    """Extract visual tokens from hidden states using image_input_idx."""
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


def find_max_token(model, preprocessor, dataset, num_images, prompt, use_n_token_only,
                   target_layers, device):
    """
    Find the vision token with maximum L2 norm across all images and layers.
    Returns the full embedding of that token.
    """
    max_l2_norm = 0
    max_embedding = None
    max_info = {}

    # Also track stats per layer
    layer_stats = {layer: {'max_norm': 0, 'all_norms': []} for layer in target_layers}

    for i in tqdm(range(num_images), desc="Processing images"):
        example_data = dataset.get(i, np.random)
        example = {"image": example_data["image"], "messages": [prompt]}
        batch = preprocessor(example, rng=np.random)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None

                # Get layer 0 features (after connector MLP)
                image_features_layer0, _ = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    image_features_layer0 = image_features_layer0[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    image_features_layer0 = image_features_layer0[:, :, use_n_token_only, :]

                # Check layer 0
                if 0 in target_layers:
                    feats = image_features_layer0.float()  # [1, chunks, patches, dim]
                    l2_norms = torch.linalg.norm(feats, dim=-1).view(-1)  # [total_patches]
                    layer_stats[0]['all_norms'].extend(l2_norms.cpu().tolist())

                    max_idx = l2_norms.argmax().item()
                    max_norm = l2_norms[max_idx].item()
                    layer_stats[0]['max_norm'] = max(layer_stats[0]['max_norm'], max_norm)

                    if max_norm > max_l2_norm:
                        max_l2_norm = max_norm
                        max_embedding = feats.view(-1, feats.shape[-1])[max_idx].cpu().numpy()
                        max_info = {'image_idx': i, 'token_idx': max_idx, 'l2_norm': max_norm, 'layer': 0}

                # Forward through LLM
                output = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=image_input_idx_tensor,
                    output_hidden_states=True
                )

                # Check other layers
                for layer_idx in target_layers:
                    if layer_idx == 0:
                        continue
                    if layer_idx >= len(output.hidden_states):
                        continue

                    hs = output.hidden_states[layer_idx]
                    feats = gather_visual_features(hs.float(), image_input_idx_tensor)  # [1, chunks, patches, dim]
                    l2_norms = torch.linalg.norm(feats, dim=-1).view(-1)
                    layer_stats[layer_idx]['all_norms'].extend(l2_norms.cpu().tolist())

                    max_idx = l2_norms.argmax().item()
                    max_norm = l2_norms[max_idx].item()
                    layer_stats[layer_idx]['max_norm'] = max(layer_stats[layer_idx]['max_norm'], max_norm)

                    if max_norm > max_l2_norm:
                        max_l2_norm = max_norm
                        max_embedding = feats.view(-1, feats.shape[-1])[max_idx].cpu().numpy()
                        max_info = {'image_idx': i, 'token_idx': max_idx, 'l2_norm': max_norm, 'layer': layer_idx}

                clear_gpu_memory()

        clear_gpu_memory()

    return max_embedding, max_info, layer_stats


def main():
    parser = argparse.ArgumentParser(description="Extract max L2 norm vision token embedding")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="analysis_results/max_token_embeddings")
    parser.add_argument("--target-layers", type=str, default="0,4,8,16,24,31",
                        help="Layers to check for max (comma-separated)")
    args = parser.parse_args()

    device = torch.device("cuda")
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    print(f"{'='*80}")
    print(f"Extract Max L2 Norm Vision Token Embedding")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Target layers: {target_layers}")
    print(f"Number of images: {args.num_images}")
    print()

    # Load model (same as sameToken_acrossLayers_l2norm.py)
    print(f"Loading model from {args.ckpt_path}")
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)

    checkpoint_file = f"{args.ckpt_path}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    is_full_checkpoint = checkpoint_size_gb > 1.0

    if not is_full_checkpoint:
        print(f"Detected stripped checkpoint ({checkpoint_size_gb:.2f} GB), loading pretrained...")
        model.reset_with_pretrained_weights()
    else:
        print(f"Detected full checkpoint ({checkpoint_size_gb:.2f} GB)")

    print("Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    num_params = len(checkpoint_weights)
    del checkpoint_weights
    gc.collect()
    print(f"Loaded {num_params} parameter tensors")

    print("Moving model to GPU (fp16)...")
    model = model.half().cuda().eval()
    torch.cuda.empty_cache()
    print(f"Model loaded on device: {device}\n")

    # Preprocessor
    model_config = ModelConfig.load(resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(model_config, for_inference=True, shuffle_messages=False,
                                          is_training=False, require_image_features=True)
    use_n_token_only = model_config.vision_backbone.use_n_token_only

    # Dataset
    print("Loading PixMo-Cap validation split...")
    dataset = PixMoCap(split="validation", mode="captions")

    # Find max token
    prompt = "Describe this image in detail."
    print(f"\nProcessing {args.num_images} images...")
    max_embedding, max_info, layer_stats = find_max_token(
        model, preprocessor, dataset, args.num_images, prompt, use_n_token_only,
        target_layers, device
    )

    print(f"\n{'='*80}")
    print(f"Results")
    print(f"{'='*80}")
    print(f"Max L2 norm: {max_info['l2_norm']:.2f} at layer {max_info['layer']}")
    print(f"  Image idx: {max_info['image_idx']}, Token idx: {max_info['token_idx']}")
    print(f"  Embedding dim: {len(max_embedding)}")

    # Embedding statistics
    stats = {
        'mean': float(np.mean(max_embedding)),
        'std': float(np.std(max_embedding)),
        'min': float(np.min(max_embedding)),
        'max': float(np.max(max_embedding)),
        'abs_mean': float(np.mean(np.abs(max_embedding))),
        'percentiles': {
            'p1': float(np.percentile(max_embedding, 1)),
            'p5': float(np.percentile(max_embedding, 5)),
            'p50': float(np.percentile(max_embedding, 50)),
            'p95': float(np.percentile(max_embedding, 95)),
            'p99': float(np.percentile(max_embedding, 99)),
        }
    }
    print(f"  mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"  min={stats['min']:.4f}, max={stats['max']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = args.ckpt_path.split("/")[-2] + "_" + args.ckpt_path.split("/")[-1]
    output_file = output_dir / f"{ckpt_name}_max_token.json"

    results = {
        'checkpoint': ckpt_name,
        'num_images': args.num_images,
        'target_layers': target_layers,
        'max_info': max_info,
        'max_embedding_values': max_embedding.tolist(),
        'embedding_dim': len(max_embedding),
        'embedding_stats': stats,
        'layer_stats': {str(k): {'max_norm': v['max_norm']} for k, v in layer_stats.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
