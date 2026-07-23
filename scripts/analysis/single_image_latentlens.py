#!/usr/bin/env python3
"""
Single-image LatentLens extraction.

Extract LatentLens (contextual nearest neighbors) for a single user-provided image
across one or more models.

Usage:
    # Single model
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/single_image_latentlens.py \
        --image-path /path/to/image.jpg \
        --models olmo-7b_vit-l-14-336 \
        --output-dir analysis_results/single_image_demo

    # Multiple models
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/single_image_latentlens.py \
        --image-path /path/to/image.jpg \
        --models olmo-7b_vit-l-14-336,llama3-8b_siglip,qwen2-7b_dinov2-large-336 \
        --output-dir analysis_results/single_image_demo

    # Generate viewer HTML as well
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/single_image_latentlens.py \
        --image-path /path/to/image.jpg \
        --models olmo-7b_vit-l-14-336,llama3-8b_siglip \
        --output-dir analysis_results/single_image_demo \
        --generate-viewer
"""

import argparse
import gc
import json
import time
import math
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path


# Model specification mapping
MODEL_CONFIGS = {
    # LLM -> contextual embedding directory
    "olmo-7b": {
        "contextual_dir": "molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview",
        "layers": [0, 1, 2, 4, 8, 16, 24, 30, 31],
        "display_name": "OLMo-7B"
    },
    "llama3-8b": {
        "contextual_dir": "molmo_data/contextual_llm_embeddings_vg/meta-llama_Meta-Llama-3-8B",
        "layers": [0, 1, 2, 4, 8, 16, 24, 30, 31],
        "display_name": "LLaMA3-8B"
    },
    "qwen2-7b": {
        "contextual_dir": "molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B",
        "layers": [0, 1, 2, 4, 8, 16, 24, 26, 27],
        "display_name": "Qwen2-7B"
    }
}

VISION_ENCODER_DISPLAY = {
    "vit-l-14-336": "ViT-L/14-336",
    "dinov2-large-336": "DINOv2-L-336",
    "siglip": "SigLIP"
}


def get_checkpoint_path(model_spec: str) -> str:
    """Convert model spec like 'olmo-7b_vit-l-14-336' to checkpoint path."""
    # Special case: qwen2-7b_vit-l-14-336 has _seed10 suffix
    if model_spec == "qwen2-7b_vit-l-14-336":
        base = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10"
    else:
        base = f"molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_{model_spec}"
    # Support both old (step12000-unsharded) and new (latest-unsharded) checkpoint layouts
    for step in ("step12000-unsharded", "latest-unsharded"):
        candidate = f"{base}/{step}"
        if Path(candidate).exists():
            return candidate
    return f"{base}/step12000-unsharded"  # let downstream raise a clear error


def parse_model_spec(model_spec: str) -> dict:
    """Parse model spec like 'olmo-7b_vit-l-14-336' into components."""
    parts = model_spec.split("_")
    # Find LLM (first part that matches known LLMs)
    llm = None
    ve = None
    for i, part in enumerate(parts):
        if part in MODEL_CONFIGS:
            llm = part
            ve = "_".join(parts[i+1:])
            break

    if llm is None:
        raise ValueError(f"Unknown model spec: {model_spec}. Expected format: llm_vision-encoder")

    return {
        "llm": llm,
        "vision_encoder": ve,
        "checkpoint_path": get_checkpoint_path(model_spec),
        "contextual_dir": MODEL_CONFIGS[llm]["contextual_dir"],
        "layers": MODEL_CONFIGS[llm]["layers"],
        "display_name": f"{MODEL_CONFIGS[llm]['display_name']} + {VISION_ENCODER_DISPLAY.get(ve, ve)}"
    }


def find_available_layers(contextual_dir):
    """Find all layer directories with caches."""
    contextual_path = Path(contextual_dir)
    layers = []
    for layer_dir in contextual_path.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            cache_file = layer_dir / "embeddings_cache.pt"
            if cache_file.exists():
                layer_idx = int(layer_dir.name.split("_")[1])
                layers.append(layer_idx)
    return sorted(layers)


def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def load_and_preprocess_image(image_path: str, preprocessor, device):
    """Load and preprocess a single image."""
    from olmo.data.model_preprocessor import load_image

    # Load image
    pil_image = Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image_array = load_image(pil_image)

    # Create example for preprocessor
    prompt = "Describe this image in detail."
    example = {"image": pil_image, "messages": [prompt]}
    batch = preprocessor(example, rng=np.random)

    # Store preprocessed data on GPU
    cached_data = {
        'images': torch.tensor(batch.get("images")).to(device),
        'image_masks': torch.tensor(batch.get("image_masks")).to(device) if batch.get("image_masks") is not None else None,
        'input_tokens': torch.tensor(batch["input_tokens"]).to(device),
        'image_input_idx': torch.tensor(batch.get("image_input_idx")).to(device) if batch.get("image_input_idx") is not None else None,
        'pil_image': pil_image  # Keep for viewer
    }

    return cached_data


def extract_visual_features(model, cached_batch, use_n_token_only, visual_layers, device, precision_dtype=torch.float16):
    """
    Extract features from pre-cached batch.
    Returns: dict[visual_layer] -> features [num_patches, hidden_dim], and metadata
    """
    features_by_layer = {}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=precision_dtype):
            images = cached_batch['images'].unsqueeze(0)
            image_masks = cached_batch['image_masks'].unsqueeze(0) if cached_batch['image_masks'] is not None else None

            need_layer_0 = 0 in visual_layers
            need_llm_layers = any(l > 0 for l in visual_layers)

            # Layer 0: vision backbone output
            if need_layer_0:
                feats_l0, _ = model.vision_backbone(images, image_masks, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    feats_l0 = feats_l0[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    feats_l0 = feats_l0[:, :, use_n_token_only, :]

                B, num_chunks, patches_per_chunk, hidden_dim = feats_l0.shape
                features_by_layer[0] = torch.nn.functional.normalize(feats_l0.view(-1, hidden_dim), dim=-1).float()
                del feats_l0

            # LLM layers: one forward pass for all
            if need_llm_layers:
                input_ids = cached_batch['input_tokens'].unsqueeze(0)
                image_input_idx = cached_batch['image_input_idx'].unsqueeze(0) if cached_batch['image_input_idx'] is not None else None

                output = model(
                    input_ids=input_ids,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                hidden_states = output.hidden_states

                B = image_input_idx.shape[0]
                num_chunks = image_input_idx.shape[1]
                patches_per_chunk = image_input_idx.shape[2]
                hidden_dim = hidden_states[0].shape[-1]

                flat_pos = image_input_idx.view(B, -1)
                valid_mask = flat_pos >= 0

                for vl in visual_layers:
                    if vl == 0:
                        continue
                    layer_idx = min(vl, len(hidden_states) - 1)
                    hs = hidden_states[layer_idx]

                    feats = torch.zeros((B, num_chunks, patches_per_chunk, hidden_dim), device=hs.device, dtype=hs.dtype)
                    for b in range(B):
                        valid = flat_pos[b][valid_mask[b]]
                        if valid.numel() > 0:
                            feats.view(B, -1, hidden_dim)[b, valid_mask[b], :] = hs[b, valid.long(), :]

                    features_by_layer[vl] = torch.nn.functional.normalize(feats.view(-1, hidden_dim), dim=-1).float()

                del hidden_states, output

            del images, image_masks
            torch.cuda.empty_cache()

    num_patches = features_by_layer[visual_layers[0]].shape[0]
    metadata = {
        'num_chunks': num_chunks,
        'patches_per_chunk': patches_per_chunk,
        'hidden_dim': hidden_dim,
        'num_patches': num_patches
    }

    return features_by_layer, metadata


def extract_logitlens_predictions(model, cached_batch, use_n_token_only, visual_layers, device, top_k=5, precision_dtype=torch.float16):
    """
    Extract LogitLens predictions by applying ln_f + ff_out to hidden states.
    Returns: dict[visual_layer] -> list of top-k predictions per patch
    """
    logitlens_by_layer = {}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=precision_dtype):
            images = cached_batch['images'].unsqueeze(0)
            image_masks = cached_batch['image_masks'].unsqueeze(0) if cached_batch['image_masks'] is not None else None
            input_ids = cached_batch['input_tokens'].unsqueeze(0)
            image_input_idx = cached_batch['image_input_idx'].unsqueeze(0) if cached_batch['image_input_idx'] is not None else None

            # Forward pass
            output = model(
                input_ids=input_ids,
                images=images,
                image_masks=image_masks,
                image_input_idx=image_input_idx,
                output_hidden_states=True,
                last_logits_only=False,
            )
            hidden_states = output.hidden_states

            B = image_input_idx.shape[0]
            num_chunks = image_input_idx.shape[1]
            patches_per_chunk = image_input_idx.shape[2]
            d_model = hidden_states[0].shape[-1]

            flat_pos = image_input_idx.view(B, -1)
            valid_mask = flat_pos >= 0

            # Get tokenizer for decoding
            tokenizer = model.config.get_tokenizer()

            for vl in visual_layers:
                if vl == 0:
                    # Layer 0 is vision backbone output, can't apply LM head
                    logitlens_by_layer[vl] = None
                    continue

                layer_idx = min(vl, len(hidden_states) - 1)
                hs = hidden_states[layer_idx]

                # Gather visual features
                visual_features = torch.zeros((B, num_chunks, patches_per_chunk, d_model), device=hs.device, dtype=hs.dtype)
                for b in range(B):
                    valid = flat_pos[b][valid_mask[b]]
                    if valid.numel() > 0:
                        visual_features.view(B, -1, d_model)[b, valid_mask[b], :] = hs[b, valid.long(), :]

                # Apply LogitLens: ln_f -> ff_out
                visual_features_normed = model.transformer.ln_f(visual_features)
                logits = model.transformer.ff_out(visual_features_normed)

                if model.config.scale_logits:
                    logits = logits / math.sqrt(model.config.d_model)

                # Get top-k predictions
                topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
                topk_values = topk_values.cpu()
                topk_indices = topk_indices.cpu()

                # Store predictions per patch
                layer_predictions = []
                for chunk_idx in range(num_chunks):
                    for patch_idx in range(patches_per_chunk):
                        patch_preds = []
                        for k in range(top_k):
                            token_id = topk_indices[0, chunk_idx, patch_idx, k].item()
                            logit_val = topk_values[0, chunk_idx, patch_idx, k].item()
                            token_str = tokenizer.decode([token_id])
                            patch_preds.append({
                                'token': token_str,
                                'token_id': token_id,
                                'logit': logit_val
                            })
                        layer_predictions.append(patch_preds)

                logitlens_by_layer[vl] = layer_predictions

            del hidden_states, output
            torch.cuda.empty_cache()

    return logitlens_by_layer, num_chunks, patches_per_chunk


def process_single_model(model_config: dict, cached_image: dict, device, top_k: int = 5, use_bf16: bool = False):
    """Process a single model on the cached image and return results."""

    print(f"\n{'='*70}")
    print(f"PROCESSING: {model_config['display_name']}")
    print(f"{'='*70}")

    ckpt_path = model_config['checkpoint_path']
    contextual_dir = model_config['contextual_dir']
    visual_layers = model_config['layers']

    # Find available contextual layers
    ctx_layers = find_available_layers(contextual_dir)
    print(f"Visual layers: {visual_layers}")
    print(f"Contextual layers: {ctx_layers}")

    # Load model
    print(f"\nLoading model from {ckpt_path}...")
    load_start = time.time()

    cfg = TrainConfig.load(f"{ckpt_path}/config.yaml")

    ckpt_file = f"{ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024**3)

    target_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if ckpt_size_gb < 1.0:
        # Stripped checkpoint: init on CPU, load connector, reset with pretrained base
        cfg.model.init_device = "cpu"
        model = Molmo(cfg.model)
        print(f"  Stripped checkpoint ({ckpt_size_gb:.2f} GB) - loading pretrained...")
        model.reset_with_pretrained_weights()
        weights = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(weights, strict=False)
        del weights
        gc.collect()
        model = model.to(dtype=target_dtype).cuda().eval()
    else:
        # Full checkpoint (~30GB float32): use meta init (0 CommittedAS) + GPU
        # loading (0 CommittedAS). CPU init is not viable: 7B params × 4B = ~28GB
        # CommittedAS, which exceeds system headroom (vm.overcommit_memory=2).
        # Must use bfloat16 here — fp16 + assign=True produces NaN in forward pass
        # (root cause: likely softmax overflow; bf16's wider exponent avoids it).
        cfg.model.init_device = "meta"
        model = Molmo(cfg.model)
        print(f"  Loading {ckpt_size_gb:.1f} GB to GPU (meta init, bfloat16)...")
        weights = torch.load(ckpt_file, map_location=device)
        for k in list(weights.keys()):
            if weights[k].is_floating_point():
                weights[k] = weights[k].to(torch.bfloat16)
        torch.cuda.empty_cache()
        model.load_state_dict(weights, strict=False, assign=True)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        model = model.eval()

    torch.cuda.empty_cache()

    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")

    # Inference dtype must match model weight dtype to avoid fp16 overflow/NaN.
    # (e.g. bf16 weights + fp16 autocast → NaN in attention softmax)
    precision_dtype = next(model.parameters()).dtype

    # Get model config for preprocessing
    model_cfg = ModelConfig.load(resource_path(ckpt_path, "config.yaml"), key="model", validate_paths=False)
    use_n_token_only = model_cfg.vision_backbone.use_n_token_only

    # Extract LogitLens predictions (one forward pass for all layers)
    print(f"\n  Extracting LogitLens predictions...")
    logitlens_by_layer, ll_num_chunks, ll_patches_per_chunk = extract_logitlens_predictions(
        model, cached_image, use_n_token_only, visual_layers, device, top_k, precision_dtype
    )
    print(f"  ✓ LogitLens done")

    # Storage for candidates
    candidates = {vl: {} for vl in visual_layers}
    shape_info = None
    ctx_metadata_cache = {}

    # Extract visual features ONCE (avoid redundant forward passes per contextual layer)
    features_by_layer, shape_info = extract_visual_features(
        model, cached_image, use_n_token_only, visual_layers, device, precision_dtype
    )

    # Free model from GPU before loading contextual caches
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Process each contextual cache
    for ctx_idx, ctx_layer in enumerate(ctx_layers):
        print(f"\n  Contextual layer {ctx_layer} ({ctx_idx + 1}/{len(ctx_layers)})...", end=" ", flush=True)

        # Load cache to GPU (model already freed above)
        cache_file = Path(contextual_dir) / f"layer_{ctx_layer}" / "embeddings_cache.pt"
        cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
        embeddings = cache_data['embeddings'].to(device)
        metadata = cache_data['metadata']
        torch.nn.functional.normalize(embeddings, dim=-1, out=embeddings)
        ctx_metadata_cache[ctx_layer] = metadata
        del cache_data

        # Search on GPU
        for vl in visual_layers:
            feats = features_by_layer[vl]
            similarity = torch.matmul(feats, embeddings.T)
            top_vals, top_idxs = torch.topk(similarity, k=top_k, dim=-1)
            candidates[vl][ctx_layer] = (top_vals.cpu(), top_idxs.cpu())
            del similarity

        del embeddings
        gc.collect()
        torch.cuda.empty_cache()
        print("done")

    del features_by_layer
    gc.collect()
    torch.cuda.empty_cache()

    # Build results
    print(f"\n  Building results...")
    num_patches = shape_info['num_patches']
    num_chunks = shape_info['num_chunks']
    patches_per_chunk = shape_info['patches_per_chunk']
    hidden_dim = shape_info['hidden_dim']

    results = {}

    for vl in visual_layers:
        all_vals = torch.stack([candidates[vl][cl][0] for cl in ctx_layers])
        all_idxs = torch.stack([candidates[vl][cl][1] for cl in ctx_layers])

        chunks_results = []
        for chunk_idx in range(num_chunks):
            chunk_results = {
                "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                "patches": []
            }

            for local_patch_idx in range(patches_per_chunk):
                global_patch_idx = chunk_idx * patches_per_chunk + local_patch_idx

                patch_vals = all_vals[:, global_patch_idx, :]
                patch_idxs = all_idxs[:, global_patch_idx, :]

                flat_vals = patch_vals.flatten()
                flat_idxs = patch_idxs.flatten()
                ctx_ids = torch.arange(len(ctx_layers)).unsqueeze(1).expand(-1, top_k).flatten()

                global_top_vals, global_top_pos = torch.topk(flat_vals, k=top_k)

                nearest = []
                for k_idx in range(top_k):
                    pos = global_top_pos[k_idx].item()
                    sim = global_top_vals[k_idx].item()
                    ctx_idx = ctx_ids[pos].item()
                    emb_idx = flat_idxs[pos].item()

                    ctx_layer = ctx_layers[ctx_idx]
                    meta = ctx_metadata_cache[ctx_layer][emb_idx]
                    nearest.append({
                        'token_str': meta['token_str'],
                        'token_id': meta['token_id'],
                        'caption': meta['caption'],
                        'position': meta['position'],
                        'similarity': sim,
                        'contextual_layer': ctx_layer
                    })

                row, col = patch_idx_to_row_col(local_patch_idx, patches_per_chunk)
                chunk_results["patches"].append({
                    "patch_idx": local_patch_idx,
                    "patch_row": row,
                    "patch_col": col,
                    "nearest_contextual_neighbors": nearest
                })

            chunks_results.append(chunk_results)

        results[vl] = {
            "llm_layer_used": vl,
            "chunks": chunks_results
        }

    return {
        "model_spec": model_config,
        "visual_layers": visual_layers,
        "contextual_layers_used": ctx_layers,
        "patches_per_chunk": patches_per_chunk,
        "num_chunks": num_chunks,
        "results_by_layer": results,
        "logitlens_by_layer": logitlens_by_layer
    }


def main():
    parser = argparse.ArgumentParser(description="Single-image LatentLens extraction")
    parser.add_argument("--image-path", type=str, required=True,
                       help="Path to input image file")
    parser.add_argument("--models", type=str, required=True,
                       help="Comma-separated list of models (e.g., 'olmo-7b_vit-l-14-336,llama3-8b_siglip')")
    parser.add_argument("--output-dir", type=str, default="analysis_results/single_image_demo",
                       help="Output directory for results")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of nearest neighbors to retrieve")
    parser.add_argument("--use-bf16", action="store_true",
                       help="Use bfloat16 instead of float16")
    parser.add_argument("--generate-viewer", action="store_true",
                       help="Generate HTML viewer after extraction")
    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Parse model specs
    model_specs = [spec.strip() for spec in args.models.split(",")]
    model_configs = [parse_model_spec(spec) for spec in model_specs]

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SINGLE-IMAGE LATENTLENS")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Models: {[c['display_name'] for c in model_configs]}")
    print(f"Output: {output_dir}")
    print()

    # For the first model, load and preprocess image (we need its preprocessor)
    # Actually, we'll do this per-model since preprocessing depends on vision encoder

    all_results = {}
    pil_image = None

    for model_config in model_configs:
        # Create preprocessor for this model
        ckpt_path = model_config['checkpoint_path']
        model_cfg = ModelConfig.load(resource_path(ckpt_path, "config.yaml"), key="model", validate_paths=False)
        model_cfg.system_prompt_kind = "none"
        preprocessor = build_mm_preprocessor(model_cfg, for_inference=True, shuffle_messages=False,
                                             is_training=False, require_image_features=True)

        # Load and preprocess image
        cached_image = load_and_preprocess_image(str(image_path), preprocessor, device)
        if pil_image is None:
            pil_image = cached_image['pil_image']

        # Process this model
        model_results = process_single_model(model_config, cached_image, device,
                                            top_k=args.top_k, use_bf16=args.use_bf16)

        all_results[model_config['display_name']] = model_results

        # Cleanup
        del cached_image
        gc.collect()
        torch.cuda.empty_cache()

    # Save results JSON
    results_file = output_dir / "latentlens_results.json"

    # Convert to serializable format
    serializable_results = {}
    for model_name, result in all_results.items():
        serializable_results[model_name] = {
            "model_spec": {
                "llm": result["model_spec"]["llm"],
                "vision_encoder": result["model_spec"]["vision_encoder"],
                "display_name": result["model_spec"]["display_name"]
            },
            "visual_layers": result["visual_layers"],
            "contextual_layers_used": result["contextual_layers_used"],
            "patches_per_chunk": result["patches_per_chunk"],
            "num_chunks": result["num_chunks"],
            "results_by_layer": result["results_by_layer"],
            "logitlens_by_layer": result["logitlens_by_layer"]
        }

    with open(results_file, 'w') as f:
        json.dump({
            "image_path": str(image_path.absolute()),
            "models": serializable_results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {results_file}")

    # Save image copy for viewer
    if pil_image:
        image_copy_path = output_dir / "image.jpg"
        pil_image.save(image_copy_path, quality=95)
        print(f"✓ Image saved to: {image_copy_path}")

    # Generate viewer if requested
    if args.generate_viewer:
        print(f"\nGenerating HTML viewer...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from single_image_viewer import generate_viewer
        viewer_path = generate_viewer(output_dir)
        print(f"✓ Viewer saved to: {viewer_path}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
