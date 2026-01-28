#!/usr/bin/env python3
"""
Compute cosine similarity of TEXT tokens across layers to layer 0 for Qwen2-VL.

This script traces text tokens through Qwen2-VL LLM layers and computes the cosine similarity
between each text token at layer N and the same token at layer 0 (raw embedding lookup).

This is the Qwen2-VL equivalent of scripts/analysis/sameToken_acrossLayers_text_similarity.py

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/sameToken_acrossLayers_text_similarity.py \
        --num-images 100 --output-dir analysis_results/sameToken_acrossLayers_text_similarity/qwen2_vl
"""

import argparse
import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Import dataset (optional)
try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available.")


# Qwen2-VL special token IDs
IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> token ID in Qwen2-VL
VISION_START_TOKEN_ID = 151652  # <|vision_start|>
VISION_END_TOKEN_ID = 151653  # <|vision_end|>
VISION_PAD_TOKEN_ID = 151654  # <|vision_pad|>


def get_special_token_ids():
    """Get special token IDs to exclude from text analysis."""
    return {
        IMAGE_PAD_TOKEN_ID,
        VISION_START_TOKEN_ID,
        VISION_END_TOKEN_ID,
        VISION_PAD_TOKEN_ID,
    }


def find_text_token_positions(input_ids):
    """
    Find positions of text tokens (non-vision, non-special) in the input sequence.

    Returns:
        text_positions: list of positions that are actual text tokens
        num_text_tokens: count of text tokens
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()

    if len(input_ids.shape) == 2:
        input_ids = input_ids[0]

    special_ids = get_special_token_ids()

    text_positions = []
    for pos, token_id in enumerate(input_ids):
        if token_id not in special_ids:
            text_positions.append(pos)

    return text_positions, len(text_positions)


def compute_text_similarity_to_layer0(model, processor, image, caption, device, layers_to_analyze=None):
    """
    Compute cosine similarity of text tokens at each layer to their layer 0 version.

    Returns:
        dict with:
        - 'layer_similarities': dict[layer_idx] -> {"same_token": {"mean": x}, "baseline": {"mean": y}}
        - 'metadata': dict with shape info
    """
    # Prepare inputs with image and caption
    # Format: <|vision_start|><|image_pad|>...<|vision_end|>caption_text
    text_with_image = f"<|image_pad|>{caption}"

    model_inputs = processor(
        images=[image],
        text=text_with_image,
        return_tensors="pt"
    )
    model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in model_inputs.items()}

    # Find text token positions
    input_ids = model_inputs['input_ids']
    text_positions, num_text_tokens = find_text_token_positions(input_ids)

    if num_text_tokens == 0:
        raise ValueError("No text tokens found in input sequence")

    # Convert to tensor for indexing
    text_positions_tensor = torch.tensor(text_positions, device=device)

    # Run forward pass
    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            outputs = model(**model_inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)

    # Extract and normalize layer 0 text tokens
    layer0_features = hidden_states[0][0, text_positions_tensor, :].float()  # [num_text, hidden_dim]
    layer0_features_norm = torch.nn.functional.normalize(layer0_features, dim=-1)

    # Determine which layers to analyze
    if layers_to_analyze is None:
        layers_to_analyze = list(range(1, num_layers))

    layer_similarities = {}

    for layer_idx in layers_to_analyze:
        if layer_idx >= num_layers:
            continue

        # Extract and normalize text tokens at this layer
        layerN_features = hidden_states[layer_idx][0, text_positions_tensor, :].float()
        layerN_features_norm = torch.nn.functional.normalize(layerN_features, dim=-1)

        # SAME TOKEN: Compute cosine similarity between same-position tokens
        similarity_same = (layer0_features_norm * layerN_features_norm).sum(dim=-1)  # [num_text]

        # BASELINE: Compare to shuffled tokens (different positions)
        shuffled_indices = torch.randperm(num_text_tokens, device=device)
        layerN_shuffled = layerN_features_norm[shuffled_indices, :]
        similarity_baseline = (layer0_features_norm * layerN_shuffled).sum(dim=-1)

        layer_similarities[int(layer_idx)] = {
            "same_token": {
                "mean": float(similarity_same.mean().item()),
                "std": float(similarity_same.std().item()),
                "min": float(similarity_same.min().item()),
                "max": float(similarity_same.max().item()),
            },
            "baseline_different_token": {
                "mean": float(similarity_baseline.mean().item()),
                "std": float(similarity_baseline.std().item()),
                "min": float(similarity_baseline.min().item()),
                "max": float(similarity_baseline.max().item()),
            },
            "num_text_tokens": num_text_tokens
        }

        del layerN_features, layerN_features_norm, similarity_same, similarity_baseline

    metadata = {
        'num_text_tokens': num_text_tokens,
        'hidden_dim': int(hidden_states[0].shape[-1]),
        'num_layers': num_layers,
        'total_seq_len': int(input_ids.shape[1])
    }

    # Clean up
    del hidden_states, outputs, layer0_features, layer0_features_norm
    torch.cuda.empty_cache()

    return {
        'layer_similarities': layer_similarities,
        'metadata': metadata
    }


def preload_images(dataset, num_images, max_size=1024):
    """Preload and preprocess images with their captions."""
    cached_data = []

    for img_idx in range(num_images):
        if dataset is not None:
            example = dataset.get(img_idx, np.random)
            image = Image.open(example["image"]).convert('RGB')
            caption = ""
            if "message_list" in example and len(example["message_list"]) > 0:
                caption = example["message_list"][0].get("text", "")
        else:
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
            color = colors[img_idx % len(colors)]
            image = Image.new('RGB', (224, 224), color=color)
            caption = f"A solid {color} image with some text description."

        # Resize if too large
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        cached_data.append({'image': image, 'caption': caption})

    return cached_data


def main():
    parser = argparse.ArgumentParser(description="Compute text token similarity across layers for Qwen2-VL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output-dir", type=str,
                        default="analysis_results/sameToken_acrossLayers_text_similarity/qwen2_vl")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layers to analyze (default: all layers 1-27)")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Parse layers (default: all 27 layers for 28-layer Qwen2-VL)
    if args.layers:
        layers_to_analyze = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers_to_analyze = list(range(1, 28))  # All layers 1-27 (28 total including layer 0)

    print("=" * 70)
    print("QWEN2-VL TEXT TOKEN SIMILARITY ACROSS LAYERS")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Images: {args.num_images}")
    print(f"Layers: {layers_to_analyze}")
    print()

    # Load model
    print("Loading Qwen2-VL model...")
    load_start = time.time()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Model loaded in {time.time() - load_start:.1f}s")
    print()

    # Load dataset
    if HAVE_PIXMOCAP:
        dataset = PixMoCap(split=args.split, mode="captions")
    else:
        dataset = None

    # Preload images
    print(f"Preloading {args.num_images} images...")
    cached_images = preload_images(dataset, args.num_images)
    print()

    # Process images
    print("Computing text token similarities...")
    process_start = time.time()

    all_results = []

    # Accumulators for global averages
    layer_similarities_sum = {l: {"same": 0.0, "baseline": 0.0} for l in layers_to_analyze}
    layer_similarities_count = {l: 0 for l in layers_to_analyze}

    for img_idx in tqdm(range(args.num_images), desc="Processing"):
        image_data = cached_images[img_idx]

        result = compute_text_similarity_to_layer0(
            model, processor, image_data['image'], image_data['caption'], device, layers_to_analyze
        )

        # Store per-image results
        image_result = {
            'image_idx': img_idx,
            'layer_similarities': result['layer_similarities'],
            'metadata': result['metadata']
        }
        all_results.append(image_result)

        # Accumulate for global averages
        for layer_idx, layer_data in result['layer_similarities'].items():
            num_tokens = layer_data['num_text_tokens']
            layer_similarities_sum[layer_idx]["same"] += layer_data['same_token']['mean'] * num_tokens
            layer_similarities_sum[layer_idx]["baseline"] += layer_data['baseline_different_token']['mean'] * num_tokens
            layer_similarities_count[layer_idx] += num_tokens

        gc.collect()
        torch.cuda.empty_cache()

    process_time = time.time() - process_start
    print(f"Processed {args.num_images} images in {process_time:.1f}s")
    print()

    # Compute global averages
    global_similarities = {}
    for layer_idx in layers_to_analyze:
        total_count = layer_similarities_count[layer_idx]
        if total_count > 0:
            global_similarities[layer_idx] = {
                "same_token": {
                    "mean_similarity": layer_similarities_sum[layer_idx]["same"] / total_count,
                },
                "baseline_different_token": {
                    "mean_similarity": layer_similarities_sum[layer_idx]["baseline"] / total_count,
                },
                "total_text_tokens": total_count
            }

    # Save results
    output_dir = Path(args.output_dir)
    model_name_safe = args.model_name.replace("/", "_")
    output_dir = output_dir / model_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_file = output_dir / "text_similarity_across_layers_summary.json"
    summary_data = {
        'model_name': args.model_name,
        'split': args.split,
        'num_images': args.num_images,
        'layers_analyzed': layers_to_analyze,
        'global_averages': global_similarities,
        'processing_time_seconds': process_time
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary: {summary_file}")

    # Save detailed results
    detailed_file = output_dir / "text_similarity_across_layers_detailed.json"
    detailed_data = {
        'model_name': args.model_name,
        'split': args.split,
        'num_images': args.num_images,
        'per_image_results': all_results
    }

    with open(detailed_file, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    print(f"Saved detailed results: {detailed_file}")

    # Print summary
    print()
    print("=" * 70)
    print("Summary: Average Cosine Similarity of Text Tokens to Layer 0")
    print("=" * 70)
    print(f"{'Layer':<10} {'Same Token':<20} {'Different Token':<20}")
    print("-" * 50)
    for layer_idx in sorted(global_similarities.keys()):
        same = global_similarities[layer_idx]['same_token']['mean_similarity']
        baseline = global_similarities[layer_idx]['baseline_different_token']['mean_similarity']
        print(f"{layer_idx:<10} {same:<20.4f} {baseline:<20.4f}")
    print("=" * 70)
    print()
    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
