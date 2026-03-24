#!/usr/bin/env python3
"""
EmbeddingLens (Static Nearest Neighbor) Analysis for Molmo-7B-D Vision Tokens.

Finds the nearest vocabulary embeddings for vision tokens at each LLM layer.
Uses the BASE CROP (12x12=144 tokens) for spatial analysis.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/molmo_7b/nearest_neighbors.py \
        --num-images 100 --layers "0,1,2,4,8,16,24,26,27"
"""

import argparse
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor

from preprocessing import (
    preprocess_image_molmo, get_base_crop_grid, get_base_crop_token_positions,
    validate_base_crop, TOKENS_PER_CROP, TOKENS_PER_CROP_H, TOKENS_PER_CROP_W,
    MODEL_NAME,
)

try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available.")


def decode_token(tokenizer, idx):
    token = tokenizer.decode([int(idx)])
    return token.encode('utf-8').decode('utf-8')


def prepare_molmo_inputs(processor, image, prompt, device):
    """
    Prepare Molmo inputs and return processed tensors + base crop positions.

    Molmo's processor returns image_input_idx which maps vision tokens to
    sequence positions. We extract base crop (crop 0) positions for analysis.
    """
    inputs = processor.process(images=[image], text=prompt)

    # Get base crop token positions before batching
    image_input_idx = inputs['image_input_idx']  # (num_crops, 144)
    validate_base_crop(image_input_idx)
    base_positions = get_base_crop_token_positions(image_input_idx)

    # Convert to batched tensors for model
    # CRITICAL: cast floating point tensors to float16 to match model dtype
    batch = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                v = v.to(torch.float16)
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v

    return batch, base_positions


def main():
    parser = argparse.ArgumentParser(description="EmbeddingLens for Molmo-7B-D")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output-dir", type=str,
                       default="analysis_results/nearest_neighbors/molmo_7b")
    parser.add_argument("--layers", type=str, default="0,1,2,4,8,16,24,26,27")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    layers_to_analyze = [int(l.strip()) for l in args.layers.split(",")]

    grid_h, grid_w, num_tokens = get_base_crop_grid()

    print("=" * 70)
    print("MOLMO-7B-D EMBEDDINGLENS ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Images: {args.num_images}, Layers: {layers_to_analyze}, Top-K: {args.top_k}")
    print(f"Grid: {grid_h}x{grid_w} = {num_tokens} base crop tokens")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Get vocabulary embeddings
    # Molmo: model.model.transformer.wte.embedding is a Parameter (not Embedding.weight)
    embed_weight = model.model.transformer.wte.embedding  # [152064, 3584]
    vocab_embeddings = F.normalize(embed_weight.float(), dim=-1).to(device)
    print(f"Vocabulary embeddings: {vocab_embeddings.shape}")

    if not HAVE_PIXMOCAP:
        print("ERROR: PixMoCap dataset required")
        return
    dataset = PixMoCap(split=args.split, mode="captions")

    print(f"\nProcessing {args.num_images} images...")
    all_results = []
    prompt = "Describe this image in detail."

    for img_idx in tqdm(range(args.num_images)):
        example = dataset.get(img_idx, np.random)
        image = Image.open(example["image"]).convert('RGB')

        caption = ""
        if "message_list" in example and len(example["message_list"]) > 0:
            caption = example["message_list"][0].get("text", "")

        # Process through Molmo's processor (handles multi-crop internally)
        batch, base_positions = prepare_molmo_inputs(processor, image, prompt, device)

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = model(**batch, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        # Extract vision tokens at base crop positions
        # base_positions is a list of (seq_pos, row, col)
        seq_positions = [p[0] for p in base_positions]

        layers_data = {}
        for layer_idx in layers_to_analyze:
            if layer_idx >= len(hidden_states):
                continue

            hs = hidden_states[layer_idx].squeeze(0)  # [seq_len, hidden_dim]

            # Gather base crop vision tokens by their sequence positions
            pos_tensor = torch.tensor(seq_positions, device=hs.device)
            vision_hs = hs[pos_tensor]  # [144, 3584]
            vision_hs_normed = F.normalize(vision_hs.float(), dim=-1)

            # Cosine similarity NN search
            sims = torch.mm(vision_hs_normed, vocab_embeddings.T)
            topk_sims, topk_ids = torch.topk(sims, k=args.top_k, dim=-1)
            topk_sims = topk_sims.detach().cpu().numpy()
            topk_ids = topk_ids.detach().cpu().numpy()

            patches = []
            for i, (seq_pos, row, col) in enumerate(base_positions):
                neighbors = []
                for sim, tid in zip(topk_sims[i], topk_ids[i]):
                    neighbors.append({
                        "token": decode_token(processor.tokenizer, tid),
                        "token_id": int(tid),
                        "similarity": float(sim)
                    })
                patches.append({
                    "patch_idx": i,
                    "patch_row": row,
                    "patch_col": col,
                    "nearest_neighbors": neighbors
                })
            layers_data[str(layer_idx)] = patches

        all_results.append({
            "image_idx": img_idx,
            "ground_truth_caption": caption,
            "num_vision_tokens": len(base_positions),
            "grid_info": {"height": grid_h, "width": grid_w},
            "layers": layers_data,
        })

        del hidden_states, outputs
        gc.collect()
        torch.cuda.empty_cache()

    # Save per-layer files
    output_dir = Path(args.output_dir) / MODEL_NAME.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in layers_to_analyze:
        layer_results = {
            "model_name": MODEL_NAME,
            "layer": layer_idx,
            "num_images": len(all_results),
            "top_k": args.top_k,
            "results": []
        }
        for img_result in all_results:
            lk = str(layer_idx)
            if lk in img_result["layers"]:
                layer_results["results"].append({
                    "image_idx": img_result["image_idx"],
                    "ground_truth_caption": img_result["ground_truth_caption"],
                    "num_vision_tokens": img_result["num_vision_tokens"],
                    "grid_info": img_result["grid_info"],
                    "patches": img_result["layers"][lk]
                })
        out_file = output_dir / f"nearest_neighbors_layer{layer_idx}_topk{args.top_k}.json"
        with open(out_file, 'w') as f:
            json.dump(layer_results, f, indent=2)
        print(f"  Saved layer {layer_idx}: {out_file}")

    print(f"\nDONE! Results: {output_dir}")


if __name__ == "__main__":
    main()
