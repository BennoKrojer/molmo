#!/usr/bin/env python3
"""
LogitLens Analysis for Molmo-72B Vision Tokens.

Applies the LM head (ln_f + ff_out) to intermediate hidden states.
Uses the BASE CROP (12x12=144 tokens) for spatial analysis.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/molmo_72b/logitlens.py \
        --num-images 100 --layers "0,1,2,4,8,16,24,26,27"
"""

import argparse
import gc
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor

from preprocessing import (
    get_base_crop_grid, TOKENS_PER_CROP_H, TOKENS_PER_CROP_W, MODEL_NAME,
)
from nearest_neighbors import prepare_molmo_inputs, decode_token

try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available.")


def main():
    parser = argparse.ArgumentParser(description="LogitLens for Molmo-72B")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output-dir", type=str,
                       default="analysis_results/logit_lens/molmo_72b")
    parser.add_argument("--layers", type=str, default="0,1,2,4,8,16,40,60,72,78,79")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    layers_to_analyze = [int(l.strip()) for l in args.layers.split(",")]
    grid_h, grid_w, num_tokens = get_base_crop_grid()

    print("=" * 70)
    print("MOLMO-7B-D LOGITLENS ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Images: {args.num_images}, Layers: {layers_to_analyze}, Top-K: {args.top_k}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Get LM head and final norm
    # Molmo: model.model.transformer.ff_out (Linear) and .ln_f (RMSLayerNorm)
    ff_out = model.model.transformer.ff_out  # Linear(3584, 152064)
    ln_f = model.model.transformer.ln_f  # RMSLayerNorm

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

        batch, base_positions = prepare_molmo_inputs(processor, image, prompt, device)

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = model(**batch, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        seq_positions = [p[0] for p in base_positions]

        layers_data = {}
        for layer_idx in layers_to_analyze:
            if layer_idx >= len(hidden_states):
                continue

            hs = hidden_states[layer_idx].squeeze(0)
            pos_tensor = torch.tensor(seq_positions, device=hs.device)
            vision_hs = hs[pos_tensor]  # [144, 3584]

            # Apply layer norm then LM head (ff_out)
            vision_hs_normed = ln_f(vision_hs)
            logits = ff_out(vision_hs_normed.to(ff_out.weight.dtype)).float()

            topk_vals, topk_ids = torch.topk(logits, k=args.top_k, dim=-1)
            topk_vals = topk_vals.detach().cpu().numpy()
            topk_ids = topk_ids.detach().cpu().numpy()

            patches = []
            for i, (seq_pos, row, col) in enumerate(base_positions):
                preds = []
                for val, tid in zip(topk_vals[i], topk_ids[i]):
                    preds.append({
                        "token": decode_token(processor.tokenizer, tid),
                        "token_id": int(tid),
                        "logit": float(val)
                    })
                patches.append({
                    "patch_idx": i,
                    "patch_row": row,
                    "patch_col": col,
                    "top_predictions": preds
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
        out_file = output_dir / f"logit_lens_layer{layer_idx}_topk{args.top_k}.json"
        with open(out_file, 'w') as f:
            json.dump(layer_results, f, indent=2)
        print(f"  Saved layer {layer_idx}: {out_file}")

    print(f"\nDONE! Results: {output_dir}")


if __name__ == "__main__":
    main()
