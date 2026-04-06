#!/usr/bin/env python3
"""
LogitLens Analysis for LLaVA-NeXT-34B Vision Tokens.

Applies the LM head (ln_f + lm_head) to intermediate hidden states.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/llava_next/logitlens.py \
        --num-images 100 --layers "0,1,2,4,8,16,24,30,31"
"""

import argparse
import gc
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import LlavaNextForConditionalGeneration, AutoProcessor

from preprocessing import (
    preprocess_image_llava, NUM_VISION_TOKENS, GRID_H, GRID_W,
    IMAGE_TOKEN_ID, MODEL_NAME,
)
from nearest_neighbors import find_vision_token_range, decode_token

try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False
    print("Warning: PixMoCap dataset not available.")


def main():
    parser = argparse.ArgumentParser(description="LogitLens for LLaVA-NeXT-34B")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output-dir", type=str,
                       default="analysis_results/logit_lens/llava_next")
    parser.add_argument("--layers", type=str, default="0,1,2,4,8,16,30,45,58,59")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    layers_to_analyze = [int(l.strip()) for l in args.layers.split(",")]

    print("=" * 70)
    print("LLAVA-1.5 LOGITLENS ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Images: {args.num_images}, Layers: {layers_to_analyze}, Top-K: {args.top_k}")
    print()

    # Load model
    print("Loading model...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Get LM head and final norm
    lm_head = model.lm_head  # Linear(4096, 32064)
    ln_f = model.model.language_model.norm  # LlamaRMSNorm

    if not HAVE_PIXMOCAP:
        print("ERROR: PixMoCap dataset required")
        return
    dataset = PixMoCap(split=args.split, mode="captions")

    print(f"\nProcessing {args.num_images} images...")
    all_results = []
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"

    for img_idx in tqdm(range(args.num_images)):
        example = dataset.get(img_idx, np.random)
        image = Image.open(example["image"]).convert('RGB')
        image = preprocess_image_llava(image, target_size=336, force_square=True)

        caption = ""
        if "message_list" in example and len(example["message_list"]) > 0:
            caption = example["message_list"][0].get("text", "")

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        vision_start, vision_end, num_vision = find_vision_token_range(
            inputs['input_ids']
        )

        if num_vision == 0:
            continue

        layers_data = {}
        for layer_idx in layers_to_analyze:
            if layer_idx >= len(hidden_states):
                continue

            hs = hidden_states[layer_idx]
            vision_hs = hs[:, vision_start:vision_end, :].squeeze(0)  # [576, 4096]

            # Apply layer norm then LM head
            vision_hs_normed = ln_f(vision_hs)
            logits = lm_head(vision_hs_normed.to(lm_head.weight.dtype)).float()

            topk_vals, topk_ids = torch.topk(logits, k=args.top_k, dim=-1)
            topk_vals = topk_vals.detach().cpu().numpy()
            topk_ids = topk_ids.detach().cpu().numpy()

            patches = []
            for patch_idx in range(num_vision):
                row = patch_idx // GRID_W
                col = patch_idx % GRID_W
                preds = []
                for val, tid in zip(topk_vals[patch_idx], topk_ids[patch_idx]):
                    preds.append({
                        "token": decode_token(processor.tokenizer, tid),
                        "token_id": int(tid),
                        "logit": float(val)
                    })
                patches.append({
                    "patch_idx": patch_idx,
                    "patch_row": row,
                    "patch_col": col,
                    "top_predictions": preds
                })
            layers_data[str(layer_idx)] = patches

        all_results.append({
            "image_idx": img_idx,
            "ground_truth_caption": caption,
            "num_vision_tokens": num_vision,
            "grid_info": {"height": GRID_H, "width": GRID_W},
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
