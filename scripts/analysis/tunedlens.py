#!/usr/bin/env python3
"""
Tuned Lens Inference for Visual Tokens

Applies trained per-layer affine probes before the LM head, producing
vocabulary predictions that are better calibrated than raw LogitLens.

    LogitLens:  tokens = top-k(lm_head(ln_f(h_l)))
    TunedLens:  tokens = top-k(lm_head(ln_f(T_l(h_l))))

Output format is IDENTICAL to logitlens.py, so the existing LLM judge
pipeline (run_single_model_with_viz_logitlens.py) works unchanged with:
    --base-dir analysis_results/tuned_lens

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/tunedlens.py \\
        --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_siglip/step12000-unsharded \\
        --probes-path analysis_results/tunedlens_probes/train_mlp-only_pixmo_cap_resize_llama3-8b_siglip/probes.pt \\
        --layers "0,1,2,4,8,16,24,30,31" \\
        --num-images 100 \\
        --output-dir analysis_results/tuned_lens
"""

import argparse
import gc
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


class TunedLensProbe(nn.Module):
    """Per-layer affine probe (must match train_tunedlens.py definition)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def load_probes(probes_path: str, device: torch.device):
    """Load saved probes and return dict {layer_idx: probe}."""
    state = torch.load(probes_path, map_location="cpu")
    d_model = state["d_model"]
    layer_indices = state["layer_indices"]

    probes = {}
    for l in layer_indices:
        probe = TunedLensProbe(d_model)
        probe.load_state_dict(state["probes"][l])
        probe = probe.half().to(device).eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        probes[l] = probe

    print(f"  Loaded probes for layers {layer_indices}  (d_model={d_model})")
    if "training_config" in state:
        tc = state["training_config"]
        print(f"  Trained on {tc.get('num_train_images')} images × {tc.get('epochs')} epochs,"
              f" final_loss={tc.get('final_loss', '?'):.4f}")
    return probes, d_model, layer_indices


def patch_idx_to_row_col(patch_idx: int, patches_per_chunk: int):
    grid_size = int(math.sqrt(patches_per_chunk))
    return patch_idx // grid_size, patch_idx % grid_size


def decode_token(tokenizer, idx: int) -> str:
    token = tokenizer.decode([int(idx)])
    return token.encode("utf-8").decode("utf-8")


def apply_lm_head(model: Molmo, h: torch.Tensor) -> torch.Tensor:
    """Apply ln_f + ff_out (+ optional logit scale) — identical to train_tunedlens.py."""
    normed = model.transformer.ln_f(h)
    logits = model.transformer.ff_out(normed)
    if model.config.scale_logits:
        logits = logits / math.sqrt(model.config.d_model)
    return logits


def main():
    parser = argparse.ArgumentParser(
        description="Tuned Lens inference for visual tokens (same output as logitlens.py)"
    )
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model checkpoint directory (step12000-unsharded)")
    parser.add_argument("--probes-path", type=str, required=True,
                        help="Path to probes.pt saved by train_tunedlens.py")
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layer indices to run inference for")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of validation images to process (default: 100)")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation"],
                        help="Dataset split (default: validation)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions per patch (default: 5)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/tuned_lens",
                        help="Output directory for JSON results")
    args = parser.parse_args()

    analyzed_layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda")

    print("=" * 70)
    print("TUNED LENS INFERENCE")
    print("=" * 70)
    print(f"Checkpoint:   {args.ckpt_path}")
    print(f"Probes:       {args.probes_path}")
    print(f"Layers:       {analyzed_layers}")
    print(f"Images:       {args.num_images} ({args.split})")
    print(f"Top-k:        {args.top_k}")
    print()

    # ===== LOAD MODEL =====
    print("Loading model...")
    t0 = time.time()

    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)

    ckpt_file = f"{args.ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024 ** 3)
    if ckpt_size_gb < 1.0:
        print(f"  Stripped checkpoint ({ckpt_size_gb:.2f} GB) — loading pretrained weights first...")
        model.reset_with_pretrained_weights()

    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    model = model.half().cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    print(f"✓ Model loaded in {time.time() - t0:.1f}s")

    # ===== LOAD PROBES =====
    print("Loading probes...")
    probes, d_model, probe_layers = load_probes(args.probes_path, device)

    # Warn if requested layers have no probe (fall back to LogitLens for those)
    missing = [l for l in analyzed_layers if l not in probes]
    if missing:
        print(f"  WARNING: no probe for layers {missing} — LogitLens will be used as fallback")

    # ===== LOAD PREPROCESSOR + DATASET =====
    model_config = ModelConfig.load(
        resource_path(args.ckpt_path, "config.yaml"), key="model", validate_paths=False
    )
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True,
    )
    use_n_token_only = model_config.vision_backbone.use_n_token_only

    dataset = PixMoCap(split=args.split, mode="captions")
    prompt = "Describe this image in detail."
    print(f"  Dataset: PixMoCap {args.split} split")
    print()

    # ===== INFERENCE =====
    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    # Collect per-image results (each image has all layers)
    all_image_results = []  # list of {image_idx, caption, layers: [{layer_idx, chunks}]}

    for i in tqdm(range(args.num_images), desc="Images"):
        example_data = dataset.get(i, np.random)
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            caption_text = example_data["message_list"][0].get("text", "")

        example = {"image": example_data["image"], "messages": [prompt]}
        batch = preprocessor(example, rng=np.random)

        input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
        images_tensor = torch.tensor(batch["images"]).unsqueeze(0).to(device)
        image_masks = (
            torch.tensor(batch["image_masks"]).unsqueeze(0).to(device)
            if batch.get("image_masks") is not None else None
        )
        image_input_idx = (
            torch.tensor(batch["image_input_idx"]).unsqueeze(0).to(device)
            if batch.get("image_input_idx") is not None else None
        )

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
            hidden_states = output.hidden_states

            # Grid layout
            B = 1
            num_chunks = image_input_idx.shape[1]
            patches_per_chunk = image_input_idx.shape[2]
            flat_pos = image_input_idx.view(B, -1)
            valid_mask = flat_pos >= 0

            image_result = {
                "image_idx": i,
                "ground_truth_caption": caption_text,
                "layers": [],
            }

            for layer_idx in analyzed_layers:
                if layer_idx >= len(hidden_states):
                    continue

                hs = hidden_states[layer_idx]  # [1, seq_len, d_model], float16

                # Gather visual token features → [1, num_chunks, patches_per_chunk, d_model]
                visual_features = torch.zeros(
                    (B, num_chunks, patches_per_chunk, d_model),
                    device=device, dtype=torch.float16,
                )
                for b in range(B):
                    valid_pos = flat_pos[b][valid_mask[b]]
                    if valid_pos.numel() > 0:
                        gathered = hs[b, valid_pos.long(), :]
                        visual_features.view(B, -1, d_model)[b, valid_mask[b], :] = gathered

                # Apply probe (if available) then LM head
                vf_flat = visual_features.view(-1, d_model)  # [N_vis, d_model], float16

                if layer_idx in probes:
                    # TunedLens: apply probe (half → half, probe weights are half)
                    vf_translated = probes[layer_idx](vf_flat)  # float16
                else:
                    # Fallback to LogitLens
                    vf_translated = vf_flat

                logits = apply_lm_head(model, vf_translated)  # [N_vis, vocab], float16
                logits = logits.view(B, num_chunks, patches_per_chunk, -1)

                # Top-k
                topk_values, topk_indices = torch.topk(logits, k=args.top_k, dim=-1)
                topk_values = topk_values.cpu()
                topk_indices = topk_indices.cpu()

                layer_result = {"layer_idx": layer_idx, "chunks": []}
                for chunk_idx in range(num_chunks):
                    chunk_result = {
                        "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                        "patches": [],
                    }
                    for patch_idx in range(patches_per_chunk):
                        vals = topk_values[0, chunk_idx, patch_idx].numpy()
                        idxs = topk_indices[0, chunk_idx, patch_idx].numpy()
                        top_preds = [
                            {
                                "token": decode_token(preprocessor.tokenizer, idx),
                                "token_id": int(idx),
                                "logit": float(val),
                            }
                            for val, idx in zip(vals, idxs)
                        ]
                        row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                        chunk_result["patches"].append({
                            "patch_idx": patch_idx,
                            "patch_row": row,
                            "patch_col": col,
                            "top_predictions": top_preds,
                        })
                    layer_result["chunks"].append(chunk_result)

                image_result["layers"].append(layer_result)

            del input_ids, images_tensor, image_masks, image_input_idx
            del output, hidden_states, visual_features, logits, topk_values, topk_indices
            torch.cuda.empty_cache()

        all_image_results.append(image_result)

    # ===== SAVE RESULTS =====
    # One file per layer — identical structure to logitlens.py output
    ckpt_name = Path(args.ckpt_path).parent.name + "_" + Path(args.ckpt_path).name
    out_dir = Path(args.output_dir) / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {out_dir} ...")

    for layer_idx in analyzed_layers:
        layer_data = []
        for img_res in all_image_results:
            ld = next((l for l in img_res["layers"] if l["layer_idx"] == layer_idx), None)
            if ld is None:
                continue
            layer_data.append({
                "image_idx": img_res["image_idx"],
                "ground_truth_caption": img_res["ground_truth_caption"],
                "chunks": ld["chunks"],
            })

        out_file = out_dir / f"logit_lens_layer{layer_idx}_topk{args.top_k}_multi-gpu.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": args.ckpt_path,
                    "split": args.split,
                    "num_images": args.num_images,
                    "num_processes": 1,
                    "top_k": args.top_k,
                    "layer_idx": layer_idx,
                    "method": "tuned_lens",
                    "probes_path": args.probes_path,
                    "results": layer_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  Layer {layer_idx:2d} → {out_file.name}")

    print(f"\n✓ Done. {len(analyzed_layers)} layers saved to {out_dir}")


if __name__ == "__main__":
    main()
