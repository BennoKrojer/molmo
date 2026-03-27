#!/usr/bin/env python3
"""
Tuned Lens Training for Visual Tokens

Trains per-layer affine probes T_l that map intermediate hidden states to the
final-layer representation space. At inference, applying T_l before the LM head
gives better vocabulary predictions than raw LogitLens.

    LogitLens:  tokens = top-k(lm_head(ln_f(h_l)))
    TunedLens:  tokens = top-k(lm_head(ln_f(T_l(h_l))))
    where T_l(h) = W_l @ h + b_l,  initialized to I and 0.

Training objective (forward KL):
    minimize  KL(target_probs || tuned_probs)
    where target = lm_head(ln_f(h_final))   [final layer, frozen]
          tuned  = lm_head(ln_f(T_l(h_l)))  [per-layer probe, learned]

Training data: visual tokens only from PixMoCap training images.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/train_tunedlens.py \\
        --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_siglip/step12000-unsharded \\
        --layers "0,1,2,4,8,16,24,30,31" \\
        --num-train-images 200 \\
        --epochs 3 \\
        --output-dir analysis_results/tunedlens_probes
"""

import argparse
import gc
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


class TunedLensProbe(nn.Module):
    """Per-layer affine probe: T_l(h) = W_l @ h + b_l.

    Initialized to identity (W_l = I) and zero bias so training starts
    from the LogitLens baseline.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def apply_lm_head(model: Molmo, h: torch.Tensor) -> torch.Tensor:
    """Apply ln_f + ff_out (+ optional logit scale) to hidden states h."""
    normed = model.transformer.ln_f(h)
    logits = model.transformer.ff_out(normed)
    if model.config.scale_logits:
        logits = logits / math.sqrt(model.config.d_model)
    return logits


def main():
    parser = argparse.ArgumentParser(
        description="Train Tuned Lens probes for visual token interpretation"
    )
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model checkpoint directory (step12000-unsharded)")
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layer indices, e.g. '0,1,2,4,8,16,24,30,31'")
    parser.add_argument("--num-train-images", type=int, default=200,
                        help="Number of PixMoCap training images to use (default: 200)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs over the image set (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW learning rate (default: 1e-3)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/tunedlens_probes",
                        help="Directory to save trained probes")
    args = parser.parse_args()

    analyzed_layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda")

    print("=" * 70)
    print("TUNED LENS TRAINING")
    print("=" * 70)
    print(f"Checkpoint:      {args.ckpt_path}")
    print(f"Layers:          {analyzed_layers}")
    print(f"Train images:    {args.num_train_images}")
    print(f"Epochs:          {args.epochs}")
    print(f"Learning rate:   {args.lr}")
    print(f"Output dir:      {args.output_dir}")
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
    torch.cuda.empty_cache()
    print(f"✓ Model loaded in {time.time() - t0:.1f}s")

    # Freeze everything — only probes will have gradients
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.d_model
    print(f"  d_model={d_model}, scale_logits={model.config.scale_logits}")

    # ===== CREATE PROBES =====
    probes = {l: TunedLensProbe(d_model).float().cuda() for l in analyzed_layers}
    all_params = [p for probe in probes.values() for p in probe.parameters()]
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    total_params = sum(p.numel() for p in all_params)
    print(f"  Created {len(probes)} probes — {total_params / 1e6:.1f}M total parameters")

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
    dataset = PixMoCap(split="train", mode="captions")
    prompt = "Describe this image in detail."
    print(f"  Dataset: PixMoCap train split (using images 0–{args.num_train_images - 1})")
    print()

    # ===== TRAINING LOOP =====
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    ckpt_name = Path(args.ckpt_path).parent.name
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_history = []

    for epoch in range(args.epochs):
        epoch_losses = []

        for img_idx in tqdm(range(args.num_train_images), desc=f"Epoch {epoch + 1}/{args.epochs}"):
            example_data = dataset.get(img_idx, np.random)
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

            # ---- Forward pass with frozen model ----
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(
                        input_ids=input_ids,
                        images=images_tensor,
                        image_masks=image_masks,
                        image_input_idx=image_input_idx,
                        output_hidden_states=True,
                        last_logits_only=False,
                    )

                hidden_states = output.hidden_states  # tuple of [1, seq_len, d_model]

                # ---- Identify visual token positions ----
                if image_input_idx is not None:
                    flat_pos = image_input_idx[0].view(-1)
                    valid_mask = flat_pos >= 0
                    vis_positions = flat_pos[valid_mask].long()  # [N_vis]
                else:
                    vis_positions = None

                if vis_positions is None or vis_positions.numel() == 0:
                    # No visual tokens — skip
                    del input_ids, images_tensor, image_masks, image_input_idx, output
                    torch.cuda.empty_cache()
                    continue

                # ---- Compute target distribution from final layer (frozen) ----
                h_final_vis = hidden_states[-1][0, vis_positions].float()  # [N_vis, d_model]
                with torch.autocast("cuda", dtype=torch.float16):
                    target_logits = apply_lm_head(model, h_final_vis.half())  # float16
                target_probs = F.softmax(target_logits.float(), dim=-1).detach()  # [N_vis, vocab]

            # ---- Probe training (autograd enabled, only probe params have grad) ----
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for layer_idx in analyzed_layers:
                # Extract visual token hidden states for this layer
                h_l_vis = hidden_states[layer_idx][0, vis_positions].float().clone()  # [N_vis, d_model]

                # Apply probe (float32) → cast to float16 → apply frozen lm head
                translated = probes[layer_idx](h_l_vis)  # float32
                with torch.autocast("cuda", dtype=torch.float16):
                    pred_logits = apply_lm_head(model, translated.half())  # float16
                pred_log_probs = F.log_softmax(pred_logits.float(), dim=-1)  # [N_vis, vocab]

                # KL(target || tuned): gradient pushes tuned to cover target's modes
                loss = F.kl_div(
                    pred_log_probs,
                    target_probs,
                    reduction="batchmean",
                    log_target=False,
                )
                total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

            del input_ids, images_tensor, image_masks, image_input_idx
            del output, hidden_states, h_final_vis, target_logits, target_probs, total_loss
            torch.cuda.empty_cache()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{args.epochs}: avg_loss={avg_loss:.4f}")

    # ===== SAVE PROBES =====
    print()
    probe_path = output_dir / "probes.pt"
    print(f"Saving probes to {probe_path} ...")
    torch.save(
        {
            "layer_indices": analyzed_layers,
            "d_model": d_model,
            "probes": {l: probes[l].state_dict() for l in analyzed_layers},
            "training_config": {
                "num_train_images": args.num_train_images,
                "epochs": args.epochs,
                "lr": args.lr,
                "loss_history": loss_history,
                "final_loss": loss_history[-1],
            },
        },
        probe_path,
    )
    print(f"✓ Saved probes for layers {analyzed_layers}")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    print(f"  Loss history: {[f'{l:.4f}' for l in loss_history]}")


if __name__ == "__main__":
    main()
