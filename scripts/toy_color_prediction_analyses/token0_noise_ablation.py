import itertools
import random
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from olmo.data.pixmo_datasets import ColorImageDataset
from olmo.config import ModelConfig
from olmo.util import resource_path
from olmo.model import Molmo
from olmo.data import build_mm_preprocessor

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
TOTAL_TOKENS = 144
EMBEDDING_ANALYSIS_PATH = "analysis_results/embedding_variances/step1600-unsharded/embedding_analysis_train_final.json"
CKPT = "molmo_data/checkpoints/caption-prompt_1color-per-image/step1600-unsharded"
SPLIT = "train"
PROMPT = "Output the color shown in the image:"

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def get_avg_norm():
    """Load JSON and compute average norm from position_norms."""
    with open(EMBEDDING_ANALYSIS_PATH, "r") as f:
        data = json.load(f)
    norms = data["aggregate_statistics"]["position_norms"].values()
    return sum(norms) / len(norms)

def get_position_norms():
    """Load JSON and return dictionary of position-specific norms."""
    with open(EMBEDDING_ANALYSIS_PATH, "r") as f:
        data = json.load(f)
    # Convert string keys to integers and return the position_norms dict
    position_norms = {}
    for pos_str, norm in data["aggregate_statistics"]["position_norms"].items():
        position_norms[int(pos_str)] = norm
    return position_norms

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB max allocated")

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, 'memory_summary'):
            print(torch.cuda.memory_summary())

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate(model, preprocessor, dataset, prompt, num_images, perturb=False, subset="all", use_fixed_tokens_1to143=None, use_position_specific_norms=False, collect_avg_tokens_1to143=False, collected_avg_tokens_1to143=None):
    """Return accuracy when only token 0 is real (others are noise if perturb=True)."""
    correct = 0
    avg_norm = get_avg_norm() if perturb and not use_position_specific_norms else None
    position_norms = get_position_norms() if use_position_specific_norms else None

    if use_fixed_tokens_1to143:
            FIXED_TOKENS_1TO143 = None

    if collect_avg_tokens_1to143:
        avg_tokens_1to143 = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(num_images), desc="Evaluating"):

            ex = dataset.get(i, np.random)
            true_color = ex["metadata"]["color_name"].strip().lower()

            example = {
                "image": ex["image"],
                "messages": {"messages": [prompt], "style": "none"},
            }
            batch = preprocessor(example, rng=np.random)

            images = torch.tensor(batch["images"]).unsqueeze(0).cuda()
            image_masks = (
                torch.tensor(batch["image_masks"]).unsqueeze(0).cuda()
                if batch.get("image_masks") is not None
                else None
            )
            input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extract image features
            with torch.no_grad():
                image_features = model.vision_backbone(images, image_masks)
            
            if collect_avg_tokens_1to143:
                avg_tokens_1to143.append(image_features[:, :, 1:, :])

            # Replace all tokens except 0 with noise if perturbing
            if perturb:
                if use_position_specific_norms:
                    # Use position-specific norms for each token
                    B, C, T, D = image_features.shape
                    for token_pos in range(1, T):  # Skip token 0
                        if token_pos in position_norms:
                            # Create noise for this specific position
                            noise_for_pos = torch.randn((B, C, 1, D), device=image_features.device, dtype=image_features.dtype)
                            # Scale to position-specific norm
                            noise_norm = torch.norm(noise_for_pos, dim=-1, keepdim=True)
                            scaled_noise = noise_for_pos * (position_norms[token_pos] / noise_norm)
                            # Replace this position with scaled noise
                            image_features[:, :, token_pos:token_pos+1, :] = scaled_noise
                else:
                    # Use average norm for all positions (original approach)
                    noise = torch.randn_like(image_features)
                    # Scale noise to have norm equal to average from JSON
                    noise_norm = torch.norm(noise, dim=-1, keepdim=True)
                    noise = noise * (avg_norm / noise_norm)
                    # Keep only token 0, replace others with noise
                    image_features[:,:, 1:, :] = noise[:,:, 1:, :]

            if use_fixed_tokens_1to143:
                if i == 0:
                    FIXED_TOKENS_1TO143 = image_features[:, :, 1:, :]
                image_features[:, :, 1:, :] = FIXED_TOKENS_1TO143
            elif collected_avg_tokens_1to143 is not None:
                image_features[:, :, 1:, :] = collected_avg_tokens_1to143
            

            if subset != "all":
                kept_indices = torch.tensor(subset).cuda()
            else:
                kept_indices = None

            output = model.generate(
                input_ids=input_ids,
                images=images,  # Use target image as base
                image_masks=image_masks,
                image_input_idx=(
                    torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda()
                    if batch.get("image_input_idx") is not None
                    else None
                ),
                max_steps=5,
                is_distributed=False,
                precomputed_image_features=image_features,
                subset_of_visual_tokens=kept_indices
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pred = preprocessor.tokenizer.decode(
                output.token_ids[0, 0].cpu().tolist()
            ).strip().lower()

            if pred == true_color:
                correct += 1

            # --- clean up ---
            del images, image_masks, input_ids, output, image_features
            if hasattr(model, "clear_kv_cache"):
                model.clear_kv_cache()
            torch.cuda.empty_cache()

    if collect_avg_tokens_1to143:
        avg_tokens_1to143 = torch.mean(torch.stack(avg_tokens_1to143), dim=0)
        return correct / num_images, avg_tokens_1to143
    else:
        return correct / num_images

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)

    # model + preprocessor -----------------------------------------------------
    model = Molmo.from_checkpoint(CKPT, device="cuda")
    model.eval()

    cfg = (
        model.config
        if "hf:" in CKPT
        else ModelConfig.load(
            resource_path(CKPT, "config.yaml"), key="model", validate_paths=False
        )
    )
    cfg.system_prompt_kind = "style"  # disable length‑conditioning prompt tricks

    pre = build_mm_preprocessor(
        cfg,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True,
    )

    # dataset --------------------------------------------------------------
    ds = ColorImageDataset(split=SPLIT)
    NUM_IMGS = min(200, len(ds))

    print("\nInitial GPU Memory:")
    print_gpu_memory()

    # First run: baseline (all real tokens)
    print("\nRunning baseline...")
    baseline_acc, avg_tokens_1to143 = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        perturb=False,  # Use real tokens
        collect_avg_tokens_1to143=True,
    )

    print("\nAfter baseline run:")
    print_gpu_memory()

    # Cleanup after baseline
    print("\nCleaning up after baseline...")
    if hasattr(model, "clear_kv_cache"):
        model.clear_kv_cache()
    clear_gpu_memory()

    print("\nAfter cleanup:")
    print_gpu_memory()

    # Second run: only token 0 is real, others are noise
    print("\nRunning perturbed...")
    perturbed_acc = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        perturb=True,  # Replace tokens with noise
    )

    # New run: only token 0 is real, others are position-specific noise
    print("\nRunning perturbed with position-specific norms...")
    perturbed_position_acc = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        perturb=True,  # Replace tokens with noise
        use_position_specific_norms=True,  # Use position-specific norms
    )

    patched_acc = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        use_fixed_tokens_1to143=True,
    )

    patched_acc_with_avg_tokens_1to143 = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        collected_avg_tokens_1to143=avg_tokens_1to143,
    )

    print("\nAfter perturbed run:")
    print_gpu_memory()

    # Third run: only token 0 is real, others are removed
    print("\nRunning subset...")
    subset_acc_0 = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        subset=[0],
    )


    subset_53_112_143_acc = evaluate(
        model=model,
        preprocessor=pre,
        dataset=ds,
        prompt=PROMPT,
        num_images=NUM_IMGS,
        subset=[53, 112, 143],
    )

    print("\nAfter subset run:")
    print_gpu_memory()

    # save --------------------------------------------------------------------
    results = {
        "baseline_accuracy": baseline_acc,
        "perturbed_accuracy": perturbed_acc,
        "perturbed_position_specific_accuracy": perturbed_position_acc,
        "subset_0_accuracy": subset_acc_0,
        "subset_53_112_143_accuracy": subset_53_112_143_acc,
        "patched_accuracy": patched_acc,
        "patched_accuracy_with_avg_tokens_1to143": patched_acc_with_avg_tokens_1to143,
    }

    out = Path(f"analysis_results/token0_noise_ablation_{SPLIT}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results → {out.resolve()}")
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print(f"Perturbed accuracy: {perturbed_acc:.3f}")
    print(f"Perturbed (position-specific) accuracy: {perturbed_position_acc:.3f}")
    print(f"Subset [0] accuracy: {subset_acc_0:.3f}")
    print(f"Subset [53, 112, 143] accuracy: {subset_53_112_143_acc:.3f}")
    print(f"Patched accuracy: {patched_acc:.3f}")
    print(f"Patched accuracy with avg tokens 1to143: {patched_acc_with_avg_tokens_1to143:.3f}")
if __name__ == "__main__":
    main()