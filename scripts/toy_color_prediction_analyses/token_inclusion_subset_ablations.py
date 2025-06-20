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

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TOTAL_TOKENS = 576
SPECIAL_TOKENS = [1, 23, 551]  # hypothesis‑relevant tokens
NUM_SAMPLES   = 50               # random pairs / triplets to test

# Custom token combinations to test (set to None to use random sampling)
CUSTOM_GROUPS = None
# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def all_token_indices():
    """Return list[0..143]."""
    return list(range(TOTAL_TOKENS))


def sample_token_combinations(n: int, k: int, include=None):
    """Randomly sample *n* k‑sized combos from pool (optionally including some)."""
    pool = [i for i in all_token_indices() if include is None or i in include]
    combos = list(itertools.combinations(pool, k))
    if n >= len(combos):  # rare edge case: ask for more than exist
        return combos
    return random.sample(combos, n)


# ----------------------------------------------------------------------------
# CORE EVALUATION
# ----------------------------------------------------------------------------

def evaluate_subset(model, preprocessor, dataset, prompt, included_tokens, num_images):
    """Return accuracy when only *included_tokens* are fed (others are masked out)."""

    kept_indices = sorted(set(included_tokens))
    correct = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(num_images):
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

            output = model.generate(
                input_ids=input_ids,
                images=images,
                image_masks=image_masks,
                image_input_idx=(
                    torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda()
                    if batch.get("image_input_idx") is not None
                    else None
                ),
                max_steps=5,
                is_distributed=False,
                subset_of_visual_tokens=kept_indices,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pred = preprocessor.tokenizer.decode(
                output.token_ids[0, 0].cpu().tolist()
            ).strip().lower()

            if pred == true_color:
                correct += 1

            # --- clean up ---
            del images, image_masks, input_ids, output
            if hasattr(model, "clear_kv_cache"):
                model.clear_kv_cache()
            torch.cuda.empty_cache()

    return correct / num_images


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)

    prompt = "Output the color shown in the image:"
    ckpt = "molmo_data/checkpoints/caption-prompt_1color-per-image/step1600-unsharded"
    ckpt_name = "caption-prompt_1color-per-image_step1600"

    # model + preprocessor -----------------------------------------------------
    model = Molmo.from_checkpoint(ckpt, device="cuda")
    model.eval()

    cfg = (
        model.config
        if "hf:" in ckpt
        else ModelConfig.load(
            resource_path(ckpt, "config.yaml"), key="model", validate_paths=False
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
    split = "train"
    ds = ColorImageDataset(split=split)
    NUM_IMGS = min(200, len(ds))

    # -------------------------------------------------------------------------
    results = {
        "single_token_included": {},
        "token_pairs_included": {},
        "token_triplets_included": {},
    }

    # Load existing results if available
    out = Path(f"analysis_results/token_inclusion_study_{split}_{ckpt_name}.json")
    if out.exists():
        with open(out, "r") as f:
            results = json.load(f)

    # Only run standard ablations if CUSTOM_GROUPS is None
    if CUSTOM_GROUPS is None:
        # 1‑token ablations --------------------------------------------------------
        for tok in tqdm(range(TOTAL_TOKENS), desc="1‑token"):
            acc = evaluate_subset(model, pre, ds, prompt, [tok], NUM_IMGS)
            results["single_token_included"][str(tok)] = acc

        # 2‑token ablations --------------------------------------------------------
        pair_list = sample_token_combinations(NUM_SAMPLES, 2) + list(
            itertools.combinations(SPECIAL_TOKENS, 2)
        )
        for pair in tqdm(pair_list, desc="2‑token"):
            acc = evaluate_subset(model, pre, ds, prompt, list(pair), NUM_IMGS)
            results["token_pairs_included"][str(pair)] = acc

        # 3‑token ablations --------------------------------------------------------
        triplet_list = sample_token_combinations(NUM_SAMPLES, 3) + [tuple(SPECIAL_TOKENS)]
        for tri in tqdm(triplet_list, desc="3‑token"):
            acc = evaluate_subset(model, pre, ds, prompt, list(tri), NUM_IMGS)
            results["token_triplets_included"][str(tri)] = acc
    else:
        # Run custom groups --------------------------------------------------------
        results["custom_groups"] = {}
        for group in tqdm(CUSTOM_GROUPS, desc="custom-groups"):
            acc = evaluate_subset(model, pre, ds, prompt, group, NUM_IMGS)
            results["custom_groups"][str(group)] = acc

    # save --------------------------------------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results → {out.resolve()}")


if __name__ == "__main__":
    main() 