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
TOTAL_TOKENS = 144
SET_OF_PATCH_TOKENS = [[1], [32], [551], [553], [32, 1, 551, 553]]  # tokens to patch between images
# PATCH_TOKENS = list(range(TOTAL_TOKENS))
NUM_SAMPLES = 100  # number of random image pairs to test

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def evaluate_patched_pair(model, preprocessor, dataset, prompt, source_idx, target_idx, patch_tokens):
    """Evaluate model when patching tokens from source image to target image."""
    
    # Get source and target examples
    source_ex = dataset.get(source_idx, np.random)
    target_ex = dataset.get(target_idx, np.random)
    
    true_source_color = source_ex["metadata"]["color_name"].strip().lower()
    true_target_color = target_ex["metadata"]["color_name"].strip().lower()
    
    # Process both images
    source_batch = preprocessor({
        "image": source_ex["image"],
        "messages": {"messages": [prompt], "style": "none"},
    }, rng=np.random)
    
    target_batch = preprocessor({
        "image": target_ex["image"],
        "messages": {"messages": [prompt], "style": "none"},
    }, rng=np.random)
    
    # Get visual features for both images
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Get source image features
        source_images = torch.tensor(source_batch["images"]).unsqueeze(0).cuda()
        source_masks = torch.tensor(source_batch["image_masks"]).unsqueeze(0).cuda() if source_batch.get("image_masks") is not None else None
        source_features = model.vision_backbone(source_images, source_masks)
        
        # Get target image features
        target_images = torch.tensor(target_batch["images"]).unsqueeze(0).cuda()
        target_masks = torch.tensor(target_batch["image_masks"]).unsqueeze(0).cuda() if target_batch.get("image_masks") is not None else None
        target_features = model.vision_backbone(target_images, target_masks)
        
        # Create patched version by replacing tokens
        patched_features = target_features.clone()
        for token_idx in patch_tokens:
            patched_features[:, :, token_idx, :] = source_features[:, :, token_idx, :]
        
        # Run inference with patched features
        input_ids = torch.tensor(target_batch["input_tokens"]).unsqueeze(0).cuda()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        output = model.generate(
            input_ids=input_ids,
            images=target_images,  # Use target image as base
            image_masks=target_masks,
            image_input_idx=torch.tensor(target_batch.get("image_input_idx")).unsqueeze(0).cuda() if target_batch.get("image_input_idx") is not None else None,
            max_steps=5,
            is_distributed=False,
            precomputed_image_features=patched_features
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        pred = preprocessor.tokenizer.decode(
            output.token_ids[0, 0].cpu().tolist()
        ).strip().lower()
        
        # Clean up
        del source_images, source_masks, source_features
        del target_images, target_masks, target_features
        del patched_features, input_ids, output
        if hasattr(model, "clear_kv_cache"):
            model.clear_kv_cache()
        torch.cuda.empty_cache()
        
    return {
        "source_color": true_source_color,
        "target_color": true_target_color,
        "predicted_color": pred,
        "is_correct": pred == true_source_color
    }

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)
    
    prompt = "Output the color shown in the image:"
    ckpt = "molmo_data/checkpoints/caption-prompt_1color-per-image/step1600-unsharded"
    ckpt_name = ckpt.split("/")[-2] + "_" + ckpt.split("/")[-1]
    
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
    print(f"LENGTH OF DATASET: {len(ds)}")
    
    # -------------------------------------------------------------------------
    # Initialize results structure
    results = {
        "token_set_results": {},
        "overall_results": {
            "num_samples": NUM_SAMPLES,
            "token_sets": [f"tokens_{'_'.join(map(str, tokens))}" for tokens in SET_OF_PATCH_TOKENS],
            "accuracies": {}
        }
    }
    
    # Load existing results if available
    out = Path(f"analysis_results/token_patching_study_{split}_{ckpt_name}.json")
    if out.exists():
        try:
            with open(out, "r") as f:
                old_results = json.load(f)
                # Only load results if they match our current structure
                if "token_set_results" in old_results and "overall_results" in old_results:
                    results = old_results
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    # Generate random pairs of indices
    indices = list(range(NUM_IMGS))
    # Generate exactly NUM_SAMPLES random pairs by sampling with replacement
    pairs = []
    for _ in range(NUM_SAMPLES):
        source_idx = random.choice(indices)
        target_idx = random.choice(indices)
        pairs.append((source_idx, target_idx))
    
    # Run experiments for each token set
    for patch_tokens in SET_OF_PATCH_TOKENS:
        token_set_name = f"tokens_{'_'.join(map(str, patch_tokens))}"
        
        # Initialize results for this token set if not exists
        if token_set_name not in results["token_set_results"]:
            results["token_set_results"][token_set_name] = {
                "patching_results": [],
                "accuracy": 0.0
            }
        
        correct_predictions = 0
        total_predictions = 0
        
        # Run patching experiments for this token set
        for source_idx, target_idx in tqdm(pairs, desc=f"Running experiments for {token_set_name}"):
            result = evaluate_patched_pair(model, pre, ds, prompt, source_idx, target_idx, patch_tokens)
            results["token_set_results"][token_set_name]["patching_results"].append({
                "source_idx": source_idx,
                "target_idx": target_idx,
                **result
            })
            
            if result["is_correct"]:
                correct_predictions += 1
            total_predictions += 1
        
        # Calculate accuracy for this token set
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        results["token_set_results"][token_set_name]["accuracy"] = accuracy
        results["overall_results"]["accuracies"][token_set_name] = accuracy
        print(f"Accuracy for {token_set_name}: {accuracy:.2%}")
    
    # save --------------------------------------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results → {out.resolve()}")
    print("\nOverall Results:")
    for token_set, accuracy in results["overall_results"]["accuracies"].items():
        print(f"{token_set}: {accuracy:.2%}")

if __name__ == "__main__":
    main() 