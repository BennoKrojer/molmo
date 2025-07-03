"""
This script tests the mosaic-trained model on mosaic data to analyze if the 144 visual tokens
encode local information. For each visual token position:
a) Replace with noise and check if the generated response changes only at that position
b) Replace with a token from another image and check if it predicts that token's color

The goal is to understand if visual token i corresponds to spatial position i in the output sequence.

This version uses pre-computed cached features to avoid OOM issues.
"""

import itertools
import random
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from olmo.data.pixmo_datasets import ColorMosaicDataset
from olmo.config import ModelConfig
from olmo.util import resource_path
from olmo.model import Molmo
from olmo.data import build_mm_preprocessor

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
TOTAL_TOKENS = 576
CKPT = "molmo_data/checkpoints/caption-prompt_mosaic-image/step1600-unsharded"
SPLIT = "train"
PROMPT = "What is the sequence of colors in this grid of colors, read from left to right like a page?"
NUM_IMAGES = 40  # Further reduced for testing
NUM_TOKEN_POSITIONS = 4  # Further reduced for testing
NUM_SOURCE_IMAGES = 4  # Increased to have enough source images for token replacement
USE_CACHE = True  # Whether to use pre-computed features
MAX_STEPS = 600  # Need 150 steps to generate 144 color tokens

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def precompute_all_features(model, preprocessor, dataset, num_images, cache_dir):
    """Pre-compute and cache image features only (no baseline responses to avoid OOM)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Pre-computing image features for {num_images} images...")
    
    for i in tqdm(range(num_images), desc="Pre-computing image features"):
        feature_file = cache_dir / f"features_{i}.pt"
        batch_file = cache_dir / f"batch_{i}.pt"
        
        # Skip if already cached
        if feature_file.exists() and batch_file.exists():
            continue
            
        # Get example and preprocess
        ex = dataset.get(i, np.random)
        example = {
            "image": ex["image"],
            "messages": {"messages": [PROMPT], "style": "none"},
        }
        batch = preprocessor(example, rng=np.random)
        
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                
                # Get image features ONLY - no generate call
                image_features = model.vision_backbone(images_tensor, image_masks_tensor)
                
                # Save to disk
                torch.save(image_features.detach().cpu(), feature_file)
                torch.save(batch, batch_file)
                
                # Clear GPU tensors
                del images_tensor, image_masks_tensor, image_features
                if hasattr(model, "clear_kv_cache"):
                    model.clear_kv_cache()
                clear_gpu_memory()

def precompute_baseline_responses(model, preprocessor, num_images, cache_dir):
    """Generate and cache baseline responses using pre-computed features."""
    cache_dir = Path(cache_dir)
    
    print(f"Pre-computing baseline responses for {num_images} images...")
    
    for i in tqdm(range(num_images), desc="Pre-computing baseline responses"):
        response_file = cache_dir / f"response_{i}.json"
        
        # Skip if already cached
        if response_file.exists():
            continue
            
        # Load cached data
        features, batch = load_cached_features_and_batch(i, cache_dir)
        
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()
                features_gpu = features.cuda()
                
                # Clear cache before generate call like working scripts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate baseline response using cached features
                output = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                    max_steps=MAX_STEPS,
                    is_distributed=False,
                    precomputed_image_features=features_gpu
                )
                token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                baseline_response = preprocessor.tokenizer.decode(token_ids[0]).strip()
                
                # Save response and token_ids to disk
                with open(response_file, 'w') as f:
                    json.dump({
                        "baseline_response": baseline_response,
                        "token_ids": token_ids[0].tolist()
                    }, f)
                
                # Clear GPU tensors exactly like working scripts
                del images_tensor, image_masks_tensor, input_ids, features_gpu, output
                if hasattr(model, "clear_kv_cache"):
                    model.clear_kv_cache()
                torch.cuda.empty_cache()
        
        # Additional cleanup after each image
        del features, batch
        clear_gpu_memory()

def load_cached_features_and_batch(image_idx, cache_dir):
    """Load pre-computed features and batch from cache."""
    cache_dir = Path(cache_dir)
    feature_file = cache_dir / f"features_{image_idx}.pt"
    batch_file = cache_dir / f"batch_{image_idx}.pt"
    
    features = torch.load(feature_file, map_location='cpu', weights_only=False)
    batch = torch.load(batch_file, map_location='cpu', weights_only=False)
    
    return features, batch

def load_cached_data(image_idx, cache_dir):
    """Load pre-computed baseline response and token_ids from cache."""
    cache_dir = Path(cache_dir)
    response_file = cache_dir / f"response_{image_idx}.json"
    
    with open(response_file, 'r') as f:
        response_data = json.load(f)
    
    return response_data["baseline_response"], response_data["token_ids"]

def test_with_modified_features(model, preprocessor, batch, modified_features):
    """Run inference with modified features."""
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            # Move data to GPU
            images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
            image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
            input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()
            modified_features_gpu = modified_features.cuda()
            
            # Clear cache before generate call like working scripts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate response with modified features
            output = model.generate(
                input_ids=input_ids,
                images=images_tensor,
                image_masks=image_masks_tensor,
                image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                max_steps=MAX_STEPS,
                is_distributed=False,
                precomputed_image_features=modified_features_gpu
            )
            token_ids = output.token_ids[:, 0].detach().cpu().numpy()
            response = preprocessor.tokenizer.decode(token_ids[0]).strip()
            
            # Clear GPU tensors exactly like working scripts
            del images_tensor, image_masks_tensor, input_ids, modified_features_gpu, output
            if hasattr(model, "clear_kv_cache"):
                model.clear_kv_cache()
            torch.cuda.empty_cache()
            
            return response, token_ids[0]

def analyze_response_differences(baseline_response, baseline_token_ids, modified_response, modified_token_ids, tokenizer):
    """Analyze how the response changed between baseline and modified versions using actual tokens."""
    # Convert token IDs to individual token strings
    baseline_tokens = [tokenizer.decode([token_id]) for token_id in baseline_token_ids]
    modified_tokens = [tokenizer.decode([token_id]) for token_id in modified_token_ids]
    
    # Find positions where tokens differ
    different_positions = []
    min_len = min(len(baseline_tokens), len(modified_tokens))
    
    for i in range(min_len):
        if baseline_token_ids[i] != modified_token_ids[i]:
            different_positions.append({
                "position": i,
                "baseline_token": baseline_tokens[i],
                "modified_token": modified_tokens[i],
                "baseline_token_id": int(baseline_token_ids[i]),
                "modified_token_id": int(modified_token_ids[i])
            })
    
    # Handle length differences
    length_difference = len(modified_tokens) - len(baseline_tokens)
    
    # Add tokens that were added or removed
    if len(modified_tokens) > len(baseline_tokens):
        for i in range(len(baseline_tokens), len(modified_tokens)):
            different_positions.append({
                "position": i,
                "baseline_token": None,
                "modified_token": modified_tokens[i],
                "baseline_token_id": None,
                "modified_token_id": int(modified_token_ids[i]),
                "change_type": "added"
            })
    elif len(baseline_tokens) > len(modified_tokens):
        for i in range(len(modified_tokens), len(baseline_tokens)):
            different_positions.append({
                "position": i,
                "baseline_token": baseline_tokens[i],
                "modified_token": None,
                "baseline_token_id": int(baseline_token_ids[i]),
                "modified_token_id": None,
                "change_type": "removed"
            })
    
    return {
        "baseline_response": baseline_response,
        "modified_response": modified_response,
        "baseline_tokens": baseline_tokens,
        "modified_tokens": modified_tokens,
        "baseline_token_ids": [int(tid) for tid in baseline_token_ids],
        "modified_token_ids": [int(tid) for tid in modified_token_ids],
        "responses_identical": baseline_response == modified_response,
        "token_ids_identical": baseline_token_ids.tolist() == modified_token_ids.tolist() if hasattr(baseline_token_ids, 'tolist') else list(baseline_token_ids) == list(modified_token_ids),
        "different_positions": different_positions,
        "num_differences": len(different_positions),
        "length_difference": length_difference,
        "baseline_length": len(baseline_tokens),
        "modified_length": len(modified_tokens)
    }

def extract_color_sequence_from_dataset(dataset, image_idx):
    """Extract the ground truth color sequence from the dataset."""
    ex = dataset.get(image_idx, np.random)
    if "color_sequence" in ex.get("metadata", {}):
        return ex["metadata"]["color_sequence"]
    elif "colors" in ex.get("metadata", {}):
        return ex["metadata"]["colors"]
    else:
        # Try to extract from other metadata fields
        metadata = ex.get("metadata", {})
        for key in metadata:
            if "color" in key.lower() and isinstance(metadata[key], list):
                return metadata[key]
        return None

def print_summary_statistics(results):
    """Print aggregate statistics about the localization experiments."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Collect all experiments
    all_noise_experiments = []
    all_token_replacements = []
    
    for image_result in results["experiments"]:
        for token_exp in image_result["token_experiments"]:
            if token_exp["noise_replacement"]:
                all_noise_experiments.append(token_exp["noise_replacement"])
            
            for replacement in token_exp["token_replacements"]:
                all_token_replacements.append(replacement)
    
    total_experiments = len(all_noise_experiments) + len(all_token_replacements)
    
    print(f"Total Images Processed: {len(results['experiments'])}")
    print(f"Total Token Positions Tested: {len(all_noise_experiments)}")
    print(f"Total Replacement Experiments: {len(all_token_replacements)}")
    print(f"Total Experiments: {total_experiments}")
    
    if not all_noise_experiments:
        print("No experiments found to analyze!")
        return
    
    print("\n" + "-"*60)
    print("NOISE REPLACEMENT ANALYSIS")
    print("-"*60)
    
    # Noise replacement statistics
    noise_changed_responses = sum(1 for exp in all_noise_experiments if not exp["responses_identical"])
    noise_response_change_rate = noise_changed_responses / len(all_noise_experiments) * 100
    
    noise_edit_distances = [exp["num_differences"] for exp in all_noise_experiments]
    avg_noise_edit_distance = np.mean(noise_edit_distances)
    median_noise_edit_distance = np.median(noise_edit_distances)
    
    # Perfect localization: exactly 1 token changed
    perfect_noise_localization = sum(1 for exp in all_noise_experiments if exp["num_differences"] == 1)
    perfect_noise_rate = perfect_noise_localization / len(all_noise_experiments) * 100
    
    # No change at all
    no_change_noise = sum(1 for exp in all_noise_experiments if exp["num_differences"] == 0)
    no_change_noise_rate = no_change_noise / len(all_noise_experiments) * 100
    
    print(f"Response Change Rate: {noise_response_change_rate:.1f}% ({noise_changed_responses}/{len(all_noise_experiments)})")
    print(f"Average Edit Distance: {avg_noise_edit_distance:.2f} tokens")
    print(f"Median Edit Distance: {median_noise_edit_distance:.1f} tokens")
    print(f"Perfect Localization Rate (exactly 1 change): {perfect_noise_rate:.1f}% ({perfect_noise_localization}/{len(all_noise_experiments)})")
    print(f"No Change Rate: {no_change_noise_rate:.1f}% ({no_change_noise}/{len(all_noise_experiments)})")
    
    if all_token_replacements:
        print("\n" + "-"*60)
        print("TOKEN REPLACEMENT ANALYSIS")
        print("-"*60)
        
        # Token replacement statistics
        token_changed_responses = sum(1 for exp in all_token_replacements if not exp["responses_identical"])
        token_response_change_rate = token_changed_responses / len(all_token_replacements) * 100
        
        token_edit_distances = [exp["num_differences"] for exp in all_token_replacements]
        avg_token_edit_distance = np.mean(token_edit_distances)
        median_token_edit_distance = np.median(token_edit_distances)
        
        # Perfect localization: exactly 1 token changed
        perfect_token_localization = sum(1 for exp in all_token_replacements if exp["num_differences"] == 1)
        perfect_token_rate = perfect_token_localization / len(all_token_replacements) * 100
        
        # No change at all
        no_change_token = sum(1 for exp in all_token_replacements if exp["num_differences"] == 0)
        no_change_token_rate = no_change_token / len(all_token_replacements) * 100
        
        # Color prediction accuracy: does replacing token i with token from source predict source's color?
        color_prediction_successes = 0
        color_prediction_attempts = 0
        
        for exp in all_token_replacements:
            source_color = exp.get("source_ground_truth_color_at_position")
            target_color = exp.get("target_ground_truth_color_at_position")
            
            if source_color and target_color and source_color != target_color:
                color_prediction_attempts += 1
                
                # Check if any of the changed tokens match the source color
                for diff in exp["different_positions"]:
                    if diff.get("modified_token") and source_color.lower() in diff["modified_token"].lower():
                        color_prediction_successes += 1
                        break
        
        print(f"Response Change Rate: {token_response_change_rate:.1f}% ({token_changed_responses}/{len(all_token_replacements)})")
        print(f"Average Edit Distance: {avg_token_edit_distance:.2f} tokens")
        print(f"Median Edit Distance: {median_token_edit_distance:.1f} tokens")
        print(f"Perfect Localization Rate (exactly 1 change): {perfect_token_rate:.1f}% ({perfect_token_localization}/{len(all_token_replacements)})")
        print(f"No Change Rate: {no_change_token_rate:.1f}% ({no_change_token}/{len(all_token_replacements)})")
        
        if color_prediction_attempts > 0:
            color_prediction_rate = color_prediction_successes / color_prediction_attempts * 100
            print(f"Color Prediction Success Rate: {color_prediction_rate:.1f}% ({color_prediction_successes}/{color_prediction_attempts})")
        else:
            print("Color Prediction Success Rate: N/A (no valid attempts)")
    
    print("\n" + "-"*60)
    print("COMPARATIVE ANALYSIS")
    print("-"*60)
    
    if all_token_replacements:
        print(f"Noise vs Token Replacement Response Change Rate: {noise_response_change_rate:.1f}% vs {token_response_change_rate:.1f}%")
        print(f"Noise vs Token Replacement Average Edit Distance: {avg_noise_edit_distance:.2f} vs {avg_token_edit_distance:.2f}")
        print(f"Noise vs Token Replacement Perfect Localization: {perfect_noise_rate:.1f}% vs {perfect_token_rate:.1f}%")
    
    # Position-based analysis
    print("\n" + "-"*60)
    print("SPATIAL POSITION ANALYSIS")
    print("-"*60)
    
    position_stats = {}
    for exp in all_noise_experiments:
        pos = exp.get("token_position")
        if pos is not None:
            if pos not in position_stats:
                position_stats[pos] = {"noise_changes": 0, "noise_total": 0, "token_changes": 0, "token_total": 0}
            position_stats[pos]["noise_total"] += 1
            if not exp["responses_identical"]:
                position_stats[pos]["noise_changes"] += 1
    
    for exp in all_token_replacements:
        pos = exp.get("token_position")
        if pos is not None:
            if pos not in position_stats:
                position_stats[pos] = {"noise_changes": 0, "noise_total": 0, "token_changes": 0, "token_total": 0}
            position_stats[pos]["token_total"] += 1
            if not exp["responses_identical"]:
                position_stats[pos]["token_changes"] += 1
    
    if position_stats:
        # Find most/least responsive positions
        position_response_rates = []
        for pos, stats in position_stats.items():
            total_experiments = stats["noise_total"] + stats["token_total"]
            total_changes = stats["noise_changes"] + stats["token_changes"]
            if total_experiments > 0:
                response_rate = total_changes / total_experiments
                position_response_rates.append((pos, response_rate, total_experiments))
        
        if position_response_rates:
            position_response_rates.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Most Responsive Positions (top 3):")
            for i, (pos, rate, total) in enumerate(position_response_rates[:3]):
                print(f"  Position {pos}: {rate*100:.1f}% response rate ({int(rate*total)}/{total} experiments)")
            
            print(f"Least Responsive Positions (bottom 3):")
            for i, (pos, rate, total) in enumerate(position_response_rates[-3:]):
                print(f"  Position {pos}: {rate*100:.1f}% response rate ({int(rate*total)}/{total} experiments)")
    
    print("\n" + "="*80)
    print("INTERPRETATION SUMMARY")
    print("="*80)
    
    if perfect_noise_rate > 50:
        print("✓ STRONG spatial localization detected - most noise replacements affect exactly 1 output token")
    elif perfect_noise_rate > 25:
        print("~ MODERATE spatial localization detected - some noise replacements show localized effects")
    else:
        print("✗ WEAK spatial localization detected - noise replacements cause distributed changes")
    
    if all_token_replacements and color_prediction_attempts > 0:
        if color_prediction_rate > 50:
            print("✓ STRONG color prediction ability - token swaps successfully predict source colors")
        elif color_prediction_rate > 25:
            print("~ MODERATE color prediction ability - some token swaps predict source colors")
        else:
            print("✗ WEAK color prediction ability - token swaps rarely predict source colors")
    
    if no_change_noise_rate > 30:
        print("⚠ HIGH no-change rate suggests some visual tokens may be redundant or less important")
    
    print("="*80)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)
    
    # Setup results directory
    ckpt_name = CKPT.split("/")[-2] + "_" + CKPT.split("/")[-1]
    results_dir = Path("analysis_results/mosaic_token_localization") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = results_dir / "cached_features"
    
    # Load model and preprocessor
    print(f"Loading model from {CKPT}")
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
    
    preprocessor = build_mm_preprocessor(
        cfg,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True,
    )
    
    # Load dataset
    dataset = ColorMosaicDataset(split=SPLIT, grid_size=24)
    num_images = min(NUM_IMAGES, len(dataset))
    
    print(f"Will process {num_images} images")
    
    # Pre-compute all features if using cache
    if USE_CACHE:
        precompute_all_features(model, preprocessor, dataset, num_images, cache_dir)
        precompute_baseline_responses(model, preprocessor, num_images, cache_dir)
        print("Pre-computation complete!")
    
    # Initialize results
    all_results = {
        "checkpoint": CKPT,
        "prompt": PROMPT,
        "dataset": "ColorMosaicDataset",
        "split": SPLIT,
        "num_images": num_images,
        "num_token_positions_per_image": NUM_TOKEN_POSITIONS,
        "total_tokens": TOTAL_TOKENS,
        "use_cache": USE_CACHE,
        "experiments": []
    }
    
    # Load source images for token replacement
    source_images_data = []
    for i in range(min(NUM_SOURCE_IMAGES, num_images)):
        if USE_CACHE:
            features, batch = load_cached_features_and_batch(i, cache_dir)
        else:
            # This would use the old approach, but we're focusing on cache for now
            raise NotImplementedError("Non-cache mode not implemented in this version")
        
        # Extract ground truth color sequence
        ground_truth_colors = extract_color_sequence_from_dataset(dataset, i)
        
        source_images_data.append({
            "image_idx": i,
            "features": features,
            "response": load_cached_data(i, cache_dir)[0],
            "ground_truth_colors": ground_truth_colors
        })
    
    # Main experiment loop
    for image_idx in tqdm(range(num_images), desc="Processing images"):
        print(f"\nProcessing image {image_idx}...")
        
        # Load cached data
        if USE_CACHE:
            original_features, batch = load_cached_features_and_batch(image_idx, cache_dir)
        
        # Extract ground truth for current image
        current_ground_truth = extract_color_sequence_from_dataset(dataset, image_idx)
        
        # Sample token positions to test
        token_positions = random.sample(range(TOTAL_TOKENS), NUM_TOKEN_POSITIONS)
        
        image_results = {
            "image_idx": image_idx,
            "ground_truth_colors": current_ground_truth,
            "baseline_response": load_cached_data(image_idx, cache_dir)[0],
            "token_experiments": []
        }
        
        for token_pos in tqdm(token_positions, desc=f"Testing token positions for image {image_idx}", leave=False):
            print(f"  Processing token position {token_pos}")
            token_experiment = {
                "token_position": token_pos,
                "noise_replacement": None,
                "token_replacements": []
            }
            
            # Test 1: Replace with noise
            print(f"    Testing noise replacement at position {token_pos}")
            modified_features = original_features.clone()
            
            # Generate noise with similar statistics to the avg token
            avg_token = torch.mean(original_features, dim=2, keepdim=True)
            avg_token_norm = torch.norm(avg_token, dim=-1, keepdim=True)
            noise = torch.randn_like(avg_token)
            noise_norm = torch.norm(noise, dim=-1, keepdim=True)
            scaled_noise = noise * (avg_token_norm / noise_norm)
            modified_features[:, :, token_pos, :] = scaled_noise
            
            # Clear intermediate tensors
            del avg_token, avg_token_norm, noise, noise_norm, scaled_noise
            
            noise_response, noise_token_ids = test_with_modified_features(model, preprocessor, batch, modified_features)
            baseline_response, baseline_token_ids = load_cached_data(image_idx, cache_dir)
            noise_analysis = analyze_response_differences(baseline_response, baseline_token_ids, noise_response, noise_token_ids, preprocessor.tokenizer)
            
            # Add information about what token was replaced with noise
            if current_ground_truth and token_pos < len(current_ground_truth):
                noise_analysis["target_ground_truth_color_at_position"] = current_ground_truth[token_pos]
            else:
                noise_analysis["target_ground_truth_color_at_position"] = None
                
            noise_analysis["token_position"] = token_pos
            noise_analysis["noise_replacement_summary"] = {
                "replaced_position": token_pos,
                "target_color_at_position": noise_analysis["target_ground_truth_color_at_position"],
                "replacement_type": "noise",
                "responses_changed": not noise_analysis["responses_identical"],
                "edit_distance": noise_analysis["num_differences"]
            }
            
            token_experiment["noise_replacement"] = noise_analysis
            
            # Clean up after noise test
            del modified_features, baseline_response
            if hasattr(model, "clear_kv_cache"):
                model.clear_kv_cache()
            torch.cuda.empty_cache()
            print(f"    Noise replacement completed")
            
            # Test 2: Replace with tokens from other images
            for source_data in source_images_data:
                if source_data["image_idx"] == image_idx:
                    continue  # Skip self-replacement
                
                print(f"    Testing token replacement from image {source_data['image_idx']} at position {token_pos}")
                modified_features = original_features.clone()
                modified_features[:, :, token_pos, :] = source_data["features"][:, :, token_pos, :]
                
                replacement_response, replacement_token_ids = test_with_modified_features(model, preprocessor, batch, modified_features)
                baseline_response, baseline_token_ids = load_cached_data(image_idx, cache_dir)
                replacement_analysis = analyze_response_differences(baseline_response, baseline_token_ids, replacement_response, replacement_token_ids, preprocessor.tokenizer)
                
                # Add detailed information about the replacement
                replacement_analysis["source_image_idx"] = source_data["image_idx"]
                replacement_analysis["source_baseline_response"] = source_data["response"]
                replacement_analysis["source_ground_truth_colors"] = source_data["ground_truth_colors"]
                
                # Add information about what token was replaced
                if current_ground_truth and token_pos < len(current_ground_truth):
                    replacement_analysis["target_ground_truth_color_at_position"] = current_ground_truth[token_pos]
                else:
                    replacement_analysis["target_ground_truth_color_at_position"] = None
                    
                if source_data["ground_truth_colors"] and token_pos < len(source_data["ground_truth_colors"]):
                    replacement_analysis["source_ground_truth_color_at_position"] = source_data["ground_truth_colors"][token_pos]
                else:
                    replacement_analysis["source_ground_truth_color_at_position"] = None
                
                replacement_analysis["token_position"] = token_pos
                replacement_analysis["replacement_summary"] = {
                    "replaced_position": token_pos,
                    "target_color_at_position": replacement_analysis["target_ground_truth_color_at_position"],
                    "source_color_at_position": replacement_analysis["source_ground_truth_color_at_position"],
                    "responses_changed": not replacement_analysis["responses_identical"],
                    "edit_distance": replacement_analysis["num_differences"]
                }
                
                token_experiment["token_replacements"].append(replacement_analysis)
                
                # Clean up after each replacement test
                del modified_features, baseline_response
                if hasattr(model, "clear_kv_cache"):
                    model.clear_kv_cache()
                torch.cuda.empty_cache()
                print(f"    Token replacement completed")
            
            image_results["token_experiments"].append(token_experiment)
            
            # Clean up after each token position
            if hasattr(model, "clear_kv_cache"):
                model.clear_kv_cache()
            torch.cuda.empty_cache()
        
        all_results["experiments"].append(image_results)
        
        # Clean up after each image like working scripts
        del original_features, batch
        if hasattr(model, "clear_kv_cache"):
            model.clear_kv_cache()
        torch.cuda.empty_cache()
        
        # Save intermediate results every 2 images
        if (image_idx + 1) % 2 == 0:
            temp_file = results_dir / f"intermediate_results_cached_{image_idx + 1}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  Saved intermediate results to {temp_file}")
    
    # Save final results
    output_file = results_dir / "mosaic_token_localization_analysis_cached.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total images processed: {len(all_results['experiments'])}")

    # Print summary statistics
    print_summary_statistics(all_results)

if __name__ == "__main__":
    main() 