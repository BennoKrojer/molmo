"""Analyze variance and norms of visual token embeddings across positions"""
import logging
import sys
import json
from pathlib import Path
import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap, ColorImageDataset

log = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_stable_variance(data, axis=0, eps=1e-8):
    """Compute variance in a numerically stable way."""
    # Convert to float64 for better numerical stability
    data = data.astype(np.float64)
    
    # Center the data first
    mean = np.mean(data, axis=axis, keepdims=True)
    centered = data - mean
    
    # Compute variance with epsilon to avoid division by zero
    var = np.mean(centered ** 2, axis=axis) + eps
    
    # Replace any remaining infinities or NaNs with the maximum finite value
    var = np.nan_to_num(var, nan=0.0, posinf=np.finfo(var.dtype).max)
    
    return var

def compute_stable_norm(data, axis=-1, eps=1e-8):
    """Compute L2 norm in a numerically stable way."""
    # Convert to float64 for better numerical stability
    data = data.astype(np.float64)
    
    # Square the values and add small epsilon
    squared = data ** 2 + eps
    
    # Sum along specified axis
    summed = np.sum(squared, axis=axis)
    
    # Take the square root
    norm = np.sqrt(summed)
    
    # Replace any infinities or NaNs with the maximum finite value
    norm = np.nan_to_num(norm, nan=0.0, posinf=np.finfo(norm.dtype).max)
    
    return norm

def compute_normalized_variance(data, axis=0, eps=1e-8):
    """Compute variance after normalizing each vector to unit length."""
    # Compute norms for normalization
    norms = compute_stable_norm(data, axis=-1, eps=eps)[..., None]  # Add dimension for broadcasting
    
    # Normalize the vectors to unit length
    normalized_data = data / norms
    
    # Compute variance on normalized vectors
    normalized_var = compute_stable_variance(normalized_data, axis=axis, eps=eps)
    
    return normalized_var

def compute_cosine_similarity(x, y, eps=1e-8):
    """Compute cosine similarity between two vectors."""
    # Ensure inputs are float64 for stability
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    # Compute dot product
    dot_product = np.sum(x * y, axis=-1)
    
    # Compute norms
    norm_x = compute_stable_norm(x)
    norm_y = compute_stable_norm(y)
    
    # Compute similarity
    similarity = dot_product / (norm_x * norm_y + eps)
    
    # Clip to valid range [-1, 1] to handle numerical errors
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return similarity

def compute_pairwise_cosine_similarity(embeddings, eps=1e-8):
    """Compute pairwise cosine similarities between all embeddings."""
    # Ensure input is float64
    embeddings = embeddings.astype(np.float64)
    
    # Normalize embeddings
    norms = compute_stable_norm(embeddings)[..., None]
    normalized = embeddings / (norms + eps)
    
    # Compute pairwise similarities
    similarities = np.matmul(normalized, normalized.T)
    
    # Clip to valid range [-1, 1]
    similarities = np.clip(similarities, -1.0, 1.0)
    
    return similarities

def analyze_token_similarities(embeddings):
    """Analyze similarities between different token positions."""
    # Get number of tokens
    num_tokens = embeddings.shape[0]
    
    if num_tokens < 3:  # Need at least 3 tokens for this analysis
        return None
    
    # Extract tokens
    first_token = embeddings[0]
    last_token = embeddings[-1]
    middle_tokens = embeddings[1:-1]
    
    # Compute average middle token
    avg_middle_token = np.mean(middle_tokens, axis=0)
    
    # Compute average similarity between middle tokens
    # More efficient than storing all pairs
    middle_similarities = compute_pairwise_cosine_similarity(middle_tokens)
    # Get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(len(middle_tokens), k=1)
    mean_middle_similarity = float(np.mean(middle_similarities[triu_indices]))
    
    results = {
        # Core similarities
        "first_last_similarity": float(compute_cosine_similarity(first_token, last_token)),
        "first_avg_middle_similarity": float(compute_cosine_similarity(first_token, avg_middle_token)),
        "last_avg_middle_similarity": float(compute_cosine_similarity(last_token, avg_middle_token)),
        "mean_middle_similarity": mean_middle_similarity
    }
    
    return results

def plot_statistics(results_dir, split_name, position_variances, position_norms, normalized_variances):
    """Create plots for variances and norms across positions."""
    plt.figure(figsize=(15, 5))
    
    # Convert dictionaries to position-ordered lists if they are dictionaries
    if isinstance(position_variances, dict):
        # Create a list of the original length filled with zeros
        ordered_variances = [0] * len(position_variances)
        # Fill in the values at their correct positions
        for pos, val in position_variances.items():
            ordered_variances[int(pos)] = val
        position_variances = ordered_variances
        
    if isinstance(position_norms, dict):
        ordered_norms = [0] * len(position_norms)
        for pos, val in position_norms.items():
            ordered_norms[int(pos)] = val
        position_norms = ordered_norms
        
    if isinstance(normalized_variances, dict):
        ordered_norm_vars = [0] * len(normalized_variances)
        for pos, val in normalized_variances.items():
            ordered_norm_vars[int(pos)] = val
        normalized_variances = ordered_norm_vars
    
    # Plot variances
    plt.subplot(1, 3, 1)
    plt.plot(position_variances, label='Raw Variance')
    plt.title(f'Raw Embedding Variance\nper Position ({split_name})')
    plt.xlabel('Position')
    plt.ylabel('Variance')
    plt.yscale('log')  # Log scale to better see the distribution
    plt.grid(True)
    
    # Plot normalized variances
    plt.subplot(1, 3, 2)
    plt.plot(normalized_variances, label='Normalized Variance', color='green')
    plt.title(f'Normalized Embedding Variance\nper Position ({split_name})')
    plt.xlabel('Position')
    plt.ylabel('Variance (Unit Vectors)')
    plt.yscale('log')  # Log scale to better see the distribution
    plt.grid(True)
    
    # Plot norms
    plt.subplot(1, 3, 3)
    plt.plot(position_norms, label='Norm', color='orange')
    plt.title(f'Average Embedding Norm\nper Position ({split_name})')
    plt.xlabel('Position')
    plt.ylabel('L2 Norm')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'embedding_stats_{split_name}.png')
    plt.close()

def get_cached_tokens_path(results_dir, split):
    """Get path for cached tokens."""
    return results_dir / f"cached_visual_tokens_{split}.npz"

def save_tokens_to_cache(tokens_dict, cache_path):
    """Save tokens to cache file."""
    log.info(f"Saving tokens to cache: {cache_path}")
    np.savez_compressed(cache_path, **tokens_dict)

def load_tokens_from_cache(cache_path):
    """Load tokens from cache file if it exists."""
    if cache_path.exists():
        log.info(f"Loading tokens from cache: {cache_path}")
        data = np.load(cache_path)
        return {str(i): data[str(i)] for i in range(len(data.files))}
    return None

def compute_visual_tokens(model, preprocessor, dataset, num_images):
    """Compute visual tokens for all images."""
    tokens_dict = {}
    
    for i in tqdm(range(num_images), desc="Computing visual tokens"):
        try:
            example_data = dataset.get(i, np.random)
            example = {
                "image": example_data["image"],
                "messages": {
                    "messages": [""],
                    "style": "long_caption"
                }
            }
            batch = preprocessor(example, rng=np.random)

            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                    image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                    image_features = model.vision_backbone(images_tensor, image_masks_tensor)
                    image_features = image_features.float()
                    current_embeddings = image_features.cpu().numpy().squeeze()
                    tokens_dict[str(i)] = current_embeddings

                    del images_tensor, image_masks_tensor, image_features
                    clear_gpu_memory()
        except Exception as e:
            log.error(f"Error processing image {i}: {str(e)}")
            continue
            
    return tokens_dict

def process_split(model, preprocessor, dataset, num_images, results_dir):
    """Process a dataset split and compute embedding statistics."""
    split = "train" if "train" in str(dataset.split) else "validation"
    cache_path = get_cached_tokens_path(results_dir, split)
    
    # Try to load from cache first
    tokens_dict = load_tokens_from_cache(cache_path)
    
    # Compute tokens if not in cache
    if tokens_dict is None:
        if model is None or preprocessor is None:
            raise ValueError("Model and preprocessor required when cache not available!")
        tokens_dict = compute_visual_tokens(model, preprocessor, dataset, num_images)
        save_tokens_to_cache(tokens_dict, cache_path)
    
    # Process the tokens
    all_embeddings = []
    per_image_results = {}
    
    log.info(f"Processing {len(tokens_dict)} images...")
    
    for img_idx, embeddings in tokens_dict.items():
        try:
            similarities = analyze_token_similarities(embeddings)
            if similarities is not None:
                per_image_results[img_idx] = {
                    "total_tokens": len(embeddings),
                    "similarities": similarities
                }
                log.debug(f"Processed image {img_idx} with {len(embeddings)} tokens")
                all_embeddings.append(embeddings)
            else:
                log.warning(f"Skipping image {img_idx} - not enough tokens for similarity analysis")
        except Exception as e:
            log.error(f"Error processing image {img_idx}: {str(e)}")
            continue

    if not per_image_results:
        raise ValueError("No valid images were processed successfully!")

    all_embeddings = np.stack(all_embeddings, axis=0).astype(np.float64)
    
    # Compute aggregate statistics
    variance_per_position = compute_stable_variance(all_embeddings, axis=0)
    normalized_variance = compute_normalized_variance(all_embeddings, axis=0)
    embedding_norms = compute_stable_norm(all_embeddings, axis=-1)
    mean_norms = np.mean(embedding_norms, axis=0)
    
    if variance_per_position.ndim > 2:
        variance_per_position = np.mean(variance_per_position, axis=0)
        normalized_variance = np.mean(normalized_variance, axis=0)
        mean_norms = np.mean(mean_norms, axis=0)
    
    total_variance_per_position = np.mean(variance_per_position, axis=-1)
    total_normalized_variance = np.mean(normalized_variance, axis=-1)
    
    try:
        avg_similarities = {
            "mean_first_last_similarity": float(np.mean([r["similarities"]["first_last_similarity"] 
                                                        for r in per_image_results.values()])),
            "mean_first_avg_middle_similarity": float(np.mean([r["similarities"]["first_avg_middle_similarity"] 
                                                             for r in per_image_results.values()])),
            "mean_last_avg_middle_similarity": float(np.mean([r["similarities"]["last_avg_middle_similarity"] 
                                                            for r in per_image_results.values()])),
            "mean_middle_similarity": float(np.mean([r["similarities"]["mean_middle_similarity"]
                                                   for r in per_image_results.values()]))
        }
    except Exception as e:
        log.error(f"Error computing average similarities: {str(e)}")
        avg_similarities = {
            "mean_first_last_similarity": None,
            "mean_first_avg_middle_similarity": None,
            "mean_last_avg_middle_similarity": None,
            "mean_middle_similarity": None
        }
    
    results = {
        "split": split,
        "embedding_shape": list(all_embeddings.shape),
        "aggregate_statistics": {
            "mean_variance_per_chunk": float(np.mean(total_variance_per_position)),
            "mean_normalized_variance_per_chunk": float(np.mean(total_normalized_variance)),
            "position_variances": {str(i): float(v) for i, v in sorted(enumerate(total_variance_per_position), key=lambda x: x[1], reverse=True)},
            "position_normalized_variances": {str(i): float(v) for i, v in sorted(enumerate(total_normalized_variance), key=lambda x: x[1], reverse=True)},
            "position_norms": {str(i): float(v) for i, v in sorted(enumerate(mean_norms), key=lambda x: x[1], reverse=True)},
            "token_similarities": avg_similarities
        },
        "per_image_results": per_image_results,
        "num_images_processed": len(per_image_results)
    }
    
    output_file = results_dir / f"embedding_analysis_{split}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Successfully processed {len(per_image_results)} images")
    log.info(f"Results saved to {output_file}")
    
    plot_statistics(results_dir, split,
                   total_variance_per_position,
                   mean_norms,
                   total_normalized_variance)
    
    return results

def main():
    # Hardcoded parameters
    checkpoint_path = "molmo_data/checkpoints/caption-prompt_1color-per-image/step1600-unsharded"

    # Setup results directory
    ckpt_name = Path(checkpoint_path).name
    results_dir = Path("analysis_results/embedding_variances") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if we need to load the model (if cache doesn't exist)
    train_cache = get_cached_tokens_path(results_dir, "train")
    val_cache = get_cached_tokens_path(results_dir, "validation")
    
    if not train_cache.exists() or not val_cache.exists():
        # Load model only if needed
        log.info(f"Loading model from {checkpoint_path}")
        model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
        model.eval()

        # Create preprocessor
        if "hf:" in checkpoint_path:
            model_config = model.config
        else:
            model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
        model_config.system_prompt_kind = "style"
        preprocessor = build_mm_preprocessor(
            model_config,
            for_inference=True,
            shuffle_messages=False,
            is_training=False,
            require_image_features=True
        )
    else:
        model = None
        preprocessor = None
        log.info("Using cached tokens, skipping model loading")

    try:
        # Process train split
        log.info("Processing train split...")
        train_dataset = ColorImageDataset(split="train")
        num_train_images = min(500, len(train_dataset))
        train_results = process_split(model, preprocessor, train_dataset, num_train_images, results_dir)
        
        if model is not None:
            clear_gpu_memory()

        # Process validation split
        log.info("Processing validation split...")
        val_dataset = ColorImageDataset(split="validation")
        num_val_images = min(200, len(val_dataset))
        val_results = process_split(model, preprocessor, val_dataset, num_val_images, results_dir)

    except Exception as e:
        log.error(f"Error during processing: {str(e)}")
        raise

    log.info(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
