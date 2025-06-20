#!/usr/bin/env python3
"""
Analyze cosine similarity among visual tokens of the same color:
a) within the same image
b) across different images
c) baseline: tokens with different colors (within and across images)

This script uses only the vision backbone, not the full LLM.
"""

import logging
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import ColorImageDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_cosine_similarity(x, y, eps=1e-8):
    """Compute cosine similarity between two vectors."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    dot_product = np.sum(x * y, axis=-1)
    norm_x = np.sqrt(np.sum(x ** 2, axis=-1) + eps)
    norm_y = np.sqrt(np.sum(y ** 2, axis=-1) + eps)
    
    similarity = dot_product / (norm_x * norm_y + eps)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return similarity

def compute_cosine_similarity_vectorized(X, Y=None, eps=1e-8):
    """Compute cosine similarity between sets of vectors efficiently using vectorized operations."""
    X = X.astype(np.float64)
    
    if Y is None:
        Y = X
    else:
        Y = Y.astype(np.float64)
    
    # Normalize vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + eps)
    
    # Compute cosine similarity matrix
    similarities = np.dot(X_norm, Y_norm.T)
    similarities = np.clip(similarities, -1.0, 1.0)
    
    return similarities

def extract_visual_tokens(model, preprocessor, image_path, color_sequence):
    """Extract visual tokens from a single image using only the vision backbone."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        example = {
            "image": image,
            "messages": {
                "messages": [""],  # Empty prompt since we only need vision features
                "style": "long_caption"
            }
        }
        
        batch = preprocessor(example, rng=np.random)
        
        # Extract visual features using only the vision backbone
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                
                # Get visual tokens from vision backbone only
                image_features = model.vision_backbone(images_tensor, image_masks_tensor)
                image_features = image_features.float().cpu().numpy().squeeze()
                
                # Debug: Print shape information for the first image
                if not hasattr(extract_visual_tokens, '_shape_logged'):
                    log.info(f"Image features shape: {image_features.shape}")
                    log.info(f"Color sequence length: {len(color_sequence)}")
                    extract_visual_tokens._shape_logged = True
                
                # Clean up GPU memory
                del images_tensor, image_masks_tensor
                clear_gpu_memory()
                
                return image_features, color_sequence
                
    except Exception as e:
        log.error(f"Error processing image {image_path}: {str(e)}")
        return None, None

def group_tokens_by_color(image_features, color_sequence):
    """Group visual tokens by their corresponding colors."""
    color_to_tokens = defaultdict(list)
    
    # The color_sequence has 144 colors (12x12 grid)
    # The image_features may have a different number of patches depending on the vision model
    # We need to map the visual tokens to the color grid positions
    
    num_patches = image_features.shape[0] if len(image_features.shape) == 2 else image_features.shape[1]
    num_colors = len(color_sequence)
    

    # Direct 1:1 mapping
    for i, color in enumerate(color_sequence):
        if i < num_patches:
            if len(image_features.shape) == 2:
                color_to_tokens[color].append(image_features[i])
            else:
                color_to_tokens[color].append(image_features[0, i])
    
    return color_to_tokens

def compute_within_image_similarities(color_to_tokens, max_tokens_per_color=50):
    """Compute average cosine similarity among tokens of the same color within an image."""
    within_image_similarities = {}
    
    for color, tokens in color_to_tokens.items():
        if len(tokens) < 2:
            continue  # Need at least 2 tokens to compute similarity
            
        tokens = np.array(tokens)
        
        # Subsample if we have too many tokens
        if len(tokens) > max_tokens_per_color:
            indices = np.random.choice(len(tokens), max_tokens_per_color, replace=False)
            tokens = tokens[indices]
        
        # Vectorized similarity computation
        sim_matrix = compute_cosine_similarity_vectorized(tokens)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = np.triu(sim_matrix, k=1)
        similarities = upper_triangle[upper_triangle != 0]
        
        if len(similarities) > 0:
            within_image_similarities[color] = np.mean(similarities)
    
    return within_image_similarities

def compute_within_image_different_color_similarities(color_to_tokens, max_pairs=1000, max_tokens_per_color=20):
    """Compute average cosine similarity among tokens of DIFFERENT colors within an image."""
    colors = list(color_to_tokens.keys())
    if len(colors) < 2:
        return None
    
    similarities = []
    pairs_computed = 0
    
    # Subsample colors if we have too many
    if len(colors) > 10:
        colors = np.random.choice(colors, 10, replace=False).tolist()
    
    # Compare tokens between different colors
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors[i+1:], i+1):
            if pairs_computed >= max_pairs:
                break
                
            tokens1 = np.array(color_to_tokens[color1])
            tokens2 = np.array(color_to_tokens[color2])
            
            # Subsample tokens
            if len(tokens1) > max_tokens_per_color:
                indices = np.random.choice(len(tokens1), max_tokens_per_color, replace=False)
                tokens1 = tokens1[indices]
            if len(tokens2) > max_tokens_per_color:
                indices = np.random.choice(len(tokens2), max_tokens_per_color, replace=False)
                tokens2 = tokens2[indices]
            
            # Vectorized similarity computation
            sim_matrix = compute_cosine_similarity_vectorized(tokens1, tokens2)
            flat_sims = sim_matrix.flatten()
            
            # Add to similarities list (limit total pairs)
            remaining_pairs = max_pairs - pairs_computed
            if len(flat_sims) > remaining_pairs:
                flat_sims = np.random.choice(flat_sims, remaining_pairs, replace=False)
            
            similarities.extend(flat_sims)
            pairs_computed += len(flat_sims)
            
        if pairs_computed >= max_pairs:
            break
    
    return np.mean(similarities) if similarities else None

def compute_across_image_similarities(all_color_tokens, max_tokens_per_image=30):
    """Compute average cosine similarity among tokens of the same color across different images."""
    across_image_similarities = {}
    
    for color, token_lists in all_color_tokens.items():
        # Flatten all tokens of this color from all images with subsampling
        all_tokens = []
        image_indices = []
        
        for img_idx, tokens in enumerate(token_lists):
            tokens_array = np.array(tokens)
            # Subsample tokens per image
            if len(tokens_array) > max_tokens_per_image:
                indices = np.random.choice(len(tokens_array), max_tokens_per_image, replace=False)
                tokens_array = tokens_array[indices]
            
            for token in tokens_array:
                all_tokens.append(token)
                image_indices.append(img_idx)
        
        if len(all_tokens) < 2:
            continue
            
        all_tokens = np.array(all_tokens)
        image_indices = np.array(image_indices)
        
        # Create mask for different images
        n_tokens = len(all_tokens)
        img_indices_expanded = image_indices[:, np.newaxis]
        different_images_mask = img_indices_expanded != img_indices_expanded.T
        
        # Compute all similarities at once
        sim_matrix = compute_cosine_similarity_vectorized(all_tokens)
        
        # Apply mask and get upper triangle to avoid duplicates
        upper_triangle_mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        valid_mask = different_images_mask & upper_triangle_mask
        
        similarities = sim_matrix[valid_mask]
        
        if len(similarities) > 0:
            across_image_similarities[color] = np.mean(similarities)
    
    return across_image_similarities

def compute_across_image_different_color_similarities(all_color_tokens, max_pairs=5000, max_tokens_per_color_per_image=15):
    """Compute average cosine similarity among tokens of DIFFERENT colors across different images."""
    colors = list(all_color_tokens.keys())
    if len(colors) < 2:
        return None
    
    # Subsample colors if we have too many
    if len(colors) > 8:
        colors = np.random.choice(colors, 8, replace=False).tolist()
    
    similarities = []
    pairs_computed = 0
    
    # Compare tokens between different colors across images
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors[i+1:], i+1):
            if pairs_computed >= max_pairs:
                break
                
            # Collect tokens with image indices for both colors
            tokens1_list = []
            img_indices1 = []
            tokens2_list = []
            img_indices2 = []
            
            # Subsample tokens for color1
            for img_idx, tokens in enumerate(all_color_tokens[color1]):
                tokens_array = np.array(tokens)
                if len(tokens_array) > max_tokens_per_color_per_image:
                    indices = np.random.choice(len(tokens_array), max_tokens_per_color_per_image, replace=False)
                    tokens_array = tokens_array[indices]
                
                tokens1_list.extend(tokens_array)
                img_indices1.extend([img_idx] * len(tokens_array))
            
            # Subsample tokens for color2
            for img_idx, tokens in enumerate(all_color_tokens[color2]):
                tokens_array = np.array(tokens)
                if len(tokens_array) > max_tokens_per_color_per_image:
                    indices = np.random.choice(len(tokens_array), max_tokens_per_color_per_image, replace=False)
                    tokens_array = tokens_array[indices]
                
                tokens2_list.extend(tokens_array)
                img_indices2.extend([img_idx] * len(tokens_array))
            
            if not tokens1_list or not tokens2_list:
                continue
                
            tokens1 = np.array(tokens1_list)
            tokens2 = np.array(tokens2_list)
            img_indices1 = np.array(img_indices1)
            img_indices2 = np.array(img_indices2)
            
            # Compute similarity matrix
            sim_matrix = compute_cosine_similarity_vectorized(tokens1, tokens2)
            
            # Create mask for different images
            img_mask = img_indices1[:, np.newaxis] != img_indices2[np.newaxis, :]
            
            # Get similarities for different images
            valid_similarities = sim_matrix[img_mask]
            
            # Subsample if too many
            remaining_pairs = max_pairs - pairs_computed
            if len(valid_similarities) > remaining_pairs:
                valid_similarities = np.random.choice(valid_similarities, remaining_pairs, replace=False)
            
            similarities.extend(valid_similarities)
            pairs_computed += len(valid_similarities)
            
        if pairs_computed >= max_pairs:
            break
    
    return np.mean(similarities) if similarities else None

def main():
    parser = argparse.ArgumentParser(description="Analyze cosine similarity among visual tokens of the same color.")
    parser.add_argument("--checkpoint-path", type=str, default="molmo_data/checkpoints/caption-prompt_mosaic-image/step3000-unsharded",
                        help="Path to the model checkpoint")
    parser.add_argument("--data-dir", type=str, default="molmo_data/color_mosaic_images",
                        help="Directory containing color mosaic images and metadata")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of images to analyze (default: 100)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/color_similarities/caption-prompt_mosaic-image_step3000-unsharded",
                        help="Directory to save analysis results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible subsampling (default: 42)")
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    log.info(f"Loading model from {args.checkpoint_path}")
    model = Molmo.from_checkpoint(args.checkpoint_path, device="cuda")
    model.eval()
    
    # Create preprocessor
    if "hf:" in args.checkpoint_path:
        model_config = model.config
    else:
        model_config = ModelConfig.load(resource_path(args.checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "style"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )
    
    # Load color sequence metadata
    data_dir = Path(args.data_dir)
    color_sequences_path = data_dir / "color_sequences.json"
    
    if not color_sequences_path.exists():
        raise FileNotFoundError(f"Color sequences file not found at {color_sequences_path}")
    
    with open(color_sequences_path, 'r') as f:
        color_sequences = json.load(f)
    
    log.info(f"Loaded color sequences for {len(color_sequences)} images")
    
    # Process images and extract visual tokens
    num_to_process = min(args.num_images, len(color_sequences))
    log.info(f"Processing {num_to_process} images...")
    
    all_within_similarities = []  # List of dicts, one per image
    all_within_diff_similarities = []  # List of similarities for different colors within images
    all_color_tokens = defaultdict(list)  # color -> list of token lists (one per image)
    
    processed_count = 0
    pbar = tqdm(color_sequences.items(), total=num_to_process, desc="Processing images")
    
    for filename, color_sequence in pbar:
        if processed_count >= args.num_images:
            break
            
        image_path = data_dir / filename
        if not image_path.exists():
            log.warning(f"Image file not found: {image_path}")
            continue
        
        # Extract visual tokens
        image_features, color_seq = extract_visual_tokens(model, preprocessor, image_path, color_sequence)
        
        if image_features is None:
            continue
        
        # Group tokens by color for this image
        color_to_tokens = group_tokens_by_color(image_features, color_seq)
        
        # Validate that we have tokens for colors
        if not color_to_tokens:
            log.warning(f"No color tokens found for image {filename}")
            continue
        
        # Compute within-image similarities (same color)
        within_similarities = compute_within_image_similarities(color_to_tokens)
        if within_similarities:  # Only add if we have similarities
            all_within_similarities.append(within_similarities)
        
        # Compute within-image similarities (different colors) - baseline
        within_diff_sim = compute_within_image_different_color_similarities(color_to_tokens)
        if within_diff_sim is not None:
            all_within_diff_similarities.append(within_diff_sim)
        
        # Store tokens for across-image analysis
        for color, tokens in color_to_tokens.items():
            all_color_tokens[color].append(tokens)
        
        processed_count += 1
        pbar.set_postfix({"processed": processed_count})
    
    pbar.close()
    log.info(f"Successfully processed {processed_count} images")
    
    # Validate we have enough data
    if not all_within_similarities:
        log.error("No within-image similarities computed. Check your data and model setup.")
        return
    
    if not all_color_tokens:
        log.error("No color tokens collected. Check your data and model setup.")
        return
    
    # Compute across-image similarities (same color)
    log.info("Computing across-image similarities (same color)...")
    across_similarities = compute_across_image_similarities(all_color_tokens)
    
    # Compute across-image similarities (different colors) - baseline
    log.info("Computing across-image similarities (different colors)...")
    across_diff_sim = compute_across_image_different_color_similarities(all_color_tokens)
    
    # Aggregate within-image similarities by color
    log.info("Aggregating within-image similarities...")
    within_similarities_by_color = defaultdict(list)
    for img_similarities in all_within_similarities:
        for color, sim in img_similarities.items():
            within_similarities_by_color[color].append(sim)
    
    # Compute average within-image similarities
    avg_within_similarities = {}
    for color, sims in within_similarities_by_color.items():
        avg_within_similarities[color] = np.mean(sims)
    
    # Compute average within-image different color similarities
    avg_within_diff_sim = np.mean(all_within_diff_similarities) if all_within_diff_similarities else None
    
    # Print and save results
    log.info("\n" + "="*80)
    log.info("COSINE SIMILARITY ANALYSIS RESULTS")
    log.info("="*80)
    
    log.info("\nAverage cosine similarity WITHIN same image (SAME color tokens):")
    for color in sorted(avg_within_similarities.keys()):
        log.info(f"  {color}: {avg_within_similarities[color]:.4f}")
    
    overall_within_avg = np.mean(list(avg_within_similarities.values()))
    log.info(f"\nOverall average within-image (same color) similarity: {overall_within_avg:.4f}")
    
    if avg_within_diff_sim is not None:
        log.info(f"Overall average within-image (DIFFERENT color) similarity: {avg_within_diff_sim:.4f}")
        log.info(f"Difference (same - different within image): {overall_within_avg - avg_within_diff_sim:.4f}")
    
    log.info("\nAverage cosine similarity ACROSS different images (SAME color tokens):")
    for color in sorted(across_similarities.keys()):
        log.info(f"  {color}: {across_similarities[color]:.4f}")
    
    overall_across_avg = np.mean(list(across_similarities.values()))
    log.info(f"\nOverall average across-image (same color) similarity: {overall_across_avg:.4f}")
    
    if across_diff_sim is not None:
        log.info(f"Overall average across-image (DIFFERENT color) similarity: {across_diff_sim:.4f}")
        log.info(f"Difference (same - different across images): {overall_across_avg - across_diff_sim:.4f}")
    
    log.info(f"\nDifference (within same color - across same color): {overall_within_avg - overall_across_avg:.4f}")
    
    # Save detailed results
    results = {
        "within_image_same_color": {
            "by_color": avg_within_similarities,
            "overall_average": float(overall_within_avg)
        },
        "within_image_different_color": {
            "overall_average": float(avg_within_diff_sim) if avg_within_diff_sim is not None else None
        },
        "across_image_same_color": {
            "by_color": across_similarities,
            "overall_average": float(overall_across_avg)
        },
        "across_image_different_color": {
            "overall_average": float(across_diff_sim) if across_diff_sim is not None else None
        },
        "differences": {
            "within_same_minus_within_different": float(overall_within_avg - avg_within_diff_sim) if avg_within_diff_sim is not None else None,
            "across_same_minus_across_different": float(overall_across_avg - across_diff_sim) if across_diff_sim is not None else None,
            "within_same_minus_across_same": float(overall_within_avg - overall_across_avg)
        },
        "num_images_processed": processed_count,
        "colors_analyzed": list(set(avg_within_similarities.keys()) | set(across_similarities.keys())),
        "random_seed": args.seed
    }
    
    results_path = output_dir / "color_similarity_analysis_with-model-trained-on-mosaic-images.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nDetailed results saved to: {results_path}")
    log.info("="*80)

if __name__ == "__main__":
    main() 