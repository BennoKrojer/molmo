"""This script takes the checkpoint we trained on the simple color prediction tasks and runs inference on the train and validation splits.
For each image, and each of the 144 visual tokens in such an image:
a) we log the top5 nearest neighbor vocabularies from the LLM tokenizer
b) we check if the generated response is the actual color word
"""
import logging
import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import ColorMosaicDataset

log = logging.getLogger(__name__)

def decode_token(tokenizer, idx):
    """Decode a token and ensure it's a proper Unicode string."""
    token = tokenizer.decode([int(idx)])
    # Convert to actual characters by encoding and decoding through utf-8
    return token.encode('utf-8').decode('utf-8')

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_interpretability_plot(patch_statistics, output_path):
    """Create a bar plot showing top 20 token positions by interpretability percentage."""
    # patch_statistics is already sorted list of dicts with patch statistics
    if not patch_statistics:
        log.warning("No patch statistics available for plotting")
        return
    
    # Take top 20 (already sorted by accuracy)
    top_20 = patch_statistics[:20]
    
    # Prepare data for plotting
    patch_indices = [str(stats["patch_idx"]) for stats in top_20]
    accuracies = [stats.get("ground_truth_accuracy", 0) * 100 for stats in top_20]  # Convert to percentage
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(patch_indices)), accuracies, color='steelblue', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Token Position')
    plt.ylabel('Interpretability Percentage (%)')
    plt.title('Top 20 Token Positions by Interpretability\n(% of times a top-5 NN was a semantic match)')
    plt.xticks(range(len(patch_indices)), patch_indices, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, accuracy) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Interpretability plot saved to {output_path}")

def create_color_interpretability_plot(color_statistics, output_path):
    """Create a bar plot showing colors by interpretability percentage."""
    # color_statistics is already sorted list of dicts with color statistics
    if not color_statistics:
        log.warning("No color statistics available for plotting")
        return
    
    # Prepare data for plotting (take all colors, they should already be sorted)
    colors = [stats["color"] for stats in color_statistics]
    accuracies = [stats.get("ground_truth_accuracy", 0) * 100 for stats in color_statistics]  # Convert to percentage
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(colors)), accuracies, color='forestgreen', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Color')
    plt.ylabel('Interpretability Percentage (%)')
    plt.title('Color Interpretability\n(% of times a top-5 NN was a semantic match)')
    plt.xticks(range(len(colors)), colors, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, accuracy) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Color interpretability plot saved to {output_path}")

def process_split(model, preprocessor, dataset, num_images, prompt, use_n_token_only):
    """Process a dataset split and return results."""
    split_results = []
    
    # Initialize statistics tracking for patch positions
    patch_statistics = {}
    # Initialize statistics tracking for colors
    color_statistics = {}
    
    # Process each image
    for i in tqdm(range(num_images)):
        example_data = dataset.get(i, np.random)
        color_sequence = example_data["metadata"]["color_sequence"]
        
        # Create example with the provided prompt
        example = {
            "image": example_data["image"],
            "messages": {
                "messages": [prompt],
                "style": "none"
            }
        }

        # Preprocess example
        batch = preprocessor(example, rng=np.random)

        # Initialize image results
        image_results = {
            "image_idx": i,
            "true_color_sequence": color_sequence,
            "chunks": []
        }

        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                
                image_features, tokens_before_MLP = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                # Handle use_n_token_only properly - it can be int or list
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    image_features = image_features[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    image_features = image_features[:, :, use_n_token_only, :]
                image_results["feature_shape"] = list(image_features.shape)
                
                # Get token embeddings from the model
                token_embeddings = model.transformer.wte.embedding
                
                # Reshape image features to combine batch and chunks dimensions
                batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)
                
                # Normalize the embeddings for cosine similarity
                image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
                token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)
                
                # Compute cosine similarity for each patch
                similarity = torch.matmul(image_features_norm, token_embeddings_norm.T)
                
                # Get top-5 most similar tokens for each patch
                top_k = 5
                top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)
                
                # Clear intermediate tensors
                del similarity, image_features_norm, token_embeddings_norm
                clear_gpu_memory()

                # generated output - increase max_steps for multiple tokens
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()
                output = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                    max_steps=600,  # Increased for multiple color tokens
                    is_distributed=False
                )
                token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                decoded = preprocessor.tokenizer.decode(token_ids[0])
                image_results["generated_response"] = decoded

                # Clear GPU tensors
                del images_tensor, image_masks_tensor, image_features, input_ids, output
                clear_gpu_memory()

                # Tokenize the ground truth color sequence
                ground_truth_text = " ".join(color_sequence)
                ground_truth_tokens = preprocessor.tokenizer.encode(ground_truth_text)
                ground_truth_token_strings = [decode_token(preprocessor.tokenizer, token_id) for token_id in ground_truth_tokens]
                image_results["ground_truth_tokens"] = ground_truth_token_strings
                
                # Store results for each chunk and update statistics
                for chunk_idx in range(num_chunks):
                    chunk_results = {
                        "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                        "patches": []
                    }
                    
                    for patch_idx in range(patches_per_chunk):
                        patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]
                        
                        # Normalize patch tokens for comparison (lowercase)
                        patch_tokens_norm = [token.strip().lower() for token in patch_tokens]
                        
                        # Check if this patch matches its corresponding ground truth token position
                        ground_truth_match = False
                        corresponding_color = None
                        if patch_idx < len(ground_truth_token_strings) and patch_idx < len(color_sequence):
                            gt_token = ground_truth_token_strings[patch_idx]
                            corresponding_color = color_sequence[patch_idx].strip().lower()
                            gt_token_norm = gt_token.strip().lower()
                            # Use substring matching - check if the ground truth token appears as substring in any top5 token
                            ground_truth_match = any(
                                gt_token_norm in patch_token_norm
                                for patch_token_norm in patch_tokens_norm
                                if len(patch_token_norm.strip()) > 0 and len(gt_token_norm.strip()) > 0
                            )
                        
                        # Initialize patch statistics if not exists
                        if patch_idx not in patch_statistics:
                            patch_statistics[patch_idx] = {
                                "ground_truth_matches": 0,
                                "total_samples": 0
                            }
                        
                        # Update patch statistics
                        patch_statistics[patch_idx]["total_samples"] += 1
                        if ground_truth_match:
                            patch_statistics[patch_idx]["ground_truth_matches"] += 1
                        
                        # Initialize and update color statistics if we have a corresponding color
                        if corresponding_color is not None:
                            if corresponding_color not in color_statistics:
                                color_statistics[corresponding_color] = {
                                    "ground_truth_matches": 0,
                                    "total_samples": 0
                                }
                            color_statistics[corresponding_color]["total_samples"] += 1
                            if ground_truth_match:
                                color_statistics[corresponding_color]["ground_truth_matches"] += 1
                        
                        patch_results = {
                            "patch_idx": patch_idx,
                            "nearest_neighbors": [
                                {"token": token, "similarity": float(sim)}
                                for token, sim in zip(patch_tokens, patch_values)
                            ],
                            "ground_truth_match": ground_truth_match
                        }
                        chunk_results["patches"].append(patch_results)
                    
                    image_results["chunks"].append(chunk_results)

        split_results.append(image_results)
        
        # Periodically save results to avoid losing progress
        if (i + 1) % 50 == 0:
            temp_results = {
                "partial_results": True,
                "processed_images": i + 1,
                "images": split_results,
                "patch_statistics": patch_statistics,
                "color_statistics": color_statistics
            }
            temp_file = f"temp_results_{i + 1}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(temp_results, f, indent=2, ensure_ascii=False)
    
    return split_results, patch_statistics, color_statistics

def main():
    # Hardcoded parameters
    # checkpoint_path = "molmo_data/checkpoints/copy-prompt_mosaic-image/step1600-unsharded"
    checkpoint_path = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize/step3000-unsharded"
    prompt = "What is the sequence of colors in this grid of colors, read from left to right like a page?"
    # prompt = "Copy over the sequence of color words 1-by-1 from the previous context:"
    print(f"Prompt: {prompt}")

    # Setup results directory
    ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
    results_dir = Path("analysis_results/nearest_neighbors") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    # Create preprocessor
    if "hf:" in checkpoint_path:
        model_config = model.config
    else:
        model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # Override system prompt kind to avoid length conditioning
    model_config.system_prompt_kind = "style"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    use_n_token_only = model_config.vision_backbone.use_n_token_only

    # Initialize results dictionary
    all_results = {
        "checkpoint": checkpoint_path,
        "prompt": prompt,
        "dataset": "ColorMosaicDataset",
        "splits": {},
        "overall_patch_statistics": {},
        "overall_color_statistics": {}
    }

    try:
        # Process train split
        log.info("Processing train split...")
        train_dataset = ColorMosaicDataset(split="train", grid_size=24)
        num_train_images = min(10, len(train_dataset))
        train_results, train_patch_statistics, train_color_statistics = process_split(model, preprocessor, train_dataset, num_train_images, prompt, use_n_token_only)
        all_results["splits"]["train"] = {
            "num_images": num_train_images,
            "images": train_results,
            "patch_statistics": train_patch_statistics,
            "color_statistics": train_color_statistics
        }
        
        # Clear memory before processing validation split
        clear_gpu_memory()

        # Process validation split
        log.info("Processing validation split...")
        val_dataset = ColorMosaicDataset(split="validation", grid_size=24)
        num_val_images = min(10, len(val_dataset))
        val_results, val_patch_statistics, val_color_statistics = process_split(model, preprocessor, val_dataset, num_val_images, prompt, use_n_token_only)
        all_results["splits"]["validation"] = {
            "num_images": num_val_images,
            "images": val_results,
            "patch_statistics": val_patch_statistics,
            "color_statistics": val_color_statistics
        }
        
        # Combine patch statistics across splits
        combined_patch_statistics = {}
        for split_name, split_data in all_results["splits"].items():
            split_stats = split_data["patch_statistics"]
            for patch_idx, stats in split_stats.items():
                if patch_idx not in combined_patch_statistics:
                    combined_patch_statistics[patch_idx] = {
                        "ground_truth_matches": 0,
                        "total_samples": 0
                    }
                combined_patch_statistics[patch_idx]["ground_truth_matches"] += stats["ground_truth_matches"]
                combined_patch_statistics[patch_idx]["total_samples"] += stats["total_samples"]
        
        # Add accuracy percentages
        for patch_idx, stats in combined_patch_statistics.items():
            total = stats["total_samples"]
            if total > 0:
                stats["ground_truth_accuracy"] = stats["ground_truth_matches"] / total
        
        # Sort patch statistics by ground_truth_accuracy (descending) for easier analysis
        sorted_patch_statistics = []
        for patch_idx, stats in combined_patch_statistics.items():
            stats_with_idx = stats.copy()
            stats_with_idx["patch_idx"] = patch_idx
            sorted_patch_statistics.append(stats_with_idx)
        
        # Sort by ground_truth_accuracy descending
        sorted_patch_statistics.sort(key=lambda x: (x.get("ground_truth_accuracy", 0)), reverse=True)
        
        all_results["overall_patch_statistics"] = sorted_patch_statistics

        # Combine color statistics across splits
        combined_color_statistics = {}
        for split_name, split_data in all_results["splits"].items():
            split_stats = split_data["color_statistics"]
            for color, stats in split_stats.items():
                if color not in combined_color_statistics:
                    combined_color_statistics[color] = {
                        "ground_truth_matches": 0,
                        "total_samples": 0
                    }
                combined_color_statistics[color]["ground_truth_matches"] += stats["ground_truth_matches"]
                combined_color_statistics[color]["total_samples"] += stats["total_samples"]
        
        # Add accuracy percentages
        for color, stats in combined_color_statistics.items():
            total = stats["total_samples"]
            if total > 0:
                stats["ground_truth_accuracy"] = stats["ground_truth_matches"] / total
        
        # Sort color statistics by ground_truth_accuracy (descending) for easier analysis
        sorted_color_statistics = []
        for color, stats in combined_color_statistics.items():
            stats_with_color = stats.copy()
            stats_with_color["color"] = color
            sorted_color_statistics.append(stats_with_color)
        
        # Sort by ground_truth_accuracy descending
        sorted_color_statistics.sort(key=lambda x: (x.get("ground_truth_accuracy", 0)), reverse=True)
        
        all_results["overall_color_statistics"] = sorted_color_statistics

    except Exception as e:
        log.error(f"Error during processing: {str(e)}")
        # Save partial results if there's an error
        output_file = results_dir / "nearest_neighbors_analysis_color_names_mosaic_24x24_partial.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        raise

    # Save final results
    output_file = results_dir / "nearest_neighbors_analysis_color_names_mosaic_24x24.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    log.info(f"Results saved to {output_file}")
    
    # Create and save interpretability plot
    plot_file = results_dir / "nearest_neighbors_analysis_color_names_mosaic_24x24_summary_plot.png"
    create_interpretability_plot(all_results["overall_patch_statistics"], plot_file)

    # Create and save color interpretability plot
    color_plot_file = results_dir / "nearest_neighbors_analysis_color_names_mosaic_24x24_color_plot.png"
    create_color_interpretability_plot(all_results["overall_color_statistics"], color_plot_file)

if __name__ == "__main__":
    main()
