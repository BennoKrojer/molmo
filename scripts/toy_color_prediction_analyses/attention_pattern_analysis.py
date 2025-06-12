"""Analyze attention patterns in visual-language models, focusing on visual token attention patterns"""
import logging
import sys
import json
import pickle
from pathlib import Path
import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap, ColorImageDataset
from olmo.tokenizer import get_special_token_ids
import olmo.tokenizer as tok_module

log = logging.getLogger(__name__)

def get_cache_path(checkpoint_path, num_images, split_name="validation"):
    """Generate cache file path based on checkpoint and parameters."""
    ckpt_name = Path(checkpoint_path).name
    
    # Create cache directory relative to the script's location
    script_dir = Path(__file__).parent
    cache_dir = script_dir / "cache" / "attention_maps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{ckpt_name}_{split_name}_{num_images}images.pkl"
    return cache_file

def save_attention_cache(cache_path, text_to_visual_data, visual_to_visual_data, text_to_visual_per_layer, visual_to_visual_per_layer):
    """Save attention data to cache file."""
    cache_data = {
        'text_to_visual_data': text_to_visual_data,
        'visual_to_visual_data': visual_to_visual_data,
        'text_to_visual_per_layer': text_to_visual_per_layer,
        'visual_to_visual_per_layer': visual_to_visual_per_layer,
        'timestamp': datetime.datetime.now().isoformat(),
        'num_images': len(next(iter(text_to_visual_data.values()), [])),
        'num_visual_tokens': len(text_to_visual_data)
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    log.info(f"Saved attention data to cache: {cache_path}")

def load_attention_cache(cache_path):
    """Load attention data from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        log.info(f"Loaded cached attention data from: {cache_path}")
        log.info(f"Cache created: {cache_data['timestamp']}")
        log.info(f"Cached data: {cache_data['num_visual_tokens']} visual tokens, {cache_data['num_images']} images")
        
        # Handle both old and new cache formats
        text_to_visual_per_layer = cache_data.get('text_to_visual_per_layer', {})
        visual_to_visual_per_layer = cache_data.get('visual_to_visual_per_layer', {})
        
        return (cache_data['text_to_visual_data'], cache_data['visual_to_visual_data'], 
                text_to_visual_per_layer, visual_to_visual_per_layer)
    
    except (FileNotFoundError, pickle.PickleError, KeyError) as e:
        log.info(f"Could not load cache ({e}), will process from scratch")
        return None, None, None, None

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_attention_weights(model, input_ids, images, image_masks, image_input_idx):
    """Extract attention weights from model forward pass."""
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=None,
            attention_bias=None,
            response_mask=None,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            subsegment_ids=None,
            position_ids=None,
            return_logit_lenses=False,
            output_hidden_states=False,
            output_attentions=True,
            loss_masks=None
        )

    attention_weights = output.attn_maps_per_layer
    
    # Convert from [num_layers, batch, seq_len, seq_len] to list of [batch, seq_len, seq_len]
    if attention_weights is not None:
        attention_weights = [attention_weights[i] for i in range(attention_weights.size(0))]
    
    return attention_weights, output

def identify_token_positions(input_ids, image_input_idx, tokenizer):
    """Identify positions of visual and text tokens in the sequence."""
    # Get special token IDs
    special_tokens = get_special_token_ids(tokenizer)
    image_patch_token_id = special_tokens[tok_module.DEFAULT_IMAGE_PATCH_TOKEN]
    image_col_token_id = special_tokens[tok_module.DEFAULT_IM_COL_TOKEN]
    image_start_token_id = special_tokens[tok_module.DEFAULT_IM_START_TOKEN]
    image_end_token_id = special_tokens[tok_module.DEFAULT_IM_END_TOKEN]
    
    # Convert input_ids to numpy array for easier indexing
    if hasattr(input_ids, 'cpu'):
        input_ids = input_ids.cpu().numpy()
    input_ids = np.array(input_ids)
    
    # Use image_input_idx (most reliable)
    visual_positions = []
    if image_input_idx is not None:
        # Flatten nested list structure
        if hasattr(image_input_idx, 'tolist'):
            image_input_idx = image_input_idx.tolist()
        elif hasattr(image_input_idx, '__iter__'):
            image_input_idx = list(image_input_idx)
        else:
            image_input_idx = [image_input_idx]
            
        flat_positions = []
        for item in image_input_idx:
            if isinstance(item, (list, tuple)):
                flat_positions.extend(item)
            else:
                flat_positions.append(item)
                
        visual_positions = [int(idx) for idx in flat_positions if int(idx) >= 0]
    
    # Identify other special tokens
    special_token_positions = {
        'image_start': np.where(input_ids == image_start_token_id)[0].tolist(),
        'image_end': np.where(input_ids == image_end_token_id)[0].tolist(),
        'image_col': np.where(input_ids == image_col_token_id)[0].tolist(),
        'bos': np.where(input_ids == tokenizer.bos_token_id)[0].tolist() if tokenizer.bos_token_id else [],
        'eos': np.where(input_ids == tokenizer.eos_token_id)[0].tolist() if tokenizer.eos_token_id else [],
    }
    
    # Text positions are everything except visual tokens and special image tokens
    total_length = len(input_ids)
    all_positions = set(range(total_length))
    visual_positions_set = set(visual_positions)
    special_positions_set = set()
    for positions in special_token_positions.values():
        special_positions_set.update(positions)
    
    text_positions = list(all_positions - visual_positions_set - special_positions_set)
    
    return visual_positions, text_positions

def analyze_visual_token_attention(attention_weights, visual_positions, text_positions):
    """
    Analyze how much each visual token is attended to by text and other visual tokens.
    Returns dictionaries with attention scores for each visual token.
    """
    if not attention_weights or not visual_positions:
        return {}, {}, {}, {}
    
    # Initialize accumulators for each visual token
    text_to_visual_attention = {pos: [] for pos in visual_positions}
    visual_to_visual_attention = {pos: [] for pos in visual_positions}
    
    # Also track per-layer values for detailed analysis
    text_to_visual_per_layer = {pos: [] for pos in visual_positions}
    visual_to_visual_per_layer = {pos: [] for pos in visual_positions}
    
    # Process each layer
    valid_layers = 0
    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: [batch, heads, seq_len, seq_len]
        attn_np = attn.cpu().numpy().squeeze()  # Remove batch dimension if present
        
        if attn_np.ndim == 3:  # [heads, seq_len, seq_len]
            # Average across heads
            avg_attn = np.mean(attn_np, axis=0)
        else:  # Already averaged or single head
            avg_attn = attn_np
        
        # Check for NaN values in this layer
        if np.isnan(avg_attn).any():
            log.warning(f"Layer {layer_idx} contains NaN values, skipping this layer")
            continue
        
        valid_layers += 1
        
        # For each visual token, calculate how much it's attended to
        for visual_pos in visual_positions:
            # Text-to-visual attention (how much text tokens attend to this visual token)
            if text_positions:
                text_attention_to_this_visual = avg_attn[text_positions, visual_pos]
                # Additional NaN check for the extracted values
                if not np.isnan(text_attention_to_this_visual).any():
                    layer_avg = np.mean(text_attention_to_this_visual)
                    text_to_visual_attention[visual_pos].append(layer_avg)
                    text_to_visual_per_layer[visual_pos].append({
                        "layer": layer_idx,
                        "attention": float(layer_avg)
                    })
            
            # Visual-to-visual attention (how much other visual tokens attend to this visual token)
            other_visual_positions = [pos for pos in visual_positions if pos != visual_pos]
            if other_visual_positions:
                visual_attention_to_this_visual = avg_attn[other_visual_positions, visual_pos]
                # Additional NaN check for the extracted values
                if not np.isnan(visual_attention_to_this_visual).any():
                    layer_avg = np.mean(visual_attention_to_this_visual)
                    visual_to_visual_attention[visual_pos].append(layer_avg)
                    visual_to_visual_per_layer[visual_pos].append({
                        "layer": layer_idx,
                        "attention": float(layer_avg)
                    })
    
    log.info(f"Processed {valid_layers} valid layers out of {len(attention_weights)} total layers")
    
    # Average across layers for each visual token, using nanmean as extra safety
    text_to_visual_avg = {pos: np.nanmean(scores) if scores else 0.0 
                         for pos, scores in text_to_visual_attention.items()}
    visual_to_visual_avg = {pos: np.nanmean(scores) if scores else 0.0 
                           for pos, scores in visual_to_visual_attention.items()}
    
    return text_to_visual_avg, visual_to_visual_avg, text_to_visual_per_layer, visual_to_visual_per_layer

def process_images(model, preprocessor, dataset, num_images):
    """Process images and collect attention data for visual tokens."""
    all_text_to_visual = {}
    all_visual_to_visual = {}
    all_text_to_visual_per_layer = {}
    all_visual_to_visual_per_layer = {}
    
    for i in tqdm(range(num_images), desc="Processing images"):
        example_data = dataset.get(i, np.random)
        
        # Create example with prompt
        prompt = "Output the color shown in the image:"
        example = {
            "image": example_data["image"],
            "messages": {
                "messages": [prompt],
                "style": "none"
            }
        }

        # Preprocess example
        batch = preprocessor(example, rng=np.random)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                image_input_idx = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None

                # Extract attention weights
                attention_weights, output = extract_attention_weights(
                    model, input_ids, images_tensor, image_masks_tensor, image_input_idx
                )

                if attention_weights:
                    # Identify token positions
                    visual_positions, text_positions = identify_token_positions(
                        batch["input_tokens"], batch.get("image_input_idx"), preprocessor.tokenizer
                    )
                    
                    # Analyze attention for this image
                    text_to_visual, visual_to_visual, text_to_visual_per_layer, visual_to_visual_per_layer = analyze_visual_token_attention(
                        attention_weights, visual_positions, text_positions
                    )
                    
                    # Accumulate results (using relative positions within visual tokens)
                    for j, visual_pos in enumerate(visual_positions):
                        if j not in all_text_to_visual:
                            all_text_to_visual[j] = []
                            all_visual_to_visual[j] = []
                            all_text_to_visual_per_layer[j] = []
                            all_visual_to_visual_per_layer[j] = []
                        
                        all_text_to_visual[j].append(text_to_visual[visual_pos])
                        all_visual_to_visual[j].append(visual_to_visual[visual_pos])
                        all_text_to_visual_per_layer[j].append(text_to_visual_per_layer[visual_pos])
                        all_visual_to_visual_per_layer[j].append(visual_to_visual_per_layer[visual_pos])

                # Clear GPU memory
                del input_ids, images_tensor, image_masks_tensor, image_input_idx
                if 'output' in locals():
                    del output
                clear_gpu_memory()

    return all_text_to_visual, all_visual_to_visual, all_text_to_visual_per_layer, all_visual_to_visual_per_layer

def save_results_as_json(text_to_visual_data, visual_to_visual_data, text_to_visual_per_layer_data, visual_to_visual_per_layer_data, checkpoint_path, num_images, split_name):
    """Save attention analysis results as JSON files."""
    
    # Calculate average attention for each visual token across all images
    text_to_visual_avg = {token_idx: np.mean(scores) 
                         for token_idx, scores in text_to_visual_data.items() if scores}
    visual_to_visual_avg = {token_idx: np.mean(scores) 
                           for token_idx, scores in visual_to_visual_data.items() if scores}
    
    # Create ranked lists
    text_ranked = sorted(text_to_visual_avg.items(), key=lambda x: x[1], reverse=True)
    visual_ranked = sorted(visual_to_visual_avg.items(), key=lambda x: x[1], reverse=True)
    
    # Helper function to average per-layer data across images
    def average_per_layer_across_images(per_layer_data, token_idx):
        """Average per-layer attention values across all images for a given token."""
        if token_idx not in per_layer_data or not per_layer_data[token_idx]:
            return []
        
        # Collect all layer data across images
        layer_data = {}
        for image_layers in per_layer_data[token_idx]:
            if not image_layers:  # Skip empty lists
                continue
            for layer_info in image_layers:
                layer_idx = layer_info["layer"]
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = []
                layer_data[layer_idx].append(layer_info["attention"])
        
        if not layer_data:
            return []
        
        # Return just the averaged attention values per layer as a simple list
        averaged_values = []
        for layer_idx in sorted(layer_data.keys()):
            averaged_values.append(float(np.mean(layer_data[layer_idx])))
        
        return averaged_values
    
    # Prepare results dictionary
    results = {
        "checkpoint": checkpoint_path,
        "dataset": "ColorImageDataset",
        "split": split_name,
        "num_images_analyzed": len(next(iter(text_to_visual_data.values()), [])),
        "num_visual_tokens": len(text_to_visual_avg),
        "analysis": {
            "text_to_visual_attention": {
                "ranked_list": [
                    {
                        "rank": rank,
                        "visual_token_idx": int(token_idx),
                        "avg_attention": float(avg_attention),
                        "per_layer_attention": average_per_layer_across_images(text_to_visual_per_layer_data, token_idx)
                    }
                    for rank, (token_idx, avg_attention) in enumerate(text_ranked, 1)
                ],
                "statistics": {
                    "mean": float(np.mean(list(text_to_visual_avg.values()))),
                    "std": float(np.std(list(text_to_visual_avg.values()))),
                    "min": float(np.min(list(text_to_visual_avg.values()))),
                    "max": float(np.max(list(text_to_visual_avg.values())))
                }
            },
            "visual_to_visual_attention": {
                "ranked_list": [
                    {
                        "rank": rank,
                        "visual_token_idx": int(token_idx),
                        "avg_attention": float(avg_attention),
                        "per_layer_attention": average_per_layer_across_images(visual_to_visual_per_layer_data, token_idx)
                    }
                    for rank, (token_idx, avg_attention) in enumerate(visual_ranked, 1)
                ],
                "statistics": {
                    "mean": float(np.mean(list(visual_to_visual_avg.values()))),
                    "std": float(np.std(list(visual_to_visual_avg.values()))),
                    "min": float(np.min(list(visual_to_visual_avg.values()))),
                    "max": float(np.max(list(visual_to_visual_avg.values())))
                }
            }
        }
    }
    
    # Setup results directory
    ckpt_name = Path(checkpoint_path).name
    results_dir = Path("analysis_results/attention_patterns") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_file = results_dir / f"attention_analysis_{split_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log.info(f"Results saved to {output_file}")
    
    # Also print a summary to console
    print(f"\n{'='*80}")
    print("VISUAL TOKEN ATTENTION ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Analyzed {len(text_to_visual_avg)} visual tokens across {len(next(iter(text_to_visual_data.values()), []))} images")
    print(f"Results saved to: {output_file}")
    print(f"\nTop 5 visual tokens by TEXT attention:")
    for i, (token_idx, avg_attention) in enumerate(text_ranked[:5], 1):
        print(f"  {i}. Visual token {token_idx:3d}: {avg_attention:.6f}")
    print(f"\nTop 5 visual tokens by VISUAL attention:")
    for i, (token_idx, avg_attention) in enumerate(visual_ranked[:5], 1):
        print(f"  {i}. Visual token {token_idx:3d}: {avg_attention:.6f}")
    print(f"{'='*80}")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Hardcoded parameters
    checkpoint_path = "molmo_data/checkpoints/caption-prompt_1color-per-image/step1600-unsharded"
    num_images = 20  # Process 20 images for analysis
    split_name = "train"
    
    # Check for cached results first
    cache_path = get_cache_path(checkpoint_path, num_images, split_name)
    log.info(f"Checking for cached results at: {cache_path}")
    
    text_to_visual_data, visual_to_visual_data, text_to_visual_per_layer, visual_to_visual_per_layer = load_attention_cache(cache_path)
    
    if text_to_visual_data is not None and visual_to_visual_data is not None:
        log.info("Using cached attention data - skipping model loading and processing!")
    else:
        log.info("No valid cache found - processing from scratch")
        
        try:
            # Load model
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

            # Process images
            log.info("Processing images...")
            dataset = ColorImageDataset(split=split_name)
            num_images = min(num_images, len(dataset))
            
            text_to_visual_data, visual_to_visual_data, text_to_visual_per_layer, visual_to_visual_per_layer = process_images(
                model, preprocessor, dataset, num_images
            )
            
            # Save to cache
            save_attention_cache(cache_path, text_to_visual_data, visual_to_visual_data, text_to_visual_per_layer, visual_to_visual_per_layer)
            
            # Clean up GPU memory
            del model, preprocessor
            clear_gpu_memory()
            
        except Exception as e:
            log.error(f"Error during processing: {str(e)}")
            raise
    
    # Save results as JSON
    save_results_as_json(text_to_visual_data, visual_to_visual_data, text_to_visual_per_layer, visual_to_visual_per_layer, checkpoint_path, num_images, split_name)

if __name__ == "__main__":
    main()
