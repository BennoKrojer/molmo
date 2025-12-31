#!/usr/bin/env python3
"""
Extract vision tokens across ALL LLM layers in Qwen2-VL.

Based on our investigation:
1. Qwen2-VL uses a merger that reduces 256 patches → ~64 tokens (depending on image size)
2. The <|image_pad|> tokens in input_ids mark vision token positions
3. Positions 0 to (num_image_pad - 1) in hidden states are vision tokens
4. We can also access the raw visual encoder output BEFORE LLM processing

This script extracts:
- Layer 0: Vision tokens after embedding (input to first transformer layer)
- Layers 1-28: Vision tokens as they evolve through the LLM
- Raw visual encoder: Output of model.visual() BEFORE LLM processing

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_all_layers.py \
        --num-images 5 --output-dir analysis_results/qwen2_vl/vision_tokens
"""

import argparse
import json
import math
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def find_image_token_positions(input_ids, image_pad_token_id=151655):
    """
    Find positions of vision tokens in the input sequence.
    
    In Qwen2-VL, vision tokens are marked by <|image_pad|> tokens (ID 151655).
    Returns the start and end indices of vision tokens.
    """
    # Find all positions with image_pad token
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    
    if len(input_ids.shape) == 2:
        input_ids = input_ids[0]  # Remove batch dim
    
    image_positions = np.where(input_ids == image_pad_token_id)[0]
    
    if len(image_positions) == 0:
        return None, None, 0
    
    start_idx = image_positions[0]
    end_idx = image_positions[-1] + 1  # exclusive end
    num_vision_tokens = len(image_positions)
    
    return start_idx, end_idx, num_vision_tokens


def extract_vision_tokens_from_hidden_states(hidden_states, vision_start, vision_end):
    """
    Extract vision token features from hidden states.
    
    Args:
        hidden_states: List of tensors [batch, seq_len, hidden_dim]
        vision_start: Start index of vision tokens
        vision_end: End index of vision tokens (exclusive)
    
    Returns:
        dict: layer_idx -> tensor [num_vision_tokens, hidden_dim]
    """
    vision_features = {}
    
    for layer_idx, hs in enumerate(hidden_states):
        # hs shape: [batch, seq_len, hidden_dim]
        # Extract vision tokens
        vision_tokens = hs[:, vision_start:vision_end, :].squeeze(0)  # [num_vision, hidden_dim]
        vision_features[layer_idx] = vision_tokens
    
    return vision_features


def extract_raw_visual_features(model, pixel_values, image_grid_thw):
    """
    Extract raw visual features from the vision encoder BEFORE LLM processing.
    
    This gives us access to the patch-level features before the merger.
    """
    with torch.no_grad():
        # The visual encoder takes hidden_states (pixel_values) and grid_thw
        visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
    
    return visual_output


def process_single_image(model, processor, image, prompt, device):
    """
    Process a single image and extract vision tokens from all layers.
    
    Returns:
        dict with:
        - 'vision_features': dict[layer_idx] -> tensor [num_vision, hidden_dim]
        - 'raw_visual_features': tensor [num_merged_tokens, hidden_dim] 
        - 'num_vision_tokens': int
        - 'vision_positions': (start, end)
    """
    # Prepare inputs
    text_with_image = f"<|image_pad|>{prompt}"
    
    model_inputs = processor(
        images=[image],
        text=text_with_image,
        return_tensors="pt"
    )
    model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in model_inputs.items()}
    
    # Find vision token positions
    input_ids = model_inputs['input_ids']
    vision_start, vision_end, num_vision_tokens = find_image_token_positions(input_ids)
    
    if num_vision_tokens == 0:
        raise ValueError("No vision tokens found in input sequence")
    
    # Get raw visual encoder output
    raw_visual_features = extract_raw_visual_features(
        model, 
        model_inputs['pixel_values'],
        model_inputs['image_grid_thw']
    )
    
    # Run forward pass with output_hidden_states
    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            outputs = model(**model_inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    # Extract vision tokens from each layer
    vision_features = extract_vision_tokens_from_hidden_states(
        hidden_states, vision_start, vision_end
    )
    
    return {
        'vision_features': vision_features,
        'raw_visual_features': raw_visual_features,
        'num_vision_tokens': num_vision_tokens,
        'vision_positions': (vision_start, vision_end),
        'hidden_dim': hidden_states[0].shape[-1],
        'num_layers': len(hidden_states)
    }


def test_extraction(model_name="Qwen/Qwen2-VL-7B-Instruct"):
    """Test the extraction pipeline with a synthetic image."""
    
    print("=" * 80)
    print("TESTING QWEN2-VL VISION TOKEN EXTRACTION")
    print("=" * 80)
    print()
    
    # Load model
    print("[1/4] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(f"  ✓ Model loaded on {device}")
    print()
    
    # Create test image
    print("[2/4] Creating test image...")
    image = Image.new('RGB', (224, 224), color='blue')
    print(f"  ✓ Test image: {image.size}")
    print()
    
    # Process image
    print("[3/4] Extracting vision tokens...")
    prompt = "Describe this image."
    
    result = process_single_image(model, processor, image, prompt, device)
    
    print(f"  ✓ Extracted vision tokens:")
    print(f"    Number of vision tokens: {result['num_vision_tokens']}")
    print(f"    Vision positions: {result['vision_positions']}")
    print(f"    Hidden dimension: {result['hidden_dim']}")
    print(f"    Number of layers: {result['num_layers']}")
    print(f"    Raw visual features shape: {result['raw_visual_features'].shape}")
    print()
    
    # Print per-layer stats
    print("[4/4] Per-layer vision token statistics:")
    print()
    print("  Layer |   Shape        | Mean Norm | Std Norm | Min    | Max")
    print("  " + "-" * 70)
    
    for layer_idx, features in result['vision_features'].items():
        # features: [num_vision, hidden_dim]
        norms = torch.norm(features, dim=-1)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        min_val = features.min().item()
        max_val = features.max().item()
        
        print(f"  {layer_idx:5d} | {str(features.shape):14s} | {mean_norm:9.4f} | {std_norm:8.4f} | {min_val:6.3f} | {max_val:6.3f}")
    
    print()
    print("=" * 80)
    print("✓ EXTRACTION TEST PASSED!")
    print("=" * 80)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract vision tokens across all LLM layers")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--num-images", type=int, default=5,
                       help="Number of images to process")
    parser.add_argument("--output-dir", type=str, default="analysis_results/qwen2_vl/vision_tokens",
                       help="Output directory for extracted features")
    parser.add_argument("--test-only", action="store_true",
                       help="Run extraction test only (no dataset processing)")
    parser.add_argument("--use-real-images", action="store_true",
                       help="Use real images from PixMoCap dataset")
    args = parser.parse_args()
    
    if args.test_only:
        test_extraction(args.model_name)
        return
    
    print("=" * 80)
    print("QWEN2-VL VISION TOKEN EXTRACTION (ALL LAYERS)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Number of images: {args.num_images}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Load model
    print("[1/3] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"  ✓ Model loaded on {device}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare images
    print("[2/3] Preparing images...")
    if args.use_real_images:
        # Use PixMoCap dataset
        from olmo.data.pixmo_datasets import PixMoCap
        dataset = PixMoCap(split="validation", mode="captions")
        print(f"  Using PixMoCap validation set")
    else:
        # Use synthetic images
        dataset = None
        print(f"  Using synthetic test images")
    print()
    
    # Process images
    print("[3/3] Processing images...")
    all_results = []
    prompt = "Describe this image in detail."
    
    for img_idx in tqdm(range(args.num_images), desc="Processing"):
        if dataset is not None:
            # Load real image
            example = dataset.get(img_idx, np.random)
            image = Image.open(example["image"]).convert('RGB')
            
            # Resize if too large
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            caption = ""
            if "message_list" in example and len(example["message_list"]) > 0:
                caption = example["message_list"][0].get("text", "")
        else:
            # Create synthetic image with different colors
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
            color = colors[img_idx % len(colors)]
            image = Image.new('RGB', (224, 224), color=color)
            caption = f"A solid {color} image"
        
        # Extract vision tokens
        result = process_single_image(model, processor, image, prompt, device)
        
        # Convert tensors to numpy for storage
        image_result = {
            'image_idx': int(img_idx),
            'caption': caption,
            'num_vision_tokens': int(result['num_vision_tokens']),
            'vision_positions': [int(x) for x in result['vision_positions']],
            'hidden_dim': int(result['hidden_dim']),
            'num_layers': int(result['num_layers']),
            'raw_visual_shape': [int(x) for x in result['raw_visual_features'].shape],
            # Store layer statistics instead of full features (too large)
            'layer_stats': {}
        }
        
        for layer_idx, features in result['vision_features'].items():
            norms = torch.norm(features, dim=-1)
            image_result['layer_stats'][int(layer_idx)] = {
                'mean_norm': float(norms.mean().item()),
                'std_norm': float(norms.std().item()),
                'min_val': float(features.min().item()),
                'max_val': float(features.max().item())
            }
        
        all_results.append(image_result)
        
        # Clear memory
        del result
        torch.cuda.empty_cache()
    
    # Save results summary
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'num_images': args.num_images,
            'results': all_results
        }, f, indent=2)
    
    print()
    print(f"✓ Results saved to {summary_file}")
    print("=" * 80)
    

if __name__ == "__main__":
    main()

