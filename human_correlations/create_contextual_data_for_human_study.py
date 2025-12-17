#!/usr/bin/env python3
"""
Create data.json for sentence-level human study using contextual nearest neighbors.

This script:
1. Loads the existing token-level human study data (interp_data_nn/data.json)
2. For each instance, randomly selects a layer
3. Finds the corresponding contextual nearest neighbors
4. Creates sentences with highlighted tokens as candidates
5. Outputs a similar JSON format with an additional "layer" field
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_model_name(model_name):
    """Parse model name to extract LLM and encoder."""
    # Format: train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_step12000-unsharded
    # Special case: seed10
    if 'seed10' in model_name:
        pattern = r'train_mlp-only_pixmo_cap_resize_(.+?)_(.+?)_seed10_step12000-unsharded'
    else:
        pattern = r'train_mlp-only_pixmo_cap_resize_(.+?)_(.+?)_step12000-unsharded'
    
    match = re.match(pattern, model_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_available_layers(contextual_dir, model_name, visual_layer):
    """
    Detect available contextual layers for a given model.
    
    Returns:
        List of available layer numbers
    """
    model_dir = Path(contextual_dir) / model_name
    
    if not model_dir.exists():
        return []
    
    # Find all contextual neighbor files
    pattern = f"contextual_neighbors_visual{visual_layer}_contextual*.json"
    available_layers = []
    
    for file_path in model_dir.glob(pattern):
        # Extract layer number from filename
        # Format: contextual_neighbors_visual0_contextual16_multi-gpu.json
        match = re.search(r'contextual(\d+)_multi-gpu\.json', file_path.name)
        if match:
            layer_num = int(match.group(1))
            available_layers.append(layer_num)
    
    return sorted(available_layers)


def load_contextual_data(contextual_dir, model_name, visual_layer, contextual_layer):
    """
    Load contextual nearest neighbors data.
    
    Returns:
        Dict mapping (image_idx, patch_row, patch_col) -> list of neighbors
        
    Raises:
        FileNotFoundError: If the contextual data file doesn't exist
    """
    model_dir = Path(contextual_dir) / model_name
    contextual_file = model_dir / f"contextual_neighbors_visual{visual_layer}_contextual{contextual_layer}_multi-gpu.json"
    
    if not contextual_file.exists():
        raise FileNotFoundError(f"Contextual data file not found: {contextual_file}")
    
    # Load and index by patch position
    patch_data = {}
    
    print(f"Loading {contextual_file}...")
    with open(contextual_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"  Found {len(results)} images")
    
    for entry in tqdm(results, desc=f"Processing {model_name} layer {contextual_layer}"):
        image_idx = entry.get('image_idx')
        if image_idx is None:
            continue
        
        for chunk in entry.get('chunks', []):
            for patch in chunk.get('patches', []):
                patch_row = patch.get('patch_row')
                patch_col = patch.get('patch_col')
                neighbors = patch.get('nearest_contextual_neighbors', [])
                
                if patch_row is not None and patch_col is not None:
                    key = (image_idx, patch_row, patch_col)
                    patch_data[key] = neighbors
    
    print(f"  Indexed {len(patch_data)} patches")
    return patch_data


def create_sentence_with_subword(caption, token_str):
    """
    Create a candidate as a tuple of (sentence, subword_token).
    
    Returns:
        Tuple of (caption_string, token_str)
        
    This avoids the complexity of trying to highlight tokens inline,
    especially for edge cases like empty strings, whitespace, punctuation, etc.
    """
    return (caption, token_str)


def create_sentence_candidates(neighbors, num_candidates=5):
    """
    Create sentence candidates as (sentence, subword) tuples.
    
    Args:
        neighbors: List of nearest neighbor dicts with caption, position, token_str
        num_candidates: Number of candidates to return
    
    Returns:
        List of (caption, token_str) tuples
        
    Raises:
        ValueError: If there aren't enough neighbors
    """
    if len(neighbors) < num_candidates:
        raise ValueError(f"Not enough neighbors: got {len(neighbors)}, need {num_candidates}")
    
    candidates = []
    
    for i in range(num_candidates):
        neighbor = neighbors[i]
        caption = neighbor.get('caption', '')
        token_str = neighbor.get('token_str', '')
        
        if not caption:
            raise ValueError(f"Neighbor {i} has empty caption: {neighbor}")
        
        # Create tuple of (sentence, subword)
        candidate = create_sentence_with_subword(caption, token_str)
        candidates.append(candidate)
    
    return candidates


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sentence-level human study data')
    parser.add_argument('--token-data', type=str,
                       default='interp_data_nn/data.json',
                       help='Path to token-level human study data.json')
    parser.add_argument('--contextual-dir', type=str,
                       default='../analysis_results/contextual_nearest_neighbors',
                       help='Directory containing contextual nearest neighbors')
    parser.add_argument('--output', type=str,
                       default='interp_data_contextual/data.json',
                       help='Output file for sentence-level data')
    parser.add_argument('--visual-layer', type=int, default=0,
                       help='Visual layer (default: 0)')
    parser.add_argument('--contextual-layers', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16, 24, 30, 31],
                       help='Contextual layers to randomly sample from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for layer selection')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    token_data_path = Path(args.token_data)
    if not token_data_path.is_absolute():
        token_data_path = script_dir / token_data_path
    
    contextual_dir = Path(args.contextual_dir)
    if not contextual_dir.is_absolute():
        contextual_dir = script_dir / contextual_dir
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    # Set random seed
    random.seed(args.seed)
    
    # Load token-level data
    print(f"Loading token-level data from {token_data_path}...")
    with open(token_data_path, 'r') as f:
        token_data = json.load(f)
    
    print(f"Loaded {len(token_data)} instances")
    
    # Detect available layers for each unique model (pre-scan)
    unique_models = set(instance.get('model') for instance in token_data if instance.get('model'))
    print(f"Found {len(unique_models)} unique models")
    
    model_available_layers = {}
    for model_name in unique_models:
        available = get_available_layers(contextual_dir, model_name, args.visual_layer)
        model_available_layers[model_name] = available
        print(f"  {model_name}: {len(available)} layers available {available}")
    
    # Cache for loaded contextual data
    contextual_data_cache = {}
    
    output_data = []
    
    # Process instances in their original order
    print(f"\nProcessing {len(token_data)} instances in original order...")
    for instance in tqdm(token_data, desc="Processing instances"):
        model_name = instance.get('model')
        
        if not model_name:
            raise ValueError(f"Instance {instance.get('instance_id')} has no model field")
        
        # Get available layers for this model
        available_layers = model_available_layers.get(model_name, [])
        
        if not available_layers:
            raise ValueError(f"No contextual layers available for model {model_name}. Expected to find files in {contextual_dir / model_name}")
        
        # Randomly select a contextual layer from available ones
        contextual_layer = random.choice(available_layers)
        
        # Load contextual data if not already cached
        cache_key = (model_name, args.visual_layer, contextual_layer)
        if cache_key not in contextual_data_cache:
            contextual_data = load_contextual_data(
                contextual_dir,
                model_name,
                args.visual_layer,
                contextual_layer
            )
            contextual_data_cache[cache_key] = contextual_data
        else:
            contextual_data = contextual_data_cache[cache_key]
        
        # Find matching patch
        image_idx = instance.get('index')
        patch_row = instance.get('patch_row')
        patch_col = instance.get('patch_col')
        
        patch_key = (image_idx, patch_row, patch_col)
        neighbors = contextual_data.get(patch_key)
        
        if neighbors is None:
            raise ValueError(f"No neighbors found for patch {patch_key} (image_idx={image_idx}, row={patch_row}, col={patch_col}) in model {model_name} layer {contextual_layer}")
        
        # Create sentence candidates as (sentence, subword) tuples
        sentence_candidates = create_sentence_candidates(neighbors, num_candidates=5)
        
        # Create new instance with layer info and sentence candidates
        new_instance = {
            **instance,  # Copy all original fields
            'layer': contextual_layer,
            'visual_layer': args.visual_layer,
            # Replace candidates with sentence candidates (list of tuples)
            'candidates': sentence_candidates,
            # Keep original token candidates for reference
            'original_token_candidates': instance.get('candidates', [])
        }
        
        output_data.append(new_instance)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(output_data)} instances to {output_path}")
    
    # Print statistics
    layer_counts = defaultdict(int)
    for instance in output_data:
        layer_counts[instance['layer']] += 1
    
    print("\nLayer distribution:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}: {layer_counts[layer]} instances")


if __name__ == '__main__':
    main()

