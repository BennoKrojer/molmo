#!/usr/bin/env python3
"""
Extract cosine similarities for human-annotated instances from contextual NN files.

This script creates a compact JSON with the 360 human study instances and their
actual cosine similarity values from the contextual nearest neighbor analysis.
"""

import json
import ijson
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_contextual_nn_data(nn_file, target_images):
    """
    Load contextual NN data for specific images using streaming parser.

    Args:
        nn_file: Path to contextual NN JSON file
        target_images: Set of image indices we need

    Returns:
        Dict mapping (image_idx, patch_row, patch_col) -> list of neighbors
    """
    patch_data = {}

    with open(nn_file, 'rb') as f:
        parser = ijson.items(f, 'results.item')

        for image_obj in parser:
            image_idx = image_obj.get('image_idx')
            if image_idx not in target_images:
                continue

            for chunk in image_obj.get('chunks', []):
                for patch in chunk.get('patches', []):
                    patch_row = patch.get('patch_row')
                    patch_col = patch.get('patch_col')
                    neighbors = patch.get('nearest_contextual_neighbors', [])

                    key = (image_idx, patch_row, patch_col)
                    patch_data[key] = neighbors

    return patch_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract similarities for human study instances')
    parser.add_argument('--contextual-data', type=str,
                       default='interp_data_contextual/data.json',
                       help='Path to contextual human study data')
    parser.add_argument('--nn-base-dir', type=str,
                       default='../analysis_results/contextual_nearest_neighbors',
                       help='Base directory for contextual NN results')
    parser.add_argument('--output', type=str,
                       default='human_study_with_similarities.json',
                       help='Output JSON file')

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    contextual_data_path = Path(args.contextual_data)
    if not contextual_data_path.is_absolute():
        contextual_data_path = script_dir / contextual_data_path

    nn_base_dir = Path(args.nn_base_dir)
    if not nn_base_dir.is_absolute():
        nn_base_dir = script_dir / nn_base_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    # Load human study contextual data
    print(f"Loading human study data from {contextual_data_path}...")
    with open(contextual_data_path, 'r') as f:
        human_data = json.load(f)
    print(f"Found {len(human_data)} instances")

    # Group instances by model for efficient loading
    instances_by_model = defaultdict(list)
    for instance in human_data:
        model = instance['model']
        instances_by_model[model].append(instance)

    print(f"Instances span {len(instances_by_model)} models")

    # Process each model
    results = []

    for model, instances in tqdm(instances_by_model.items(), desc="Processing models"):
        # Determine NN file path
        # Model name format: train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_step12000-unsharded
        model_dir = nn_base_dir / f"{model}_step12000-unsharded"
        if not model_dir.exists():
            model_dir = nn_base_dir / model

        if not model_dir.exists():
            print(f"WARNING: Model directory not found: {model_dir}")
            continue

        # Get visual layer (should be 0 for all based on how data was generated)
        visual_layer = instances[0].get('visual_layer', 0)

        # Find the allLayers file for this visual layer
        nn_file = model_dir / f"contextual_neighbors_visual{visual_layer}_allLayers.json"
        if not nn_file.exists():
            # Try with _multi-gpu suffix
            nn_file = model_dir / f"contextual_neighbors_visual{visual_layer}_allLayers_multi-gpu.json"

        if not nn_file.exists():
            print(f"WARNING: NN file not found: {nn_file}")
            continue

        # Get target images for this model
        target_images = set(inst['index'] for inst in instances)

        # Load NN data for these images
        print(f"  Loading {nn_file.name} for {len(target_images)} images...")
        patch_data = load_contextual_nn_data(nn_file, target_images)
        print(f"  Found {len(patch_data)} patches")

        # Extract similarities for each instance
        for instance in instances:
            image_idx = instance['index']
            patch_row = instance['patch_row']
            patch_col = instance['patch_col']
            target_layer = instance['layer']

            key = (image_idx, patch_row, patch_col)
            all_neighbors = patch_data.get(key, [])

            if not all_neighbors:
                print(f"  WARNING: No neighbors for {key}")
                continue

            # Filter neighbors by contextual layer
            layer_neighbors = [n for n in all_neighbors if n.get('contextual_layer') == target_layer]

            if not layer_neighbors:
                print(f"  WARNING: No neighbors at layer {target_layer} for {key}")
                # Fall back to all neighbors (they should all be same layer in allLayers file)
                layer_neighbors = all_neighbors[:5]

            # Take top 5
            top5_neighbors = layer_neighbors[:5]

            # Create result entry
            result = {
                'instance_id': instance['id'],
                'model': model,
                'image_idx': image_idx,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'layer': target_layer,
                'visual_layer': visual_layer,
                'image_url': instance.get('image_url'),
                'caption': instance.get('caption'),
                'neighbors': [
                    {
                        'rank': i + 1,
                        'token_str': n.get('token_str'),
                        'similarity': float(n.get('similarity')) if n.get('similarity') is not None else None,
                        'source_caption': n.get('caption'),
                        'position': n.get('position'),
                    }
                    for i, n in enumerate(top5_neighbors)
                ]
            }
            results.append(result)

    # Save results
    output_data = {
        'description': 'Cosine similarities for human-annotated instances (LatentLens/contextual)',
        'total_instances': len(results),
        'instances': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(results)} instances to {output_path}")

    # Print summary
    print("\nSummary:")
    models_found = set(r['model'] for r in results)
    print(f"  Models: {len(models_found)}")
    layers = set(r['layer'] for r in results)
    print(f"  Layers used: {sorted(layers)}")

    # Show example
    if results:
        print("\nExample entry:")
        ex = results[0]
        print(f"  Instance: {ex['instance_id']}")
        print(f"  Layer: {ex['layer']}")
        print(f"  Top 5 neighbors:")
        for n in ex['neighbors']:
            print(f"    {n['rank']}. sim={n['similarity']:.4f}, token='{n['token_str'][:30]}...'")


if __name__ == '__main__':
    main()
