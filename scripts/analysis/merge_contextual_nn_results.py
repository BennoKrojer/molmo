#!/usr/bin/env python3
"""
Merge contextual nearest neighbor results from two directories.
Combines results from an existing 100-image run with a supplementary run.
"""

import argparse
import json
from pathlib import Path


def merge_results(base_file: Path, supplement_file: Path, output_file: Path):
    """Merge two contextual NN result files."""

    with open(base_file) as f:
        base_data = json.load(f)

    with open(supplement_file) as f:
        supplement_data = json.load(f)

    # Get existing image indices
    base_indices = set(r['image_idx'] for r in base_data['results'])
    supplement_indices = set(r['image_idx'] for r in supplement_data['results'])

    # Check for overlaps
    overlap = base_indices & supplement_indices
    if overlap:
        print(f"  Warning: {len(overlap)} overlapping indices, keeping base version")

    # Merge results (base takes precedence for overlaps)
    merged_results = list(base_data['results'])
    for result in supplement_data['results']:
        if result['image_idx'] not in base_indices:
            merged_results.append(result)

    # Sort by image index
    merged_results.sort(key=lambda x: x['image_idx'])

    # Update metadata
    merged_data = base_data.copy()
    merged_data['results'] = merged_results
    merged_data['num_images'] = len(merged_results)
    merged_data['image_indices'] = [r['image_idx'] for r in merged_results]

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    return len(base_data['results']), len(supplement_data['results']), len(merged_results)


def main():
    parser = argparse.ArgumentParser(description='Merge contextual NN results')
    parser.add_argument('--base-dir', required=True, help='Directory with base (100-image) results')
    parser.add_argument('--supplement-dir', required=True, help='Directory with supplementary results')
    parser.add_argument('--output-dir', help='Output directory (defaults to base-dir)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    supplement_dir = Path(args.supplement_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir

    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return

    if not supplement_dir.exists():
        print(f"Error: Supplement directory not found: {supplement_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all visual layer files in supplement directory
    supplement_files = list(supplement_dir.glob("contextual_neighbors_visual*_allLayers.json"))

    if not supplement_files:
        print(f"No supplement files found in {supplement_dir}")
        return

    print(f"Merging {len(supplement_files)} files...")

    for supplement_file in sorted(supplement_files):
        base_file = base_dir / supplement_file.name
        output_file = output_dir / supplement_file.name

        if not base_file.exists():
            print(f"  Skipping {supplement_file.name}: no base file")
            continue

        base_count, supp_count, merged_count = merge_results(base_file, supplement_file, output_file)
        print(f"  {supplement_file.name}: {base_count} + {supp_count} = {merged_count}")

    print("Done!")


if __name__ == "__main__":
    main()
