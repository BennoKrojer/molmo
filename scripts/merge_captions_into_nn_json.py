#!/usr/bin/env python3
"""
Merge generated captions from generated_captions.json into nearest_neighbors JSON format.

This allows the captioning evaluation script to work with the merged format.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def merge_captions(nn_json_path, captions_json_path, output_path=None):
    """Merge captions from generated_captions.json into nearest_neighbors JSON."""
    
    # Load nearest_neighbors JSON
    with open(nn_json_path, 'r') as f:
        nn_data = json.load(f)
    
    # Load generated captions JSON
    with open(captions_json_path, 'r') as f:
        captions_data = json.load(f)
    
    # Create a mapping from image_idx to generated caption
    caption_map = {}
    for caption_entry in captions_data.get('captions', []):
        image_idx = caption_entry.get('image_idx')
        generated_caption = caption_entry.get('generated_caption', '')
        if image_idx is not None:
            caption_map[image_idx] = generated_caption
    
    # Merge captions into nearest_neighbors structure
    if 'splits' in nn_data:
        for split_name, split_data in nn_data['splits'].items():
            if 'images' in split_data:
                for image in split_data['images']:
                    image_idx = image.get('image_idx')
                    if image_idx in caption_map:
                        image['generated_response'] = caption_map[image_idx]
    
    # Save merged data
    if output_path is None:
        output_path = nn_json_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nn_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged {len(caption_map)} captions into {output_path}")
    return len(caption_map)


def main():
    parser = argparse.ArgumentParser(description='Merge generated captions into nearest_neighbors JSON')
    parser.add_argument('--nn-json', type=str, required=True,
                       help='Path to nearest_neighbors JSON file')
    parser.add_argument('--captions-json', type=str, required=True,
                       help='Path to generated_captions.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: overwrite nn-json)')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original nn-json before merging')
    
    args = parser.parse_args()
    
    nn_json_path = Path(args.nn_json)
    captions_json_path = Path(args.captions_json)
    
    if not nn_json_path.exists():
        print(f"ERROR: Nearest neighbors JSON not found: {nn_json_path}")
        return 1
    
    if not captions_json_path.exists():
        print(f"ERROR: Captions JSON not found: {captions_json_path}")
        return 1
    
    # Create backup if requested
    if args.backup:
        backup_path = nn_json_path.with_suffix('.json.backup')
        import shutil
        shutil.copy2(nn_json_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Merge captions
    output_path = Path(args.output) if args.output else nn_json_path
    num_merged = merge_captions(nn_json_path, captions_json_path, output_path)
    
    print(f"Successfully merged {num_merged} captions")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

