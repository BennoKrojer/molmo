#!/usr/bin/env python3
"""Fix unsorted results in allLayers_multi-gpu.json files.

The multi-gpu script saves results in rank order (0,4,8,12... then 1,5,9...) 
instead of image order (0,1,2,3...). This script fixes existing JSONs.
"""

import json
from pathlib import Path
from tqdm import tqdm

def main():
    base_dir = Path("analysis_results/contextual_nearest_neighbors")
    
    # Find all non-lite allLayers JSONs
    all_jsons = list(base_dir.rglob("*allLayers_multi-gpu.json"))
    non_lite = [f for f in all_jsons if "_lite" not in str(f)]
    
    print(f"Found {len(non_lite)} non-lite allLayers JSONs to check")
    
    fixed = 0
    for json_path in tqdm(non_lite, desc="Checking/fixing"):
        with open(json_path) as f:
            data = json.load(f)
        
        results = data.get('results', [])
        if len(results) < 2:
            continue
            
        # Check if unsorted (results[1] should be image_idx 1, not 4)
        if results[1].get('image_idx', 0) == 1:
            continue  # Already sorted
        
        # Sort by image_idx
        data['results'] = sorted(results, key=lambda r: r.get('image_idx', 0))
        
        # Write back
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        fixed += 1
    
    print(f"\nFixed {fixed} JSONs")
    print("Now regenerate lite versions with: python scripts/analysis/create_lite_jsons.py")

if __name__ == "__main__":
    main()

