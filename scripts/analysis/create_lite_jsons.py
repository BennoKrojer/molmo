#!/usr/bin/env python3
"""
Create "lite" versions of analysis JSONs with only the first N images.
Uses the SAME scanning logic as the unified viewer to avoid processing junk.
"""

import json
import sys
from pathlib import Path
import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analysis.create_unified_viewer import (
    scan_analysis_results,
    get_checkpoint_name,
    LLMS,
    VISION_ENCODERS
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def create_lite_nn_json(input_path: Path, output_path: Path, num_images: int, split: str = "validation"):
    """Create lite version of nearest neighbors JSON."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # NN has splits structure
    for split_name in data.get("splits", {}).keys():
        data["splits"][split_name]["images"] = data["splits"][split_name]["images"][:num_images]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_lite_logitlens_json(input_path: Path, output_path: Path, num_images: int):
    """Create lite version of logit lens JSON."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    data["results"] = data["results"][:num_images]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_lite_contextual_json(input_path: Path, output_path: Path, num_images: int):
    """Create lite version of contextual NN JSON."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    data["results"] = data["results"][:num_images]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def process_single_file(file_type: str, layer: int, input_path: Path, output_path: Path, 
                       num_images: int, split: str):
    """Worker function to process a single file. Returns result dict."""
    # Skip if output already exists
    if output_path.exists():
        return {
            'file_type': file_type,
            'layer': layer,
            'skipped': True
        }
    
    # Process based on type
    if file_type == "NN":
        create_lite_nn_json(input_path, output_path, num_images, split)
    elif file_type == "Logit Lens":
        create_lite_logitlens_json(input_path, output_path, num_images)
    elif file_type == "Contextual NN":
        create_lite_contextual_json(input_path, output_path, num_images)
    
    return {
        'file_type': file_type,
        'layer': layer,
        'skipped': False
    }

def main():
    parser = argparse.ArgumentParser(description="Create lite JSON files with only first N images")
    parser.add_argument("--num-images", type=int, default=10,
                       help="Number of images to keep (default: 10)")
    parser.add_argument("--suffix", type=str, default="_lite10",
                       help="Suffix to add to lite directories (default: _lite10)")
    parser.add_argument("--split", type=str, default="validation",
                       help="Which split to use (default: validation)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                       help=f"Number of parallel workers (default: {multiprocessing.cpu_count()} CPUs)")
    
    args = parser.parse_args()
    
    log.info("=" * 70)
    log.info("STEP 1: Scanning for JSONs (using unified viewer logic)")
    log.info("=" * 70)
    log.info(f"Models to scan: {len(LLMS)} LLMs × {len(VISION_ENCODERS)} VEs = {len(LLMS) * len(VISION_ENCODERS)} total\n")
    
    # FIRST: Collect all files to process (don't process yet!)
    files_to_process = []  # List of (type, layer, input_path, output_path)
    
    for llm in LLMS:
        for ve in VISION_ENCODERS:
            checkpoint_name = get_checkpoint_name(llm, ve)
            
            # Scan using the SAME logic as unified viewer
            analysis_results = scan_analysis_results(checkpoint_name)
            
            # Collect NN files
            for layer, json_path in analysis_results["nn"].items():
                output_dir = json_path.parent.parent / f"{json_path.parent.name}{args.suffix}"
                output_file = output_dir / json_path.name
                files_to_process.append(("NN", layer, json_path, output_file, checkpoint_name))
            
            # Collect Logit Lens files
            for layer, json_path in analysis_results["logitlens"].items():
                output_dir = json_path.parent.parent / f"{json_path.parent.name}{args.suffix}"
                output_file = output_dir / json_path.name
                files_to_process.append(("Logit Lens", layer, json_path, output_file, checkpoint_name))
            
            # Collect Contextual NN files
            for layer, json_path in analysis_results["contextual"].items():
                output_dir = json_path.parent.parent / f"{json_path.parent.name}{args.suffix}"
                output_file = output_dir / json_path.name
                files_to_process.append(("Contextual NN", layer, json_path, output_file, checkpoint_name))
    
    # Show summary
    nn_count = sum(1 for x in files_to_process if x[0] == "NN")
    ll_count = sum(1 for x in files_to_process if x[0] == "Logit Lens")
    ctx_count = sum(1 for x in files_to_process if x[0] == "Contextual NN")
    
    log.info("=" * 70)
    log.info("SUMMARY: Files to process")
    log.info("=" * 70)
    log.info(f"  Nearest Neighbors:      {nn_count:3d} JSONs")
    log.info(f"  Logit Lens:             {ll_count:3d} JSONs")
    log.info(f"  Contextual NN:          {ctx_count:3d} JSONs")
    log.info(f"  {'─' * 40}")
    log.info(f"  TOTAL:                  {len(files_to_process):3d} JSONs")
    log.info(f"\nSettings:")
    log.info(f"  Images to keep:         {args.num_images}")
    log.info(f"  Split:                  {args.split}")
    log.info(f"  Output suffix:          {args.suffix}")
    log.info(f"  Parallel workers:       {args.workers}")
    log.info("")
    
    # NOW process with progress bar
    log.info("=" * 70)
    log.info("STEP 2: Processing JSONs (parallel)")
    log.info("=" * 70)
    
    skipped_count = 0
    
    # Submit all tasks to the process pool
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        future_to_file = {}
        for file_type, layer, input_path, output_path, checkpoint_name in files_to_process:
            future = executor.submit(
                process_single_file,
                file_type, layer, input_path, output_path, 
                args.num_images, args.split
            )
            future_to_file[future] = (file_type, layer)
        
        # Process completed tasks with progress bar
        with tqdm(total=len(files_to_process), desc="Processing JSONs", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_type, layer = future_to_file[future]
                result = future.result()
                
                # Update progress bar
                if result['skipped']:
                    pbar.set_description(f"{file_type} L{layer:02d} (skipped)")
                    skipped_count += 1
                else:
                    pbar.set_description(f"{file_type} L{layer:02d}")
                
                pbar.update(1)
    
    log.info("\n" + "=" * 70)
    log.info(f"✅ Done! Processed {len(files_to_process) - skipped_count} JSON files (skipped {skipped_count} existing)")
    log.info(f"📦 Lite versions saved with suffix: {args.suffix}")
    log.info("\nTo use lite versions, run:")
    log.info(f"  python scripts/analysis/create_unified_viewer.py \\")
    log.info(f"    --output-dir analysis_results/unified_viewer_lite \\")
    log.info(f"    --num-images {args.num_images} \\")
    log.info(f"    --lite-suffix {args.suffix}")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
