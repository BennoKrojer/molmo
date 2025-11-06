#!/usr/bin/env python3
"""
Compute interpretability heuristic for visual tokens.

Heuristic: A visual token is "interpretable" if its top cosine similarity 
to contextual embeddings (at layer 8) is significantly higher (>1.5x) than 
its top similarity to vocabulary embeddings.

This suggests the token aligns better with contextualized word meanings 
than raw vocabulary, indicating semantic interpretability.

Usage:
    python scripts/analysis/compute_interpretability_heuristic.py \
        --checkpoint-name train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336 \
        --nn-layer 0 \
        --contextual-layer 8 \
        --threshold 1.5
"""

import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm


def load_json_data(json_path):
    """Load JSON file and return data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_interpretability(checkpoint_name, nn_layer, contextual_layer, 
                            visual_layer, threshold, output_dir):
    """
    Compute interpretability heuristic for a single model.
    
    Args:
        checkpoint_name: e.g., "train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336"
        nn_layer: Layer for NN analysis (typically 0 for vision backbone)
        contextual_layer: Layer for contextual embeddings (typically 8)
        visual_layer: Visual layer for contextual NN (typically 0)
        threshold: Ratio threshold for interpretability (default 1.5)
        output_dir: Output directory for results
    """
    # Construct paths
    nn_dir = Path("analysis_results/nearest_neighbors") / f"{checkpoint_name}_step12000-unsharded"
    nn_file = nn_dir / f"nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{nn_layer}.json"
    
    contextual_dir = Path("analysis_results/contextual_nearest_neighbors") / f"{checkpoint_name}_step12000-unsharded"
    contextual_file = contextual_dir / f"contextual_neighbors_visual{visual_layer}_contextual{contextual_layer}_multi-gpu.json"
    
    # Check if files exist
    if not nn_file.exists():
        print(f"❌ NN file not found: {nn_file}")
        return False
    
    if not contextual_file.exists():
        print(f"❌ Contextual NN file not found: {contextual_file}")
        return False
    
    print(f"Loading NN data from: {nn_file.name}")
    t0 = time.time()
    nn_data = load_json_data(nn_file)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    print(f"Loading Contextual NN data from: {contextual_file.name}")
    t0 = time.time()
    contextual_data = load_json_data(contextual_file)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    # Extract results
    # NN data structure: {"splits": {"validation": {"images": [...]}}}
    # Contextual data structure: {"results": [...]}
    
    nn_images = None
    if "splits" in nn_data:
        # Use validation split
        if "validation" in nn_data["splits"]:
            nn_images = nn_data["splits"]["validation"]["images"]
        else:
            # Fallback to first split
            split_name = list(nn_data["splits"].keys())[0]
            nn_images = nn_data["splits"][split_name]["images"]
    else:
        print(f"❌ Unexpected NN data structure in {nn_file}")
        return False
    
    contextual_images = None
    if "results" in contextual_data:
        contextual_images = contextual_data["results"]
    else:
        print(f"❌ Unexpected contextual data structure in {contextual_file}")
        return False
    
    # Verify we have the same number of images
    num_images = min(len(nn_images), len(contextual_images))
    if len(nn_images) != len(contextual_images):
        print(f"⚠️  Warning: NN has {len(nn_images)} images, Contextual has {len(contextual_images)} images")
        print(f"   Using first {num_images} images")
    
    # Process each image
    print(f"\nProcessing {num_images} images...")
    results = []
    total_patches = 0
    interpretable_patches = 0
    
    t0 = time.time()
    for img_idx in tqdm(range(num_images), desc="Processing images", ncols=80):
        nn_img = nn_images[img_idx]
        contextual_img = contextual_images[img_idx]
        
        # Verify image indices match
        if nn_img.get("image_idx") != contextual_img.get("image_idx"):
            print(f"⚠️  Warning: Image index mismatch at position {img_idx}")
            print(f"   NN: {nn_img.get('image_idx')}, Contextual: {contextual_img.get('image_idx')}")
        
        image_result = {
            "image_idx": nn_img.get("image_idx", img_idx),
            "chunks": []
        }
        
        # Process each chunk
        nn_chunks = nn_img.get("chunks", [])
        contextual_chunks = contextual_img.get("chunks", [])
        
        for chunk_idx, (nn_chunk, contextual_chunk) in enumerate(zip(nn_chunks, contextual_chunks)):
            chunk_result = {
                "chunk_name": nn_chunk.get("chunk_name", f"Chunk {chunk_idx}"),
                "patches": []
            }
            
            nn_patches = nn_chunk.get("patches", [])
            contextual_patches = contextual_chunk.get("patches", [])
            
            # Process each patch
            for patch_idx, (nn_patch, contextual_patch) in enumerate(zip(nn_patches, contextual_patches)):
                # Get top similarities
                nn_neighbors = nn_patch.get("nearest_neighbors", [])
                contextual_neighbors = contextual_patch.get("nearest_contextual_neighbors", [])
                
                if not nn_neighbors or not contextual_neighbors:
                    # Skip if no neighbors
                    continue
                
                # Get top similarity scores
                nn_top_sim = nn_neighbors[0].get("similarity", 0.0)
                contextual_top_sim = contextual_neighbors[0].get("similarity", 0.0)
                
                # Compute ratio and interpretability
                ratio = contextual_top_sim / nn_top_sim if nn_top_sim > 0 else 0.0
                is_interpretable = 1 if ratio > threshold else 0
                
                total_patches += 1
                interpretable_patches += is_interpretable
                
                patch_result = {
                    "patch_idx": nn_patch.get("patch_idx", patch_idx),
                    "patch_row": nn_patch.get("patch_row"),
                    "patch_col": nn_patch.get("patch_col"),
                    "nn_top_similarity": float(nn_top_sim),
                    "contextual_top_similarity": float(contextual_top_sim),
                    "ratio": float(ratio),
                    "interpretable": is_interpretable
                }
                
                chunk_result["patches"].append(patch_result)
            
            image_result["chunks"].append(chunk_result)
        
        results.append(image_result)
    
    # Compute statistics
    processing_time = time.time() - t0
    interpretability_rate = interpretable_patches / total_patches if total_patches > 0 else 0.0
    
    print(f"\n✓ Processed {num_images} images, {total_patches} patches in {processing_time:.1f}s ({processing_time/num_images:.2f}s/img)")
    print(f"  Interpretable patches: {interpretable_patches}/{total_patches} ({interpretability_rate*100:.1f}%)")
    
    # Save results
    output_path = Path(output_dir) / f"{checkpoint_name}_step12000-unsharded"
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"interpretability_heuristic_nn{nn_layer}_contextual{contextual_layer}_threshold{threshold}.json"
    
    output_data = {
        "checkpoint_name": checkpoint_name,
        "nn_layer": nn_layer,
        "contextual_layer": contextual_layer,
        "visual_layer": visual_layer,
        "threshold": threshold,
        "num_images": num_images,
        "num_patches": total_patches,
        "interpretable_patches": interpretable_patches,
        "interpretability_rate": interpretability_rate,
        "results": results
    }
    
    print(f"\nSaving results...")
    t0 = time.time()
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved in {time.time()-t0:.1f}s")
    
    print(f"✓ Results saved to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compute interpretability heuristic for visual tokens")
    parser.add_argument("--checkpoint-name", type=str, required=True,
                       help="Checkpoint name (e.g., train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336)")
    parser.add_argument("--nn-layer", type=int, default=0,
                       help="Layer for NN analysis (default: 0 for vision backbone)")
    parser.add_argument("--contextual-layer", type=int, default=8,
                       help="Layer for contextual embeddings (default: 8)")
    parser.add_argument("--visual-layer", type=int, default=0,
                       help="Visual layer for contextual NN (default: 0)")
    parser.add_argument("--threshold", type=float, default=1.5,
                       help="Ratio threshold for interpretability (default: 1.5)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/interpretability_heuristic",
                       help="Output directory for results")
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"Computing Interpretability Heuristic")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint_name}")
    print(f"NN layer: {args.nn_layer}")
    print(f"Contextual layer: {args.contextual_layer}")
    print(f"Visual layer: {args.visual_layer}")
    print(f"Threshold: {args.threshold}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    success = compute_interpretability(
        args.checkpoint_name,
        args.nn_layer,
        args.contextual_layer,
        args.visual_layer,
        args.threshold,
        args.output_dir
    )
    
    if success:
        print(f"\n{'='*80}")
        print(f"✓ Analysis complete!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"❌ Analysis failed!")
        print(f"{'='*80}")
        exit(1)


if __name__ == "__main__":
    main()

