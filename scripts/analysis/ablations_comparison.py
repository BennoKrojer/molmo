"""
Compare ablation results to original model results.

This script compares nearest neighbor results between the original model and ablation variants,
focusing on metrics like overlap in top-k nearest neighbors.

Usage:
    python scripts/analysis/ablations_comparison.py \
        --original analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json \
        --ablation analysis_results/nearest_neighbors/ablations/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json \
        --top-k 5
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_json_file(filepath):
    """Load a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_topk_neighbors(patch_data, top_k=5):
    """Extract top-k nearest neighbor tokens from a patch."""
    neighbors = patch_data.get('nearest_neighbors', [])
    if len(neighbors) == 0:
        return set(), []
    # Get top-k tokens (they should already be sorted by similarity)
    topk_tokens = [nn['token'] for nn in neighbors[:top_k]]
    topk_similarities = [nn['similarity'] for nn in neighbors[:top_k]]
    return set(topk_tokens), topk_similarities  # Return tokens as set and similarities as list


def calculate_overlap(set1, set2):
    """Calculate overlap between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0  # Both empty, perfect overlap
    if len(set1) == 0 or len(set2) == 0:
        return 0.0  # One empty, no overlap
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    # Jaccard similarity (intersection over union)
    return intersection / union


def check_high_similarity(similarities, threshold=0.1):
    """Check if all similarities are above threshold."""
    if len(similarities) == 0:
        return False
    return all(sim > threshold for sim in similarities)


def compute_statistics(overlaps, overlap_counts):
    """Compute statistics from overlaps and overlap counts."""
    if len(overlaps) == 0:
        return None
    return {
        'num_patches': len(overlaps),
        'mean_overlap': np.mean(overlaps),
        'std_overlap': np.std(overlaps),
        'median_overlap': np.median(overlaps),
        'min_overlap': np.min(overlaps),
        'max_overlap': np.max(overlaps),
        'overlap_distribution': dict(overlap_counts),
        'overlaps': overlaps  # Keep raw overlaps for detailed analysis
    }


def compare_splits(original_data, ablation_data, top_k=5, split_name='validation', similarity_threshold=0.1):
    """Compare a specific split between original and ablation."""
    original_split = original_data['splits'].get(split_name, {})
    ablation_split = ablation_data['splits'].get(split_name, {})
    
    original_images = original_split.get('images', [])
    ablation_images = ablation_split.get('images', [])
    
    # Match images by index
    original_by_idx = {img['image_idx']: img for img in original_images}
    ablation_by_idx = {img['image_idx']: img for img in ablation_images}
    
    # Find common image indices
    common_indices = set(original_by_idx.keys()) & set(ablation_by_idx.keys())
    
    if len(common_indices) == 0:
        print(f"Warning: No common images found in {split_name} split")
        return None
    
    # Separate overlaps by similarity threshold
    overlaps_high_sim = []  # Both models have similarity > threshold
    overlaps_low_sim = []   # At least one model has similarity <= threshold
    overlaps_all = []  # All overlaps combined
    overlap_counts_high_sim = defaultdict(int)
    overlap_counts_low_sim = defaultdict(int)
    overlap_counts_all = defaultdict(int)
    
    for img_idx in common_indices:
        orig_img = original_by_idx[img_idx]
        abl_img = ablation_by_idx[img_idx]
        
        # Match chunks by name
        orig_chunks = {chunk['chunk_name']: chunk for chunk in orig_img.get('chunks', [])}
        abl_chunks = {chunk['chunk_name']: chunk for chunk in abl_img.get('chunks', [])}
        
        common_chunks = set(orig_chunks.keys()) & set(abl_chunks.keys())
        
        for chunk_name in common_chunks:
            orig_chunk = orig_chunks[chunk_name]
            abl_chunk = abl_chunks[chunk_name]
            
            # Match patches by patch_idx
            orig_patches = {patch['patch_idx']: patch for patch in orig_chunk.get('patches', [])}
            abl_patches = {patch['patch_idx']: patch for patch in abl_chunk.get('patches', [])}
            
            common_patches = set(orig_patches.keys()) & set(abl_patches.keys())
            
            for patch_idx in common_patches:
                orig_patch = orig_patches[patch_idx]
                abl_patch = abl_patches[patch_idx]
                
                # Extract top-k neighbors and similarities
                orig_topk, orig_sims = extract_topk_neighbors(orig_patch, top_k)
                abl_topk, abl_sims = extract_topk_neighbors(abl_patch, top_k)
                
                # Check if both have high similarity
                orig_high_sim = check_high_similarity(orig_sims, similarity_threshold)
                abl_high_sim = check_high_similarity(abl_sims, similarity_threshold)
                both_high_sim = orig_high_sim and abl_high_sim
                
                # Calculate overlap
                overlap = calculate_overlap(orig_topk, abl_topk)
                intersection_size = len(orig_topk & abl_topk)
                
                # Add to overall statistics
                overlaps_all.append(overlap)
                overlap_counts_all[intersection_size] += 1
                
                if both_high_sim:
                    overlaps_high_sim.append(overlap)
                    overlap_counts_high_sim[intersection_size] += 1
                else:
                    overlaps_low_sim.append(overlap)
                    overlap_counts_low_sim[intersection_size] += 1
    
    # Compute statistics for all groups
    overall_stats = compute_statistics(overlaps_all, overlap_counts_all)
    high_sim_stats = compute_statistics(overlaps_high_sim, overlap_counts_high_sim)
    low_sim_stats = compute_statistics(overlaps_low_sim, overlap_counts_low_sim)
    
    if overall_stats is None:
        return None
    
    result = {
        'split_name': split_name,
        'num_common_images': len(common_indices),
        'similarity_threshold': similarity_threshold,
        'total_patches_compared': len(overlaps_all),
    }
    
    # Always include overall statistics
    result['overall'] = overall_stats
    
    if high_sim_stats:
        result['high_similarity'] = high_sim_stats
    if low_sim_stats:
        result['low_similarity'] = low_sim_stats
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Compare ablation results to original model')
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original model results JSON file')
    parser.add_argument('--ablation', type=str, required=True,
                       help='Path to ablation model results JSON file')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top neighbors to compare (default: 5)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'validation'],
                       help='Splits to compare (default: train validation)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: auto-generated in analysis_results/ablations_comparison/)')
    parser.add_argument('--similarity-threshold', type=float, default=0.1,
                       help='Similarity threshold for distinguishing high/low similarity cases (default: 0.1)')
    
    args = parser.parse_args()
    
    # Extract model names from paths
    original_name = Path(args.original).parent.name
    ablation_name = Path(args.ablation).parent.name
    
    # Load JSON files
    print(f"Loading original results from: {args.original}")
    original_data = load_json_file(args.original)
    
    print(f"Loading ablation results from: {args.ablation}")
    ablation_data = load_json_file(args.ablation)
    
    # Get layer number for filename
    ablation_layer = ablation_data.get('llm_layer', 'unknown')
    
    # Generate default output path if not provided
    if args.output is None:
        # Save to analysis_results/ablations_comparison/ directory
        output_dir = Path('analysis_results/ablations_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        # Pattern: {ablation_name}_vs_original_layer{layer}.json
        output_filename = f"{ablation_name}_vs_original_layer{ablation_layer}.json"
        args.output = str(output_dir / output_filename)
    
    print(f"\nComparing:")
    print(f"  Original: {original_name}")
    print(f"  Ablation: {ablation_name}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print()
    
    # Compare each split
    results = {
        'original_model': original_name,
        'ablation_model': ablation_name,
        'top_k': args.top_k,
        'similarity_threshold': args.similarity_threshold,
        'original_layer': original_data.get('llm_layer', 'unknown'),
        'ablation_layer': ablation_data.get('llm_layer', 'unknown'),
        'splits': {}
    }
    
    for split_name in args.splits:
        print(f"Comparing {split_name} split...")
        split_results = compare_splits(original_data, ablation_data, args.top_k, split_name, args.similarity_threshold)
        
        if split_results is None:
            print(f"  Warning: Could not compare {split_name} split")
            continue
        
        results['splits'][split_name] = split_results
        
        print(f"  Common images: {split_results['num_common_images']}")
        print(f"  Total patches compared: {split_results['total_patches_compared']}")
        print()
        
        # Print overall statistics
        if 'overall' in split_results:
            overall = split_results['overall']
            print(f"  OVERALL (all patches):")
            print(f"    Patches: {overall['num_patches']}")
            print(f"    Mean overlap: {overall['mean_overlap']:.4f}")
            print(f"    Std overlap: {overall['std_overlap']:.4f}")
            print(f"    Median overlap: {overall['median_overlap']:.4f}")
            print(f"    Min overlap: {overall['min_overlap']:.4f}")
            print(f"    Max overlap: {overall['max_overlap']:.4f}")
            print(f"    Overlap distribution (intersection size): {overall['overlap_distribution']}")
            print()
        
        # Print high similarity statistics
        if 'high_similarity' in split_results:
            high_sim = split_results['high_similarity']
            print(f"  HIGH SIMILARITY (> {args.similarity_threshold} for both models):")
            print(f"    Patches: {high_sim['num_patches']}")
            print(f"    Mean overlap: {high_sim['mean_overlap']:.4f}")
            print(f"    Std overlap: {high_sim['std_overlap']:.4f}")
            print(f"    Median overlap: {high_sim['median_overlap']:.4f}")
            print(f"    Min overlap: {high_sim['min_overlap']:.4f}")
            print(f"    Max overlap: {high_sim['max_overlap']:.4f}")
            print(f"    Overlap distribution (intersection size): {high_sim['overlap_distribution']}")
            print()
        
        # Print low similarity statistics
        if 'low_similarity' in split_results:
            low_sim = split_results['low_similarity']
            print(f"  LOW SIMILARITY (<= {args.similarity_threshold} for at least one model):")
            print(f"    Patches: {low_sim['num_patches']}")
            print(f"    Mean overlap: {low_sim['mean_overlap']:.4f}")
            print(f"    Std overlap: {low_sim['std_overlap']:.4f}")
            print(f"    Median overlap: {low_sim['median_overlap']:.4f}")
            print(f"    Min overlap: {low_sim['min_overlap']:.4f}")
            print(f"    Max overlap: {low_sim['max_overlap']:.4f}")
            print(f"    Overlap distribution (intersection size): {low_sim['overlap_distribution']}")
            print()
    
    # Remove raw overlaps from output (too large)
    for split_name in results['splits']:
        split_data = results['splits'][split_name]
        if 'overall' in split_data and 'overlaps' in split_data['overall']:
            del split_data['overall']['overlaps']
        if 'high_similarity' in split_data and 'overlaps' in split_data['high_similarity']:
            del split_data['high_similarity']['overlaps']
        if 'low_similarity' in split_data and 'overlaps' in split_data['low_similarity']:
            del split_data['low_similarity']['overlaps']
    
    # Save results (always save now)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

