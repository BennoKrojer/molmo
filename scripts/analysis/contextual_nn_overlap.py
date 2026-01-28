"""
Compute overlap between original and ablation LatentLens (contextual NN) results.

This script computes two types of overlap:
1. Subword-level: Do both top-5 sets contain the same token_str (e.g., "dog")?
2. Phrase-level: Do both top-5 sets contain the same full phrase/caption (e.g., "look at brown dog")?

Usage:
    python scripts/analysis/contextual_nn_overlap.py \
        --original analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/contextual_neighbors_visual0_allLayers.json \
        --ablation analysis_results/contextual_nearest_neighbors/ablations/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10_step12000-unsharded/contextual_neighbors_visual0_allLayers.json \
        --top-k 5

    # Or run all ablations for a given visual layer:
    python scripts/analysis/contextual_nn_overlap.py --run-all --visual-layer 0
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
    """Extract top-k contextual nearest neighbor data from a patch.

    Returns:
        tokens: set of token_str values
        phrases: set of caption strings
        similarities: list of similarity values
    """
    neighbors = patch_data.get('nearest_contextual_neighbors', [])
    if len(neighbors) == 0:
        return set(), set(), []

    # Get top-k data
    topk = neighbors[:top_k]
    tokens = set(nn['token_str'] for nn in topk)
    phrases = set(nn['caption'] for nn in topk)
    similarities = [nn['similarity'] for nn in topk]

    return tokens, phrases, similarities


def calculate_jaccard(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union


def calculate_intersection_count(set1, set2):
    """Calculate raw intersection count."""
    return len(set1 & set2)


def check_high_similarity(similarities, threshold=0.1):
    """Check if all similarities are above threshold."""
    if len(similarities) == 0:
        return False
    return all(sim > threshold for sim in similarities)


def compare_contextual_nn(original_data, ablation_data, top_k=5, similarity_threshold=0.1):
    """Compare contextual NN results between original and ablation.

    Returns dict with:
        - subword_overlap_all: Jaccard overlap of token_str (all patches)
        - subword_overlap_high_sim: Jaccard overlap (high similarity patches only)
        - phrase_overlap_all: Jaccard overlap of captions (all patches)
        - phrase_overlap_high_sim: Jaccard overlap (high similarity patches only)
        - intersection_counts: distribution of intersection sizes
    """
    original_results = original_data.get('results', [])
    ablation_results = ablation_data.get('results', [])

    # Match by image_idx
    original_by_idx = {r['image_idx']: r for r in original_results}
    ablation_by_idx = {r['image_idx']: r for r in ablation_results}

    common_indices = set(original_by_idx.keys()) & set(ablation_by_idx.keys())

    if len(common_indices) == 0:
        print("Warning: No common images found")
        return None

    # Collect overlaps
    subword_overlaps_all = []
    subword_overlaps_high_sim = []
    phrase_overlaps_all = []
    phrase_overlaps_high_sim = []

    subword_intersection_counts_all = defaultdict(int)
    subword_intersection_counts_high_sim = defaultdict(int)
    phrase_intersection_counts_all = defaultdict(int)
    phrase_intersection_counts_high_sim = defaultdict(int)

    for img_idx in sorted(common_indices):
        orig_img = original_by_idx[img_idx]
        abl_img = ablation_by_idx[img_idx]

        # Match chunks by name
        orig_chunks = {c['chunk_name']: c for c in orig_img.get('chunks', [])}
        abl_chunks = {c['chunk_name']: c for c in abl_img.get('chunks', [])}

        common_chunks = set(orig_chunks.keys()) & set(abl_chunks.keys())

        for chunk_name in common_chunks:
            orig_chunk = orig_chunks[chunk_name]
            abl_chunk = abl_chunks[chunk_name]

            # Match patches by patch_idx
            orig_patches = {p['patch_idx']: p for p in orig_chunk.get('patches', [])}
            abl_patches = {p['patch_idx']: p for p in abl_chunk.get('patches', [])}

            common_patches = set(orig_patches.keys()) & set(abl_patches.keys())

            for patch_idx in common_patches:
                orig_patch = orig_patches[patch_idx]
                abl_patch = abl_patches[patch_idx]

                # Extract top-k data
                orig_tokens, orig_phrases, orig_sims = extract_topk_neighbors(orig_patch, top_k)
                abl_tokens, abl_phrases, abl_sims = extract_topk_neighbors(abl_patch, top_k)

                # Check if both have high similarity
                orig_high_sim = check_high_similarity(orig_sims, similarity_threshold)
                abl_high_sim = check_high_similarity(abl_sims, similarity_threshold)
                both_high_sim = orig_high_sim and abl_high_sim

                # Calculate overlaps
                subword_overlap = calculate_jaccard(orig_tokens, abl_tokens)
                phrase_overlap = calculate_jaccard(orig_phrases, abl_phrases)

                subword_intersection = calculate_intersection_count(orig_tokens, abl_tokens)
                phrase_intersection = calculate_intersection_count(orig_phrases, abl_phrases)

                # Store results
                subword_overlaps_all.append(subword_overlap)
                phrase_overlaps_all.append(phrase_overlap)
                subword_intersection_counts_all[subword_intersection] += 1
                phrase_intersection_counts_all[phrase_intersection] += 1

                if both_high_sim:
                    subword_overlaps_high_sim.append(subword_overlap)
                    phrase_overlaps_high_sim.append(phrase_overlap)
                    subword_intersection_counts_high_sim[subword_intersection] += 1
                    phrase_intersection_counts_high_sim[phrase_intersection] += 1

    def compute_stats(overlaps, intersection_counts):
        if len(overlaps) == 0:
            return None
        return {
            'num_patches': len(overlaps),
            'mean_overlap': float(np.mean(overlaps)),
            'std_overlap': float(np.std(overlaps)),
            'median_overlap': float(np.median(overlaps)),
            'intersection_distribution': dict(intersection_counts)
        }

    return {
        'num_common_images': len(common_indices),
        'similarity_threshold': similarity_threshold,
        'subword': {
            'all': compute_stats(subword_overlaps_all, subword_intersection_counts_all),
            'high_sim': compute_stats(subword_overlaps_high_sim, subword_intersection_counts_high_sim)
        },
        'phrase': {
            'all': compute_stats(phrase_overlaps_all, phrase_intersection_counts_all),
            'high_sim': compute_stats(phrase_overlaps_high_sim, phrase_intersection_counts_high_sim)
        }
    }


def get_ablation_name(ablation_path):
    """Extract ablation name from path."""
    name = Path(ablation_path).parent.name
    # Extract ablation type from checkpoint name
    if 'seed10' in name:
        return 'seed10'
    elif 'seed11' in name:
        return 'seed11'
    elif 'linear' in name:
        return 'linear'
    elif 'first-sentence' in name:
        return 'first-sentence'
    elif 'unfreeze' in name and 'topbottom' not in name:
        return 'unfreeze'
    elif 'earlier-vit-layers-6' in name:
        return 'earlier-vit-6'
    elif 'earlier-vit-layers-10' in name:
        return 'earlier-vit-10'
    elif 'topbottom' in name and 'unfreeze' in name:
        return 'topbottom-unfreeze'
    elif 'topbottom' in name:
        return 'topbottom'
    return name


def run_all_ablations(visual_layer, top_k=5, similarity_threshold=0.1):
    """Run overlap comparison for all ablations at a given visual layer."""
    base_dir = Path('analysis_results/contextual_nearest_neighbors')
    original_dir = base_dir / 'train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded'
    ablations_dir = base_dir / 'ablations'

    original_file = original_dir / f'contextual_neighbors_visual{visual_layer}_allLayers.json'

    if not original_file.exists():
        print(f"Error: Original file not found: {original_file}")
        return None

    print(f"Loading original: {original_file}")
    original_data = load_json_file(original_file)

    results = {}

    for ablation_dir in sorted(ablations_dir.iterdir()):
        if not ablation_dir.is_dir():
            continue

        ablation_file = ablation_dir / f'contextual_neighbors_visual{visual_layer}_allLayers.json'
        if not ablation_file.exists():
            print(f"  Skipping {ablation_dir.name}: no visual layer {visual_layer}")
            continue

        ablation_name = get_ablation_name(str(ablation_file))
        print(f"\nComparing with: {ablation_name}")

        ablation_data = load_json_file(ablation_file)
        comparison = compare_contextual_nn(original_data, ablation_data, top_k, similarity_threshold)

        if comparison:
            results[ablation_name] = comparison

            # Print summary
            sw_all = comparison['subword']['all']
            sw_high = comparison['subword']['high_sim']
            ph_all = comparison['phrase']['all']
            ph_high = comparison['phrase']['high_sim']

            print(f"  Images: {comparison['num_common_images']}")
            if sw_all:
                print(f"  Subword overlap (all): {sw_all['mean_overlap']*100:.1f}% ({sw_all['num_patches']} patches)")
            if sw_high:
                print(f"  Subword overlap (high-sim): {sw_high['mean_overlap']*100:.1f}% ({sw_high['num_patches']} patches)")
            if ph_all:
                print(f"  Phrase overlap (all): {ph_all['mean_overlap']*100:.1f}%")
            if ph_high:
                print(f"  Phrase overlap (high-sim): {ph_high['mean_overlap']*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute LatentLens NN overlap between original and ablation')
    parser.add_argument('--original', type=str, help='Path to original contextual NN JSON')
    parser.add_argument('--ablation', type=str, help='Path to ablation contextual NN JSON')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top neighbors to compare')
    parser.add_argument('--similarity-threshold', type=float, default=0.1,
                       help='Similarity threshold for high-sim filtering')
    parser.add_argument('--output', type=str, help='Output JSON path')
    parser.add_argument('--run-all', action='store_true',
                       help='Run all ablations for OLMo+ViT-L baseline')
    parser.add_argument('--visual-layer', type=int, default=0,
                       help='Visual layer to compare (used with --run-all)')

    args = parser.parse_args()

    if args.run_all:
        results = run_all_ablations(args.visual_layer, args.top_k, args.similarity_threshold)

        if results:
            # Save results
            output_dir = Path('analysis_results/ablations_comparison')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'contextual_nn_overlap_visual{args.visual_layer}.json'

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")

            # Print summary table
            print("\n" + "="*80)
            print(f"SUMMARY TABLE (Visual Layer {args.visual_layer})")
            print("="*80)
            print(f"{'Ablation':<20} {'Subword (all)':<15} {'Subword (high)':<15} {'Phrase (all)':<15} {'Phrase (high)':<15}")
            print("-"*80)
            for name, data in results.items():
                sw_all = data['subword']['all']['mean_overlap']*100 if data['subword']['all'] else 0
                sw_high = data['subword']['high_sim']['mean_overlap']*100 if data['subword']['high_sim'] else 0
                ph_all = data['phrase']['all']['mean_overlap']*100 if data['phrase']['all'] else 0
                ph_high = data['phrase']['high_sim']['mean_overlap']*100 if data['phrase']['high_sim'] else 0
                print(f"{name:<20} {sw_all:>12.1f}% {sw_high:>12.1f}% {ph_all:>12.1f}% {ph_high:>12.1f}%")
        return

    if not args.original or not args.ablation:
        parser.error("Either --run-all or both --original and --ablation are required")

    # Single comparison
    print(f"Loading original: {args.original}")
    original_data = load_json_file(args.original)

    print(f"Loading ablation: {args.ablation}")
    ablation_data = load_json_file(args.ablation)

    results = compare_contextual_nn(original_data, ablation_data, args.top_k, args.similarity_threshold)

    if results:
        print("\nResults:")
        print(f"  Common images: {results['num_common_images']}")

        sw_all = results['subword']['all']
        sw_high = results['subword']['high_sim']
        ph_all = results['phrase']['all']
        ph_high = results['phrase']['high_sim']

        print("\n  SUBWORD OVERLAP (token_str):")
        if sw_all:
            print(f"    All patches: {sw_all['mean_overlap']*100:.1f}% ± {sw_all['std_overlap']*100:.1f}% ({sw_all['num_patches']} patches)")
            print(f"    Intersection distribution: {sw_all['intersection_distribution']}")
        if sw_high:
            print(f"    High-sim only: {sw_high['mean_overlap']*100:.1f}% ± {sw_high['std_overlap']*100:.1f}% ({sw_high['num_patches']} patches)")

        print("\n  PHRASE OVERLAP (full caption):")
        if ph_all:
            print(f"    All patches: {ph_all['mean_overlap']*100:.1f}% ± {ph_all['std_overlap']*100:.1f}% ({ph_all['num_patches']} patches)")
            print(f"    Intersection distribution: {ph_all['intersection_distribution']}")
        if ph_high:
            print(f"    High-sim only: {ph_high['mean_overlap']*100:.1f}% ± {ph_high['std_overlap']*100:.1f}% ({ph_high['num_patches']} patches)")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
