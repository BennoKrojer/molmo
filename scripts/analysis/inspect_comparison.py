"""
Side-by-side comparison of OLD vs NEW results for manual inspection.
Shows the same images/patches from both approaches.
"""

import json
import sys

def compare_side_by_side(old_path, new_path, num_images=3, top_k=5):
    """Show side-by-side comparison of results."""
    
    print("Loading results...")
    with open(old_path, 'r') as f:
        old_data = json.load(f)
    with open(new_path, 'r') as f:
        new_data = json.load(f)
    
    print(f"\n{'='*100}")
    print(f"COMPARISON: OLD (visual{old_data['visual_layer']} vs contextual{old_data['contextual_layer']}) vs NEW (visual{new_data['visual_layer']} vs contextual{new_data['contextual_layer']})")
    print(f"{'='*100}\n")
    
    # Compare first few images
    for img_idx in range(min(num_images, len(old_data['results']))):
        old_result = old_data['results'][img_idx]
        new_result = new_data['results'][img_idx]
        
        print(f"\n{'='*100}")
        print(f"IMAGE {img_idx}")
        print(f"{'='*100}")
        print(f"\nCaption: {old_result['ground_truth_caption'][:150]}...")
        print()
        
        # Compare first chunk (usually "Full Image")
        old_chunk = old_result['chunks'][0]
        new_chunk = new_result['chunks'][0]
        
        print(f"Chunk: {old_chunk['chunk_name']}")
        print()
        
        # Compare first patch
        old_patch = old_chunk['patches'][0]
        new_patch = new_chunk['patches'][0]
        
        old_nns = old_patch['nearest_contextual_neighbors'][:top_k]
        new_nns = new_patch['nearest_contextual_neighbors'][:top_k]
        
        print(f"{'OLD (Layer 0 vision vs Layer 8 text)':<50} | {'NEW (Layer 8 vision vs Layer 8 text)':<50}")
        print(f"{'-'*50} | {'-'*50}")
        
        for i in range(top_k):
            old_nn = old_nns[i] if i < len(old_nns) else None
            new_nn = new_nns[i] if i < len(new_nns) else None
            
            if old_nn:
                old_str = f"{i+1}. '{old_nn['token_str']}' (sim={old_nn['similarity']:.4f})"
            else:
                old_str = ""
            
            if new_nn:
                new_str = f"{i+1}. '{new_nn['token_str']}' (sim={new_nn['similarity']:.4f})"
            else:
                new_str = ""
            
            print(f"{old_str:<50} | {new_str:<50}")
            
            # Show captions on next line
            if old_nn:
                old_cap = old_nn['caption'][:45] + "..." if len(old_nn['caption']) > 45 else old_nn['caption']
                print(f"   {old_cap:<47} | ", end="")
            else:
                print(f"{'':<50} | ", end="")
            
            if new_nn:
                new_cap = new_nn['caption'][:45] + "..." if len(new_nn['caption']) > 45 else new_nn['caption']
                print(f"   {new_cap}")
            else:
                print()
        
        # Summary stats
        old_top = old_nns[0]['similarity'] if old_nns else 0
        new_top = new_nns[0]['similarity'] if new_nns else 0
        old_avg = sum(nn['similarity'] for nn in old_nns) / len(old_nns) if old_nns else 0
        new_avg = sum(nn['similarity'] for nn in new_nns) / len(new_nns) if new_nns else 0
        
        print()
        print(f"Top similarity:  OLD={old_top:.4f}  NEW={new_top:.4f}  (diff: {new_top-old_top:+.4f}, {((new_top/old_top-1)*100):+.1f}%)")
        print(f"Avg top-{top_k}:   OLD={old_avg:.4f}  NEW={new_avg:.4f}  (diff: {new_avg-old_avg:+.4f}, {((new_avg/old_avg-1)*100):+.1f}%)")
        print()
    
    # Overall summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    
    all_old_tops = []
    all_new_tops = []
    all_old_avgs = []
    all_new_avgs = []
    
    for img_idx in range(min(num_images, len(old_data['results']))):
        old_result = old_data['results'][img_idx]
        new_result = new_data['results'][img_idx]
        
        old_chunk = old_result['chunks'][0]
        new_chunk = new_result['chunks'][0]
        old_patch = old_chunk['patches'][0]
        new_patch = new_chunk['patches'][0]
        
        old_nns = old_patch['nearest_contextual_neighbors'][:top_k]
        new_nns = new_patch['nearest_contextual_neighbors'][:top_k]
        
        if old_nns:
            all_old_tops.append(old_nns[0]['similarity'])
            all_old_avgs.append(sum(nn['similarity'] for nn in old_nns) / len(old_nns))
        if new_nns:
            all_new_tops.append(new_nns[0]['similarity'])
            all_new_avgs.append(sum(nn['similarity'] for nn in new_nns) / len(new_nns))
    
    if all_old_tops and all_new_tops:
        print(f"Average top similarity across {len(all_old_tops)} images:")
        print(f"  OLD: {sum(all_old_tops)/len(all_old_tops):.4f}")
        print(f"  NEW: {sum(all_new_tops)/len(all_new_tops):.4f}")
        print(f"  Difference: {sum(all_new_tops)/len(all_new_tops) - sum(all_old_tops)/len(all_old_tops):+.4f}")
        print()
        print(f"Average top-{top_k} similarity across {len(all_old_avgs)} images:")
        print(f"  OLD: {sum(all_old_avgs)/len(all_old_avgs):.4f}")
        print(f"  NEW: {sum(all_new_avgs)/len(all_new_avgs):.4f}")
        print(f"  Difference: {sum(all_new_avgs)/len(all_new_avgs) - sum(all_old_avgs)/len(all_old_avgs):+.4f}")


def create_small_comparison_json(old_path, new_path, output_path, num_images=2):
    """Create a small JSON file with just a few images for manual inspection."""
    
    with open(old_path, 'r') as f:
        old_data = json.load(f)
    with open(new_path, 'r') as f:
        new_data = json.load(f)
    
    # Create comparison structure
    comparison = {
        'old_metadata': {
            'visual_layer': old_data['visual_layer'],
            'contextual_layer': old_data['contextual_layer'],
            'description': f"Layer {old_data['visual_layer']} vision vs Layer {old_data['contextual_layer']} text"
        },
        'new_metadata': {
            'visual_layer': new_data['visual_layer'],
            'contextual_layer': new_data['contextual_layer'],
            'description': f"Layer {new_data['visual_layer']} vision vs Layer {new_data['contextual_layer']} text"
        },
        'comparisons': []
    }
    
    for img_idx in range(min(num_images, len(old_data['results']))):
        old_result = old_data['results'][img_idx]
        new_result = new_data['results'][img_idx]
        
        old_chunk = old_result['chunks'][0]
        new_chunk = new_result['chunks'][0]
        old_patch = old_chunk['patches'][0]
        new_patch = new_chunk['patches'][0]
        
        comparison['comparisons'].append({
            'image_idx': img_idx,
            'ground_truth_caption': old_result['ground_truth_caption'],
            'old': {
                'visual_layer': old_data['visual_layer'],
                'contextual_layer': old_data['contextual_layer'],
                'nearest_neighbors': old_patch['nearest_contextual_neighbors'][:10]
            },
            'new': {
                'visual_layer': new_data['visual_layer'],
                'contextual_layer': new_data['contextual_layer'],
                'nearest_neighbors': new_patch['nearest_contextual_neighbors'][:10]
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Created comparison JSON: {output_path}")
    print(f"  Contains {len(comparison['comparisons'])} images with top-10 neighbors each")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_comparison.py <old_json> <new_json> [num_images] [top_k]")
        print()
        print("Example:")
        print("  python inspect_comparison.py \\")
        print("    analysis_results/contextual_nearest_neighbors/.../visual0_contextual8_multi-gpu.json \\")
        print("    analysis_results/contextual_nearest_neighbors_test/.../visual8_contextual8_multi-gpu.json \\")
        print("    3 5")
        sys.exit(1)
    
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    # Show side-by-side comparison
    compare_side_by_side(old_path, new_path, num_images=num_images, top_k=top_k)
    
    # Create small JSON for manual inspection
    output_json = "comparison_sample.json"
    create_small_comparison_json(old_path, new_path, output_json, num_images=num_images)
    print(f"\nYou can also inspect the JSON file directly: {output_json}")

