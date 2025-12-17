"""
Compare OLD results (visual0 vs contextualN) with NEW results (visualN vs contextualN).

This helps understand the difference between the two approaches.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_json_metadata(json_path):
    """Load just the metadata and a sample of results from a JSON file."""
    print(f"Loading: {json_path}")
    
    # Read first chunk to get metadata
    with open(json_path, 'r') as f:
        chunk = f.read(10000)  # Read first 10KB
        
        # Find where results start
        results_idx = chunk.find('"results"')
        if results_idx > 0:
            metadata_str = chunk[:results_idx].rstrip().rstrip(',') + '}'
            metadata = json.loads(metadata_str)
            
            # Also try to get a sample result
            # Read more to get first image result
            f.seek(0)
            larger_chunk = f.read(50000)  # Read 50KB
            
            # Find first complete image result
            if '"image_idx": 0' in larger_chunk:
                # Try to extract first image's data
                start_idx = larger_chunk.find('"image_idx": 0')
                # Find the end of this image's data (next image_idx or end of results)
                end_markers = ['"image_idx": 1', '"image_idx": 2', ']']
                end_idx = len(larger_chunk)
                for marker in end_markers:
                    marker_idx = larger_chunk.find(marker, start_idx)
                    if marker_idx > 0 and marker_idx < end_idx:
                        end_idx = marker_idx
                
                sample_str = larger_chunk[start_idx:end_idx]
                # Try to parse it (might be incomplete)
                try:
                    # Add closing braces
                    if not sample_str.strip().endswith('}'):
                        sample_str += '}'
                    sample_result = json.loads('{' + sample_str)
                except:
                    sample_result = None
            else:
                sample_result = None
            
            return metadata, sample_result
    
    return None, None


def compare_results(old_path, new_path):
    """Compare old and new result files."""
    print("="*80)
    print("COMPARING OLD vs NEW RESULTS")
    print("="*80)
    print()
    
    old_meta, old_sample = load_json_metadata(old_path)
    new_meta, new_sample = load_json_metadata(new_path)
    
    if not old_meta or not new_meta:
        print("ERROR: Could not load metadata from one or both files")
        return
    
    print("OLD APPROACH:")
    print(f"  Visual layer: {old_meta.get('visual_layer')}")
    print(f"  Contextual layer: {old_meta.get('contextual_layer')}")
    print(f"  Compares: Layer {old_meta.get('visual_layer')} vision tokens vs Layer {old_meta.get('contextual_layer')} text embeddings")
    print()
    
    print("NEW APPROACH:")
    print(f"  Visual layer: {new_meta.get('visual_layer')}")
    print(f"  Contextual layer: {new_meta.get('contextual_layer')}")
    print(f"  Compares: Layer {new_meta.get('visual_layer')} vision tokens vs Layer {new_meta.get('contextual_layer')} text embeddings")
    print()
    
    # Compare sample results if available
    if old_sample and new_sample:
        print("="*80)
        print("SAMPLE COMPARISON (First Image, First Patch):")
        print("="*80)
        
        old_patches = old_sample.get('patches', []) or old_sample.get('chunks', [{}])[0].get('patches', [])
        new_patches = new_sample.get('patches', []) or new_sample.get('chunks', [{}])[0].get('patches', [])
        
        if old_patches and new_patches:
            old_patch = old_patches[0]
            new_patch = new_patches[0]
            
            print("\nOLD (layer 0 vision vs layer 8 text):")
            if 'nearest_contextual_neighbors' in old_patch:
                for i, nn in enumerate(old_patch['nearest_contextual_neighbors'][:3]):
                    print(f"  {i+1}. '{nn.get('token_str', 'N/A')}' (sim={nn.get('similarity', 0):.4f})")
                    print(f"     Caption: {nn.get('caption', 'N/A')[:60]}...")
            
            print("\nNEW (layer 8 vision vs layer 8 text):")
            if 'nearest_contextual_neighbors' in new_patch:
                for i, nn in enumerate(new_patch['nearest_contextual_neighbors'][:3]):
                    print(f"  {i+1}. '{nn.get('token_str', 'N/A')}' (sim={nn.get('similarity', 0):.4f})")
                    print(f"     Caption: {nn.get('caption', 'N/A')[:60]}...")
            
            # Compare top similarity scores
            if old_patch.get('nearest_contextual_neighbors') and new_patch.get('nearest_contextual_neighbors'):
                old_top_sim = old_patch['nearest_contextual_neighbors'][0].get('similarity', 0)
                new_top_sim = new_patch['nearest_contextual_neighbors'][0].get('similarity', 0)
                print(f"\nTop similarity comparison:")
                print(f"  OLD: {old_top_sim:.4f}")
                print(f"  NEW: {new_top_sim:.4f}")
                print(f"  Difference: {new_top_sim - old_top_sim:+.4f}")
    
    print()
    print("="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("OLD compares: Initial vision tokens (layer 0) with evolved text (layer N)")
    print("NEW compares: Evolved vision tokens (layer N) with evolved text (layer N)")
    print()
    print("If NEW has higher similarity, it suggests vision tokens evolve to match text.")
    print("If OLD has higher similarity, it suggests initial vision tokens already align well.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_old_vs_new_results.py <old_json> <new_json>")
        print()
        print("Example:")
        print("  python compare_old_vs_new_results.py \\")
        print("    analysis_results/contextual_nearest_neighbors/.../contextual_neighbors_visual0_contextual8_multi-gpu.json \\")
        print("    analysis_results/contextual_nearest_neighbors_test/.../contextual_neighbors_visual8_contextual8_multi-gpu.json")
        sys.exit(1)
    
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    
    compare_results(old_path, new_path)

