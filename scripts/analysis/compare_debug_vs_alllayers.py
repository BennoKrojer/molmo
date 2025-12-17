#!/usr/bin/env python3
"""Quick comparison of debug output vs allLayers JSON and same-layer JSON."""

import json

# Load debug output (ground truth)
with open("analysis_results/debug_allLayers_sequential_img0.json") as f:
    debug = json.load(f)

# Load allLayers JSON (potentially buggy)
with open("analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/contextual_neighbors_visual24_allLayers_multi-gpu.json") as f:
    alllayers = json.load(f)

# Load same-layer JSON (visual24 vs contextual24 only)
with open("analysis_results/contextual_nearest_neighbors_visualN/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/contextual_neighbors_visual24_contextual24_multi-gpu.json") as f:
    samelayer = json.load(f)

# Get image 0 from both
alllayers_img0 = [r for r in alllayers['results'] if r['image_idx'] == 0][0]
samelayer_img0 = [r for r in samelayer['results'] if r['image_idx'] == 0][0]

print("="*70)
print("COMPARISON: Debug vs allLayers vs sameLayer")
print("="*70)
print()

all_mismatches = 0
same_mismatches = 0

for patch_idx in [0, 100, 224, 300, 500]:
    print(f"--- Patch {patch_idx} ---")
    
    # Debug result (filter to layer 24 for fair comparison with same-layer)
    debug_patch = debug['patches'][patch_idx]
    debug_nn_all = debug_patch['nearest_neighbors'][0]  # Best across all layers
    debug_layer24 = [nn for nn in debug_patch['nearest_neighbors'] if nn['contextual_layer'] == 24]
    debug_nn_24 = debug_layer24[0] if debug_layer24 else None
    
    # allLayers result
    all_patch = None
    for chunk in alllayers_img0['chunks']:
        for p in chunk['patches']:
            if p['patch_idx'] == patch_idx:
                all_patch = p
                break
    all_nn = all_patch['nearest_contextual_neighbors'][0] if all_patch else None
    
    # sameLayer result
    same_patch = None
    for chunk in samelayer_img0['chunks']:
        for p in chunk['patches']:
            if p['patch_idx'] == patch_idx:
                same_patch = p
                break
    same_nn = same_patch['nearest_contextual_neighbors'][0] if same_patch else None
    
    print(f"  DEBUG (all):   sim={debug_nn_all['similarity']:.4f}, layer={debug_nn_all['contextual_layer']}, token='{debug_nn_all['token_str']}'")
    print(f"  DEBUG (L24):   sim={debug_nn_24['similarity']:.4f}, token='{debug_nn_24['token_str']}'" if debug_nn_24 else "  DEBUG (L24):   N/A")
    print(f"  sameLayer:     sim={same_nn['similarity']:.4f}, token='{same_nn['token_str']}'" if same_nn else "  sameLayer:     N/A")
    print(f"  allLayers:     sim={all_nn['similarity']:.4f}, layer={all_nn.get('contextual_layer', 'N/A')}, token='{all_nn['token_str']}'")
    
    # Check sameLayer vs DEBUG (layer 24 portion)
    if debug_nn_24 and same_nn:
        same_sim_match = abs(debug_nn_24['similarity'] - same_nn['similarity']) < 0.01
        same_tok_match = debug_nn_24['token_str'] == same_nn['token_str']
        if same_sim_match and same_tok_match:
            print(f"  sameLayer vs DEBUG(L24): ✓ MATCH")
        else:
            print(f"  sameLayer vs DEBUG(L24): ✗ MISMATCH")
            same_mismatches += 1
    
    # Check allLayers vs DEBUG (all layers)
    if debug_nn_all and all_nn:
        all_sim_match = abs(debug_nn_all['similarity'] - all_nn['similarity']) < 0.01
        all_tok_match = debug_nn_all['token_str'] == all_nn['token_str']
        if all_sim_match and all_tok_match:
            print(f"  allLayers vs DEBUG(all): ✓ MATCH")
        else:
            print(f"  allLayers vs DEBUG(all): ✗ MISMATCH")
            all_mismatches += 1
    print()

print("="*70)
print(f"sameLayer vs DEBUG(L24): {same_mismatches} mismatches")
print(f"allLayers vs DEBUG(all): {all_mismatches} mismatches")
print()
if same_mismatches == 0:
    print("✓ sameLayer script is CORRECT (matches ground truth)")
else:
    print("✗ sameLayer script has issues")
if all_mismatches > 0:
    print("✗ allLayers script is BROKEN")
else:
    print("✓ allLayers script is correct")
print("="*70)

