#!/usr/bin/env python3
"""Fix corrupted JSON files from interrupted run and check if JSONs are identical across layers."""

import json
from pathlib import Path
import sys

# Adjust this path if needed
output_dir = Path("molmo_data/contextual_llm_embeddings_vg/meta-llama_Meta-Llama-3-8B")

if not output_dir.exists():
    print(f"Directory not found: {output_dir}")
    print("Please provide the correct path as an argument")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        sys.exit(1)

progress_file = output_dir / "progress.json"

# Load progress to see which layers exist
with open(progress_file) as f:
    progress = json.load(f)

layers = sorted([int(k) for k in progress['layer_counters'].keys()])
print(f"Found {len(layers)} layers: {layers}")

# Dictionary to store loaded valid JSONs for comparison
valid_jsons = {}

# Test each layer's JSON
corrupted_files = []
for layer in layers:
    layer_file = output_dir / f"layer_{layer}" / "token_embeddings.json"
    if not layer_file.exists():
        print(f"Layer {layer}: File doesn't exist yet")
        continue
    
    try:
        with open(layer_file) as f:
            data = json.load(f)
        print(f"Layer {layer}: OK ({len(data)} tokens)")
        valid_jsons[layer] = data
    except json.JSONDecodeError as e:
        print(f"Layer {layer}: CORRUPTED - {e}")
        corrupted_files.append((layer, layer_file))

if not corrupted_files:
    print("\nNo corrupted files found!")
else:
    print(f"\nFound {len(corrupted_files)} corrupted file(s)")

# Compare valid JSONs to check if they're identical
if len(valid_jsons) >= 2:
    print("\n" + "="*80)
    print("CHECKING IF JSONs ARE IDENTICAL ACROSS LAYERS")
    print("="*80)
    
    reference_layer = list(valid_jsons.keys())[0]
    reference_data = valid_jsons[reference_layer]
    reference_tokens = set(reference_data.keys())
    
    print(f"\nUsing layer {reference_layer} as reference ({len(reference_tokens)} tokens)")
    
    all_identical = True
    for layer, data in valid_jsons.items():
        if layer == reference_layer:
            continue
        
        layer_tokens = set(data.keys())
        
        # Check token sets
        if layer_tokens != reference_tokens:
            all_identical = False
            only_in_ref = reference_tokens - layer_tokens
            only_in_layer = layer_tokens - reference_tokens
            print(f"\nLayer {layer}: DIFFERENT token set!")
            print(f"  Tokens only in layer {reference_layer}: {len(only_in_ref)} (e.g., {list(only_in_ref)[:3]})")
            print(f"  Tokens only in layer {layer}: {len(only_in_layer)} (e.g., {list(only_in_layer)[:3]})")
        else:
            # Check if the entries are the same (excluding embedding_path which differs by layer)
            different_entries = 0
            for token in list(reference_tokens)[:10]:  # Check first 10 tokens in detail
                ref_entries = reference_data[token]
                layer_entries = data[token]
                
                if len(ref_entries) != len(layer_entries):
                    different_entries += 1
                    continue
                
                # Compare caption and position (not embedding_path)
                for i, (ref_entry, layer_entry) in enumerate(zip(ref_entries, layer_entries)):
                    if (ref_entry.get('caption') != layer_entry.get('caption') or 
                        ref_entry.get('position') != layer_entry.get('position')):
                        different_entries += 1
                        break
            
            if different_entries > 0:
                all_identical = False
                print(f"\nLayer {layer}: Same tokens, but DIFFERENT caption/position entries!")
                print(f"  {different_entries}/10 sampled tokens have different entries")
            else:
                print(f"\nLayer {layer}: IDENTICAL entries (same captions/positions)!")
    
    if all_identical:
        print("\n✓ ALL LAYERS ARE IDENTICAL (same tokens, captions, and positions)")
        print("  Only embedding_path differs, which is expected")
    else:
        print("\n✗ LAYERS ARE DIFFERENT!")
        print("  This is a bug - all layers should have the same instances stored")

if not corrupted_files:
    sys.exit(0)

print("\n" + "="*80)
print("FIXING CORRUPTED FILES")
print("="*80)

for layer, corrupted_file in corrupted_files:
    print(f"\nFixing layer {layer}: {corrupted_file}")
    
    # Read the file and try to salvage what we can
    print(f"  Reading file...")
    with open(corrupted_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    print(f"  File size: {len(content)} chars, {len(content.splitlines())} lines")
    
    # Strategy: Find the last complete token entry
    # Look for pattern: "token_name": [...], or "token_name": [...]}
    # And truncate right before the incomplete entry
    
    # Find last occurrence of "],\n  " or "]\n}" which indicates a complete token entry
    last_complete_token = max(
        content.rfind('],\n  "'),
        content.rfind(']\n}')
    )
    
    if last_complete_token > 0:
        # Find where this entry ends (after the ])
        bracket_pos = content.find(']', last_complete_token)
        if bracket_pos > 0:
            # Truncate after the ] and add closing }
            if content[bracket_pos + 1:bracket_pos + 3] == '\n}':
                # Already has closing brace
                fixed_content = content[:bracket_pos + 3]
            else:
                # Need to add closing brace
                fixed_content = content[:bracket_pos + 1] + '\n}'
            
            # Save backup
            backup_file = str(corrupted_file) + '.backup'
            print(f"  Creating backup: {backup_file}")
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  Truncating from {len(content)} to {len(fixed_content)} chars")
            
            # Verify it's valid JSON
            try:
                fixed_data = json.loads(fixed_content)
                print(f"  ✓ Fixed! Now has {len(fixed_data)} tokens (was corrupted)")
                
                # Write the fixed version
                print(f"  Writing fixed JSON...")
                with open(corrupted_file, 'w', encoding='utf-8') as f:
                    json.dump(fixed_data, f, indent=2)
                print(f"  ✓ Saved successfully")
                
                # Add to valid_jsons for comparison
                valid_jsons[layer] = fixed_data
                
            except json.JSONDecodeError as e:
                print(f"  ✗ ERROR: Still invalid JSON: {e}")
                print(f"  Try manually deleting the corrupted entries in: {corrupted_file}")
        else:
            print(f"  ✗ Could not find bracket position")
    else:
        print(f"  ✗ Could not find any complete token entries")
        print(f"  The file may be too corrupted. You might need to delete: {corrupted_file}")

print("\nDone! You can now re-run your script.")

