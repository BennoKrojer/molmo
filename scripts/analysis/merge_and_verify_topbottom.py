#!/usr/bin/env python3
"""
Merge topbottom shards and run sanity checks.

This script:
1. Merges embedding files from all shards (using shutil.copy2 for speed)
2. Merges token_embeddings.json files
3. Creates metadata.json
4. Runs sanity checks comparing to OLMo

Usage:
    python scripts/analysis/merge_and_verify_topbottom.py
"""

import json
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# Configuration
NUM_SHARDS = 6
MAX_PER_TOKEN = 20
LAYERS = [1, 2, 4, 8, 16, 24, 30, 31]
BASE_DIR = Path("molmo_data/contextual_llm_embeddings_vg/train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm")
OLMO_DIR = Path("molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview")

random.seed(42)  # Reproducibility


def merge_single_layer(args):
    """Merge a single layer from all shards."""
    layer_idx, shard_dirs, output_dir = args

    layer_output = output_dir / f"layer_{layer_idx}"
    emb_output = layer_output / "embeddings"
    emb_output.mkdir(parents=True, exist_ok=True)

    # Collect all token -> entries mappings from all shards
    merged_tokens = defaultdict(list)
    all_embeddings = []  # (shard_dir, emb_file, token_str)

    for shard_dir in shard_dirs:
        token_file = shard_dir / f"layer_{layer_idx}" / "token_embeddings.json"
        if token_file.exists():
            with open(token_file) as f:
                shard_data = json.load(f)

            for token, entries in shard_data.items():
                for entry in entries:
                    merged_tokens[token].append({
                        'shard_dir': shard_dir,
                        'entry': entry
                    })

    # Limit to MAX_PER_TOKEN per token and copy files
    final_tokens = {}
    emb_counter = 0

    for token, items in merged_tokens.items():
        if len(items) > MAX_PER_TOKEN:
            items = random.sample(items, MAX_PER_TOKEN)

        token_entries = []
        for item in items:
            shard_dir = item['shard_dir']
            entry = item['entry']
            old_path = shard_dir / f"layer_{layer_idx}" / entry['embedding_path']

            if old_path.exists():
                new_name = f"emb_{emb_counter:08d}.npy"
                new_path = emb_output / new_name
                shutil.copy2(old_path, new_path)

                new_entry = entry.copy()
                new_entry['embedding_path'] = f"embeddings/{new_name}"
                token_entries.append(new_entry)
                emb_counter += 1

        if token_entries:
            final_tokens[token] = token_entries

    # Save merged token_embeddings.json
    with open(layer_output / "token_embeddings.json", 'w') as f:
        json.dump(final_tokens, f)

    return layer_idx, len(final_tokens), emb_counter


def merge_shards():
    """Merge all shards into final directory."""
    print("=" * 70)
    print("MERGING TOPBOTTOM SHARDS")
    print("=" * 70)

    # Find shard directories
    shard_dirs = []
    for i in range(NUM_SHARDS):
        shard_dir = Path(str(BASE_DIR) + f"_shard{i}")
        if shard_dir.exists():
            shard_dirs.append(shard_dir)
        else:
            print(f"WARNING: Shard {i} not found at {shard_dir}")

    if len(shard_dirs) != NUM_SHARDS:
        print(f"ERROR: Expected {NUM_SHARDS} shards, found {len(shard_dirs)}")
        sys.exit(1)

    print(f"Found {len(shard_dirs)} shard directories")

    # Clear and create output directory
    if BASE_DIR.exists():
        print(f"Removing existing {BASE_DIR}")
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True)

    # Merge layers in parallel
    print(f"\nMerging {len(LAYERS)} layers with {8} workers...")
    merge_args = [(layer, shard_dirs, BASE_DIR) for layer in LAYERS]

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(merge_single_layer, args): args[0] for args in merge_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging"):
            layer_idx, num_tokens, num_embs = future.result()
            results.append((layer_idx, num_tokens, num_embs))
            print(f"  Layer {layer_idx}: {num_tokens} tokens, {num_embs} embeddings")

    # Create metadata
    total_captions = 0
    for shard_dir in shard_dirs:
        meta_file = shard_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                shard_meta = json.load(f)
                total_captions += shard_meta.get('num_captions_processed', 0)

    metadata = {
        'model_name': 'train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm',
        'model_type': 'Molmo (LLM backbone only, text-only)',
        'checkpoint_path': 'molmo_data/checkpoints/ablations/train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm/step12000-unsharded',
        'layers_extracted': LAYERS,
        'num_captions_processed': total_captions,
        'max_captions_per_token': MAX_PER_TOKEN,
        'embedding_dtype': 'float8',
        'dataset': 'vg',
        'data_source': 'Visual Genome phrases',
        'position_filtering': {
            'enabled': False,
            'description': 'All token positions included',
            'skip_positions': [],
            'preferred_positions': 'all',
            'fallback_positions': 'N/A'
        },
        'merged_from_shards': NUM_SHARDS
    }

    with open(BASE_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Merge complete! Output: {BASE_DIR}")
    return results


def sanity_check():
    """Compare structure to OLMo - should be identical."""
    print("\n" + "=" * 70)
    print("SANITY CHECK: Comparing to OLMo")
    print("=" * 70)

    all_passed = True

    for layer in LAYERS:
        topbottom_file = BASE_DIR / f"layer_{layer}" / "token_embeddings.json"
        olmo_file = OLMO_DIR / f"layer_{layer}" / "token_embeddings.json"

        with open(topbottom_file) as f:
            topbottom_data = json.load(f)
        with open(olmo_file) as f:
            olmo_data = json.load(f)

        topbottom_tokens = set(topbottom_data.keys())
        olmo_tokens = set(olmo_data.keys())

        tb_count = sum(len(v) for v in topbottom_data.values())
        olmo_count = sum(len(v) for v in olmo_data.values())

        tokens_match = topbottom_tokens == olmo_tokens
        counts_match = tb_count == olmo_count

        status = "✓" if (tokens_match and counts_match) else "✗"

        print(f"Layer {layer}: {len(topbottom_tokens)} tokens, {tb_count} embeddings", end="")
        if tokens_match and counts_match:
            print(f" {status} matches OLMo")
        else:
            print(f" {status} MISMATCH!")
            if not tokens_match:
                only_tb = len(topbottom_tokens - olmo_tokens)
                only_olmo = len(olmo_tokens - topbottom_tokens)
                print(f"    Tokens: only in topbottom={only_tb}, only in olmo={only_olmo}")
            if not counts_match:
                print(f"    Embeddings: topbottom={tb_count}, olmo={olmo_count}")
            all_passed = False

    if all_passed:
        print("\n✓ ALL SANITY CHECKS PASSED!")
        print("  - Same token sets as OLMo")
        print("  - Same embedding counts per layer")
        print("  - Only the embedding VALUES differ (as expected for finetuned LLM)")
    else:
        print("\n✗ SOME CHECKS FAILED - investigate!")

    return all_passed


if __name__ == "__main__":
    # Step 1: Merge
    results = merge_shards()

    # Step 2: Sanity check
    passed = sanity_check()

    sys.exit(0 if passed else 1)
