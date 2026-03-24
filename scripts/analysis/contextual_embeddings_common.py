#!/usr/bin/env python3
"""
Shared infrastructure for contextual embedding extraction from off-the-shelf VLMs.

This module contains all the common code (data loading, reservoir sampling, saving,
merging, cache building) shared across model-specific extraction scripts.

Model-specific scripts (molmo_7b/, llava_1_5/, qwen2_vl/) import from here
and only override model loading + forward pass.
"""

import argparse
import gc
import json
import os
import random
import time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Import ml_dtypes for float8 support
try:
    import ml_dtypes
    HAVE_ML_DTYPES = True
except ImportError:
    HAVE_ML_DTYPES = False

# Constants
BATCH_SIZE = 32
SAVE_FREQUENCY = 1000
MAX_CAPTIONS_PER_TOKEN = 20
EMBEDDING_DTYPE = 'float16'


def convert_embedding_dtype(embedding, target_dtype='float16'):
    """Convert embedding to target dtype for storage optimization."""
    if target_dtype == 'float32':
        return embedding.astype(np.float32)
    elif target_dtype == 'float16':
        return embedding.astype(np.float16)
    elif target_dtype == 'float8':
        if HAVE_ML_DTYPES:
            FLOAT8_MAX = 448.0
            embedding_clipped = np.clip(embedding, -FLOAT8_MAX, FLOAT8_MAX)
            return embedding_clipped.astype(ml_dtypes.float8_e4m3fn)
        else:
            print("Warning: ml_dtypes not available, falling back to float16")
            return embedding.astype(np.float16)
    else:
        raise ValueError(f"Unsupported dtype: {target_dtype}")


def convert_from_stored_dtype(embedding_array):
    """Convert embedding from stored dtype back to float32."""
    dtype_str = str(embedding_array.dtype)
    if dtype_str.startswith('|V') or dtype_str.startswith('V'):
        if HAVE_ML_DTYPES:
            embedding_fp8 = embedding_array.view(ml_dtypes.float8_e4m3fn)
            return embedding_fp8.astype(np.float32)
        else:
            raise ImportError("ml_dtypes required to load float8 embeddings")
    elif 'float8' in dtype_str:
        return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float16:
        return embedding_array.astype(np.float32)
    elif embedding_array.dtype == np.float32:
        return embedding_array
    else:
        return embedding_array.astype(np.float32)


def load_vg_phrases(vg_file_path, num_phrases=None):
    """Load phrases from Visual Genome phrases file."""
    if num_phrases is None:
        print(f"Loading ALL phrases from {vg_file_path}...")
    else:
        print(f"Loading {num_phrases} phrases from {vg_file_path}...")
    phrases = []
    with open(vg_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_phrases is not None and i >= num_phrases:
                break
            phrase = line.strip()
            if phrase:
                phrases.append(phrase)
            if (i + 1) % 500000 == 0:
                print(f"  Loaded {i + 1} phrases...")
    print(f"Loaded {len(phrases)} phrases from VG")
    return phrases


def load_tsv_captions(tsv_file_path, num_captions=None):
    """Load captions from TSV file (Conceptual Captions format)."""
    if num_captions is None:
        print(f"Loading ALL captions from {tsv_file_path}...")
    else:
        print(f"Loading {num_captions} captions from {tsv_file_path}...")
    captions = []
    with open(tsv_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_captions is not None and i >= num_captions:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                caption = parts[0].strip()
                if caption:
                    captions.append(caption)
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1} captions...")
    print(f"Loaded {len(captions)} captions")
    return captions


def save_progress(output_dir, layer_dirs, captions_processed, total_captions, token_seen_counts=None):
    """Save current progress to checkpoint."""
    progress = {
        'captions_processed': captions_processed,
        'total_captions': total_captions,
        'layer_counters': {str(layer_idx): info['counter'] for layer_idx, info in layer_dirs.items()}
    }
    if token_seen_counts is not None:
        progress['token_seen_counts'] = {
            str(layer_idx): dict(layer_counts)
            for layer_idx, layer_counts in token_seen_counts.items()
        }
    progress_file = output_dir / "progress.json"
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    for layer_idx, layer_info in layer_dirs.items():
        layer_dir = layer_info['layer_dir']
        token_dict = layer_info['token_dict']
        output_file = layer_dir / "token_embeddings.json"
        with open(output_file, 'w') as f:
            json.dump(token_dict, f, indent=2)
    print(f"  Progress saved: {captions_processed}/{total_captions} captions")


def load_progress(output_dir):
    """Load existing progress from checkpoint."""
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None


def extract_contextual_embeddings(
    model_forward_fn,
    tokenizer,
    captions,
    layers_to_extract,
    batch_size,
    layer_dirs,
    device,
    output_dir,
    start_offset=0,
    max_captions_per_token=MAX_CAPTIONS_PER_TOKEN,
    embedding_dtype=EMBEDDING_DTYPE,
    apply_position_filter=True
):
    """
    Extract contextual embeddings using a model-specific forward function.

    Args:
        model_forward_fn: callable(input_ids, attention_mask) -> hidden_states tuple
            Must return a tuple of hidden states, one per layer (including embedding layer).
        tokenizer: HuggingFace tokenizer
        captions: list of strings
        layers_to_extract: list of layer indices
        ... (rest same as Qwen2-VL version)
    """
    total_embedding_count = 0
    captions_processed = start_offset

    if start_offset > 0:
        captions = captions[start_offset:]
        print(f"\nSkipping first {start_offset} captions (already processed)")

    total_batches = (len(captions) + batch_size - 1) // batch_size
    print(f"\nProcessing {len(captions)} captions in {total_batches} batches")
    print(f"Reservoir sampling: max {max_captions_per_token} embeddings per token")
    print(f"Storage dtype: {embedding_dtype}")

    token_seen_counts = {layer_idx: defaultdict(lambda: {'preferred_count': 0, 'fallback_count': 0})
                         for layer_idx in layers_to_extract}

    for batch_idx in tqdm(range(0, len(captions), batch_size), desc="Processing batches"):
        batch_captions = captions[batch_idx:batch_idx + batch_size]

        encodings = tokenizer(
            batch_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                all_hidden_states = model_forward_fn(input_ids, attention_mask)

        for sent_idx, caption in enumerate(batch_captions):
            valid_positions = torch.where(attention_mask[sent_idx] == 1)[0].cpu().numpy()
            token_ids = input_ids[sent_idx][valid_positions]

            for pos_idx, token_pos in enumerate(valid_positions):
                if apply_position_filter and pos_idx < 2:
                    continue

                token_id = token_ids[pos_idx].item()
                token_str = tokenizer.decode([token_id])

                if apply_position_filter:
                    is_preferred = pos_idx >= 10
                else:
                    is_preferred = True

                first_layer = layers_to_extract[0]
                first_layer_info = layer_dirs[first_layer]
                first_token_dict = first_layer_info['token_dict']

                if token_str not in first_token_dict:
                    first_token_dict[token_str] = {
                        'preferred': [], 'fallback': [], 'combined': []
                    }

                counts = token_seen_counts[first_layer][token_str]
                if is_preferred:
                    counts['preferred_count'] += 1
                    reservoir_type = 'preferred'
                    count = counts['preferred_count']
                else:
                    counts['fallback_count'] += 1
                    reservoir_type = 'fallback'
                    count = counts['fallback_count']

                first_reservoir = first_token_dict[token_str][reservoir_type]
                should_store = False
                replace_idx = None

                if len(first_reservoir) < max_captions_per_token:
                    should_store = True
                else:
                    j = random.randint(0, count - 1)
                    if j < max_captions_per_token:
                        should_store = True
                        replace_idx = j

                if should_store:
                    for layer_idx in layers_to_extract:
                        hidden_state = all_hidden_states[layer_idx][sent_idx, token_pos].cpu().numpy()

                        if np.isnan(hidden_state).any():
                            raise ValueError(f"NaN detected in hidden state! Layer {layer_idx}, token '{token_str}'")

                        layer_info = layer_dirs[layer_idx]
                        token_dict = layer_info['token_dict']

                        if token_str not in token_dict:
                            token_dict[token_str] = {
                                'preferred': [], 'fallback': [], 'combined': []
                            }

                        reservoir = token_dict[token_str][reservoir_type]
                        counter = layer_info['counter']

                        embeddings_dir = layer_info['embeddings_dir']
                        embedding_path = embeddings_dir / f"emb_{counter:08d}.npy"

                        hidden_state_converted = convert_embedding_dtype(hidden_state, embedding_dtype)
                        np.save(embedding_path, hidden_state_converted)

                        entry = {
                            'embedding_path': str(embedding_path.relative_to(layer_info['layer_dir'])),
                            'caption': caption,
                            'position': pos_idx,
                            'token_id': token_id,
                            'dtype': str(hidden_state_converted.dtype)
                        }

                        if replace_idx is not None and replace_idx < len(reservoir):
                            reservoir[replace_idx] = entry
                        else:
                            reservoir.append(entry)

                        layer_info['counter'] += 1
                        total_embedding_count += 1

        captions_processed += len(batch_captions)

        if captions_processed % SAVE_FREQUENCY < batch_size:
            total_captions = start_offset + len(captions)
            print(f"\nSaving progress at {captions_processed}/{total_captions} captions...")
            save_progress(output_dir, layer_dirs, captions_processed, total_captions, token_seen_counts)

        del all_hidden_states
        torch.cuda.empty_cache()

    print(f"\nExtracted {total_embedding_count} embeddings across {len(layers_to_extract)} layers")
    for layer_idx in layers_to_extract:
        num_unique = len(layer_dirs[layer_idx]['token_dict'])
        num_embeddings = layer_dirs[layer_idx]['counter']
        print(f"  Layer {layer_idx}: {num_embeddings} embeddings for {num_unique} unique tokens")

    return total_embedding_count, captions_processed


def finalize_token_embeddings(layer_dirs, max_captions_per_token):
    """Combine preferred and fallback embeddings."""
    for layer_idx, layer_info in layer_dirs.items():
        token_dict = layer_info['token_dict']
        for token_str, data in token_dict.items():
            if isinstance(data, dict) and 'preferred' in data:
                preferred = data.get('preferred', [])
                fallback = data.get('fallback', [])
                combined = preferred.copy()
                if len(combined) < max_captions_per_token:
                    needed = max_captions_per_token - len(combined)
                    combined.extend(fallback[:needed])
                token_dict[token_str] = combined


def merge_single_layer(args):
    """Merge a single layer from all shards."""
    import shutil
    layer_name, shard_dirs, output_dir, max_per_token = args
    merged_layer_dir = output_dir / layer_name
    merged_embeddings_dir = merged_layer_dir / "embeddings"
    merged_embeddings_dir.mkdir(parents=True, exist_ok=True)

    merged_tokens = {}
    embedding_counter = 0

    for shard_dir in shard_dirs:
        shard_layer_dir = shard_dir / layer_name
        token_file = shard_layer_dir / "token_embeddings.json"
        if not token_file.exists():
            continue
        with open(token_file, 'r') as f:
            shard_tokens = json.load(f)
        for token_str, entries in shard_tokens.items():
            if not isinstance(entries, list):
                continue
            if token_str not in merged_tokens:
                merged_tokens[token_str] = []
            for entry in entries:
                if len(merged_tokens[token_str]) >= max_per_token:
                    break
                src_path = shard_layer_dir / entry['embedding_path']
                if src_path.exists():
                    new_filename = f"emb_{embedding_counter:08d}.npy"
                    dst_path = merged_embeddings_dir / new_filename
                    shutil.copy2(src_path, dst_path)
                    new_entry = entry.copy()
                    new_entry['embedding_path'] = f"embeddings/{new_filename}"
                    merged_tokens[token_str].append(new_entry)
                    embedding_counter += 1

    with open(merged_layer_dir / "token_embeddings.json", 'w') as f:
        json.dump(merged_tokens, f, indent=2)
    return layer_name, len(merged_tokens), embedding_counter


def merge_shards(output_dir, num_shards, max_per_token=MAX_CAPTIONS_PER_TOKEN, build_cache=True, num_workers=8):
    """Merge shard outputs into a single directory and optionally build caches."""
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    output_dir = Path(output_dir)
    print(f"Merging {num_shards} shards into {output_dir}")

    shard_dirs = []
    for i in range(num_shards):
        shard_dir = output_dir.parent / f"{output_dir.name}_shard{i}"
        if shard_dir.exists():
            shard_dirs.append(shard_dir)
        else:
            print(f"Warning: Shard {i} not found at {shard_dir}")

    if not shard_dirs:
        print("No shard directories found!")
        return

    print(f"Found {len(shard_dirs)} shard directories")

    first_shard = shard_dirs[0]
    layer_dirs_to_merge = [d.name for d in first_shard.iterdir() if d.is_dir() and d.name.startswith("layer_")]
    print(f"Layers to merge: {layer_dirs_to_merge}")
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_args = [(layer_name, shard_dirs, output_dir, max_per_token) for layer_name in layer_dirs_to_merge]
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(merge_single_layer, args): args[0] for args in merge_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging layers"):
            layer_name = futures[future]
            layer_name, num_tokens, num_embeddings = future.result()
            results.append((layer_name, num_tokens, num_embeddings))
            print(f"  {layer_name}: {num_tokens} tokens, {num_embeddings} embeddings")

    # Merge metadata
    first_metadata_file = shard_dirs[0] / "metadata.json"
    if first_metadata_file.exists():
        with open(first_metadata_file, 'r') as f:
            metadata = json.load(f)
        metadata['merged_from_shards'] = len(shard_dirs)
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\nMerge complete! Output: {output_dir}")

    if build_cache:
        print("\n" + "=" * 70)
        print("BUILDING CACHES")
        print("=" * 70)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from precompute_contextual_caches import build_cache_for_layer
        for layer_name in layer_dirs_to_merge:
            layer_dir = output_dir / layer_name
            print(f"\nBuilding cache for {layer_name}...")
            result = build_cache_for_layer((layer_dir, max_per_token, True))
            print(f"  {result[2]}")
        print(f"\nCaches built!")


def add_common_args(parser):
    """Add common arguments shared across all model-specific scripts."""
    parser.add_argument("--num-captions", type=int, default=10000,
                       help="Number of captions to process (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--max-per-token", type=int, default=MAX_CAPTIONS_PER_TOKEN,
                       help=f"Max embeddings per token (default: {MAX_CAPTIONS_PER_TOKEN})")
    parser.add_argument("--embedding-dtype", type=str, default=EMBEDDING_DTYPE,
                       choices=['float32', 'float16', 'float8'],
                       help=f"Embedding storage dtype (default: {EMBEDDING_DTYPE})")
    parser.add_argument("--dataset", type=str, default="vg", choices=["vg", "cc"],
                       help="Dataset: 'vg' for Visual Genome phrases (default)")
    parser.add_argument("--vg-file", type=str, default="vg_phrases.txt",
                       help="Path to VG phrases file")
    parser.add_argument("--tsv-file", type=str, default="Train_GCC-training.tsv",
                       help="Path to TSV file (for CC dataset)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--shard", type=int, default=None,
                       help="Shard index for multi-GPU processing")
    parser.add_argument("--num-shards", type=int, default=1,
                       help="Total number of shards")
    parser.add_argument("--merge-shards", action="store_true",
                       help="Merge shard outputs (run after all shards complete)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Skip building caches after merging")
    return parser


def run_extraction(args, model_name, default_layers, load_model_fn, model_type_description):
    """
    Main extraction loop shared across all models.

    Args:
        args: parsed argparse namespace
        model_name: HuggingFace model name
        default_layers: list of default layer indices
        load_model_fn: callable() -> (model_forward_fn, tokenizer, num_layers)
            model_forward_fn: callable(input_ids, attention_mask) -> hidden_states tuple
        model_type_description: string for metadata (e.g., "Molmo-7B-D LLM backbone")
    """
    layers = getattr(args, 'layers', None) or default_layers

    # Handle merge-shards mode
    if args.merge_shards:
        if args.output_dir is None:
            model_safe = model_name.replace("/", "_")
            base_dir = "molmo_data/contextual_llm_embeddings_vg" if args.dataset == "vg" else "molmo_data/contextual_llm_embeddings"
            output_dir = Path(base_dir) / model_safe
        else:
            output_dir = Path(args.output_dir)
        merge_shards(output_dir, args.num_shards, args.max_per_token, build_cache=not args.no_cache)
        return

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_captions = None if args.num_captions == -1 else args.num_captions
    apply_position_filter = (args.dataset == "cc")

    shard_info = f" [Shard {args.shard}/{args.num_shards}]" if args.shard is not None else ""

    print("=" * 70)
    print(f"CONTEXTUAL EMBEDDINGS EXTRACTION: {model_name}{shard_info}")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Type: {model_type_description}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Layers: {layers}")
    print(f"Captions: {'ALL' if num_captions is None else num_captions}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding dtype: {args.embedding_dtype}")
    print()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_safe = model_name.replace("/", "_")
        base_dir = "molmo_data/contextual_llm_embeddings_vg" if args.dataset == "vg" else "molmo_data/contextual_llm_embeddings"
        output_dir = Path(base_dir) / model_safe

    if args.shard is not None:
        output_dir = output_dir.parent / f"{output_dir.name}_shard{args.shard}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load model
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    load_start = time.time()
    model_forward_fn, tokenizer, num_layers = load_model_fn()
    print(f"Model loaded in {time.time() - load_start:.1f}s")
    print(f"Model has {num_layers} transformer layers")

    # Validate layers
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} out of range [0, {num_layers-1}]")
    print()

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    if args.dataset == "vg":
        captions = load_vg_phrases(args.vg_file, num_captions)
    else:
        captions = load_tsv_captions(args.tsv_file, num_captions)

    if args.shard is not None:
        total_captions = len(captions)
        shard_size = (total_captions + args.num_shards - 1) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total_captions)
        captions = captions[start_idx:end_idx]
        print(f"Shard {args.shard}: captions {start_idx} to {end_idx-1} ({len(captions)})")
    print()

    # Create layer directories
    layer_dirs = {}
    for layer_idx in layers:
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir = layer_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        layer_dirs[layer_idx] = {
            'layer_dir': layer_dir,
            'embeddings_dir': embeddings_dir,
            'counter': 0,
            'token_dict': {}
        }

    # Resume from checkpoint
    progress = load_progress(output_dir)
    start_offset = 0
    if progress:
        start_offset = progress['captions_processed']
        print(f"Resuming from {start_offset} captions processed")
        for layer_idx in layers:
            if str(layer_idx) in progress['layer_counters']:
                layer_dirs[layer_idx]['counter'] = progress['layer_counters'][str(layer_idx)]
            token_file = layer_dirs[layer_idx]['layer_dir'] / "token_embeddings.json"
            if token_file.exists():
                with open(token_file, 'r') as f:
                    layer_dirs[layer_idx]['token_dict'] = json.load(f)

    # Extract
    print("=" * 70)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 70)
    extract_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_embeddings, captions_processed = extract_contextual_embeddings(
        model_forward_fn=model_forward_fn,
        tokenizer=tokenizer,
        captions=captions,
        layers_to_extract=layers,
        batch_size=args.batch_size,
        layer_dirs=layer_dirs,
        device=device,
        output_dir=output_dir,
        start_offset=start_offset,
        max_captions_per_token=args.max_per_token,
        embedding_dtype=args.embedding_dtype,
        apply_position_filter=apply_position_filter
    )

    print(f"\nExtraction completed in {time.time() - extract_start:.1f}s")
    print()

    # Finalize
    print("=" * 70)
    print("FINALIZING AND SAVING")
    print("=" * 70)
    finalize_token_embeddings(layer_dirs, args.max_per_token)

    for layer_idx, layer_info in layer_dirs.items():
        token_dict = layer_info['token_dict']
        output_file = layer_info['layer_dir'] / "token_embeddings.json"
        with open(output_file, 'w') as f:
            json.dump(token_dict, f, indent=2)
        num_tokens = len(token_dict)
        num_emb = sum(len(embs) if isinstance(embs, list) else 0 for embs in token_dict.values())
        print(f"  Layer {layer_idx}: {num_tokens} tokens, {num_emb} embeddings")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': model_type_description,
        'layers_extracted': layers,
        'num_captions_processed': captions_processed,
        'max_captions_per_token': args.max_per_token,
        'embedding_dtype': args.embedding_dtype,
        'dataset': args.dataset,
        'data_source': 'Visual Genome phrases' if args.dataset == 'vg' else 'Conceptual Captions',
        'position_filtering': {
            'enabled': apply_position_filter,
            'skip_positions': [0, 1] if apply_position_filter else [],
        }
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print(f"DONE! Output: {output_dir}")
    print("=" * 70)
