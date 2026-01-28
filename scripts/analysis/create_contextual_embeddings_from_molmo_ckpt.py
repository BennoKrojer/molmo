#!/usr/bin/env python3
"""
Create contextual embeddings from a Molmo checkpoint's LLM (text-only, no vision).

This script extracts contextual embeddings from an unfrozen/finetuned LLM stored
in a Molmo checkpoint. Use this when the LLM was finetuned during training (ft_llm=True).

For frozen LLMs, use create_contextual_embeddings.py with the HuggingFace model instead.

Key features:
- Loads full Molmo model from checkpoint (model.pt)
- Processes pure text through transformer blocks (no vision backbone)
- Uses reservoir sampling to limit storage per token
- Supports multi-GPU sharding for parallel processing
- Compatible with existing contextual embedding cache format

Usage:
    # Test with small number of captions (single GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/create_contextual_embeddings_from_molmo_ckpt.py \
        --ckpt-path molmo_data/checkpoints/ablations/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze/step12000-unsharded \
        --num-captions 1000 --layers 1 2 4 8 16 24 30 31

    # Full extraction with 8 GPUs in parallel:
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python scripts/analysis/create_contextual_embeddings_from_molmo_ckpt.py \
            --ckpt-path <path> --num-captions -1 --shard $i --num-shards 8 --embedding-dtype float8 &
    done

    # Then merge shards:
    python scripts/analysis/create_contextual_embeddings_from_molmo_ckpt.py --merge-shards --num-shards 8 \
        --ckpt-path <path>
"""

import argparse
import gc
import json
import os
import random
import time
import sys
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Add project root to path for olmo imports BEFORE any imports that might use olmo
# This ensures we use the local olmo package, not any globally installed one
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Now we can safely import olmo from the local package
from olmo.config import ModelConfig, TrainConfig
from olmo.model import Molmo

from transformers import AutoTokenizer

# Import ml_dtypes for float8 support
try:
    import ml_dtypes
    HAVE_ML_DTYPES = True
except ImportError:
    HAVE_ML_DTYPES = False
    print("Warning: ml_dtypes not available. Float8 embeddings will not work.")

# Constants
BATCH_SIZE = 32  # Batch size for processing
SAVE_FREQUENCY = 2000  # Save progress every N captions
MAX_CAPTIONS_PER_TOKEN = 20  # Maximum embeddings per token (reservoir sampling)
EMBEDDING_DTYPE = 'float8'  # Default dtype for embeddings
VG_FILE = "vg_phrases.txt"


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


def load_vg_phrases(vg_file_path, num_phrases=None, shard=None, num_shards=None):
    """Load phrases from Visual Genome phrases file, optionally sharded."""
    print(f"Loading phrases from {vg_file_path}...")

    phrases = []
    with open(vg_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            phrase = line.strip()
            if phrase:
                phrases.append(phrase)

            if (i + 1) % 500000 == 0:
                print(f"  Loaded {i + 1} phrases...")

    total_phrases = len(phrases)
    print(f"Loaded {total_phrases} phrases from VG")

    # Apply sharding if requested
    if shard is not None and num_shards is not None:
        shard_size = total_phrases // num_shards
        start_idx = shard * shard_size
        end_idx = start_idx + shard_size if shard < num_shards - 1 else total_phrases
        phrases = phrases[start_idx:end_idx]
        print(f"Shard {shard}/{num_shards}: phrases {start_idx}-{end_idx} ({len(phrases)} phrases)")

    # Apply num_phrases limit after sharding
    if num_phrases is not None and num_phrases > 0:
        phrases = phrases[:num_phrases]
        print(f"Limited to {len(phrases)} phrases")

    return phrases


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

    # Save each layer's token embeddings
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


def extract_contextual_embeddings_molmo(
    model,
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
    apply_position_filter=False  # VG dataset: include all positions
):
    """
    Extract contextual embeddings from Molmo's LLM (text-only, no vision).

    This processes pure text through Molmo's transformer blocks without any vision inputs,
    extracting hidden states from specified layers.
    """

    total_embedding_count = 0
    captions_processed = start_offset

    # Skip already processed captions
    if start_offset > 0:
        captions = captions[start_offset:]
        print(f"\nSkipping first {start_offset} captions (already processed)")

    total_batches = (len(captions) + batch_size - 1) // batch_size
    print(f"\nProcessing {len(captions)} captions in {total_batches} batches")
    print(f"Reservoir sampling: max {max_captions_per_token} embeddings per token")
    print(f"Storage dtype: {embedding_dtype}")
    print(f"Position filtering: {'enabled (CC dataset)' if apply_position_filter else 'disabled (VG dataset - all positions)'}")

    # Track token occurrences for reservoir sampling
    token_seen_counts = {layer_idx: defaultdict(lambda: {'preferred_count': 0, 'fallback_count': 0})
                         for layer_idx in layers_to_extract}

    # Process captions in batches
    for batch_idx in tqdm(range(0, len(captions), batch_size), desc="Processing batches"):
        batch_captions = captions[batch_idx:batch_idx + batch_size]

        # Tokenize batch - pure text, no image tokens
        encodings = tokenizer(
            batch_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        )

        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # Forward pass through Molmo (text-only, no images)
        # This uses the transformer blocks to process text
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    last_logits_only=False,
                    # No images - just text through the transformer
                )

        all_hidden_states = outputs.hidden_states

        # Process each caption in the batch
        for sent_idx, caption in enumerate(batch_captions):
            # Get valid token positions
            valid_positions = torch.where(attention_mask[sent_idx] == 1)[0].cpu().numpy()
            token_ids_for_caption = input_ids[sent_idx][valid_positions]

            # Process each token position
            for pos_idx, token_pos in enumerate(valid_positions):
                # Position filtering: skip positions 0 and 1 (only for CC dataset)
                if apply_position_filter and pos_idx < 2:
                    continue

                token_id = token_ids_for_caption[pos_idx].item()
                token_str = tokenizer.decode([token_id])

                # Determine if preferred (pos >= 10) or fallback (2 <= pos < 10)
                # For VG (no position filter), treat all as preferred
                if apply_position_filter:
                    is_preferred = pos_idx >= 10
                else:
                    is_preferred = True  # All positions equal for VG

                # Use first layer for reservoir sampling decisions
                first_layer = layers_to_extract[0]
                first_layer_info = layer_dirs[first_layer]
                first_token_dict = first_layer_info['token_dict']

                # Initialize token entry
                if token_str not in first_token_dict:
                    first_token_dict[token_str] = {
                        'preferred': [],
                        'fallback': [],
                        'combined': []
                    }

                # Get counts
                counts = token_seen_counts[first_layer][token_str]

                if is_preferred:
                    counts['preferred_count'] += 1
                    reservoir_type = 'preferred'
                    count = counts['preferred_count']
                else:
                    counts['fallback_count'] += 1
                    reservoir_type = 'fallback'
                    count = counts['fallback_count']

                # Reservoir sampling decision
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

                # Store for all layers if decided
                if should_store:
                    for layer_idx in layers_to_extract:
                        hidden_state = all_hidden_states[layer_idx][sent_idx, token_pos].float().cpu().numpy()

                        # Check for NaN
                        if np.isnan(hidden_state).any():
                            raise ValueError(
                                f"NaN detected in hidden state!\n"
                                f"Layer {layer_idx}, token '{token_str}', caption: {caption[:100]}..."
                            )

                        layer_info = layer_dirs[layer_idx]
                        token_dict = layer_info['token_dict']

                        if token_str not in token_dict:
                            token_dict[token_str] = {
                                'preferred': [],
                                'fallback': [],
                                'combined': []
                            }

                        reservoir = token_dict[token_str][reservoir_type]
                        counter = layer_info['counter']

                        # Save embedding
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

        # Save progress periodically
        if captions_processed % SAVE_FREQUENCY < batch_size:
            total_captions = start_offset + len(captions)
            print(f"\nSaving progress at {captions_processed}/{total_captions} captions...")
            save_progress(output_dir, layer_dirs, captions_processed, total_captions, token_seen_counts)

        # Clear memory
        del all_hidden_states, outputs
        torch.cuda.empty_cache()

    print(f"\n✓ Extracted {total_embedding_count} embeddings across {len(layers_to_extract)} layers")
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

                # Combine: prefer preferred, supplement with fallback
                combined = preferred.copy()
                if len(combined) < max_captions_per_token:
                    needed = max_captions_per_token - len(combined)
                    combined.extend(fallback[:needed])

                token_dict[token_str] = combined


def save_metadata(output_dir, model_name, ckpt_path, layers_extracted, num_captions,
                  embedding_dtype, dataset, shard=None, num_shards=None):
    """Save extraction metadata."""
    metadata = {
        'model_name': model_name,
        'model_type': 'Molmo (LLM backbone only, text-only)',
        'checkpoint_path': str(ckpt_path),
        'layers_extracted': layers_extracted,
        'num_captions_processed': num_captions,
        'max_captions_per_token': MAX_CAPTIONS_PER_TOKEN,
        'embedding_dtype': embedding_dtype,
        'dataset': dataset,
        'data_source': 'Visual Genome phrases',
        'position_filtering': {
            'enabled': False,
            'description': 'All token positions included',
            'skip_positions': [],
            'preferred_positions': 'all',
            'fallback_positions': 'N/A'
        }
    }

    if shard is not None:
        metadata['shard'] = shard
        metadata['num_shards'] = num_shards

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_file}")


def merge_single_layer(args):
    """Merge a single layer from all shards. Helper for parallel processing."""
    import shutil
    layer_idx, shard_dirs, output_dir, max_per_token = args

    # Create merged layer directory
    layer_dir = output_dir / f"layer_{layer_idx}"
    embeddings_dir = layer_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Collect all token embeddings
    merged_token_dict = {}
    embedding_counter = 0

    for shard_dir in shard_dirs:
        shard_layer_dir = shard_dir / f"layer_{layer_idx}"
        shard_json = shard_layer_dir / "token_embeddings.json"

        if not shard_json.exists():
            continue

        with open(shard_json, 'r') as f:
            shard_data = json.load(f)

        shard_embeddings_dir = shard_layer_dir / "embeddings"

        for token_str, entries in shard_data.items():
            if not isinstance(entries, list):
                continue

            if token_str not in merged_token_dict:
                merged_token_dict[token_str] = []

            # Copy embeddings, respecting max_per_token limit
            for entry in entries:
                if len(merged_token_dict[token_str]) >= max_per_token:
                    break

                # Copy embedding file using shutil (faster than np.load/np.save)
                old_path = shard_embeddings_dir / entry['embedding_path'].split('/')[-1]
                if old_path.exists():
                    new_filename = f"emb_{embedding_counter:08d}.npy"
                    new_path = embeddings_dir / new_filename
                    shutil.copy2(old_path, new_path)

                    # Update entry with new path
                    new_entry = entry.copy()
                    new_entry['embedding_path'] = f"embeddings/{new_filename}"
                    merged_token_dict[token_str].append(new_entry)
                    embedding_counter += 1

    # Save merged token embeddings
    with open(layer_dir / "token_embeddings.json", 'w') as f:
        json.dump(merged_token_dict, f, indent=2)

    return layer_idx, len(merged_token_dict), embedding_counter


def merge_shards(base_output_dir, num_shards, layers, num_workers=8):
    """Merge sharded outputs into a single directory (parallel across layers)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n{'='*70}")
    print(f"MERGING {num_shards} SHARDS (parallel with {num_workers} workers)")
    print(f"{'='*70}")

    merged_dir = base_output_dir
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Find all shard directories
    shard_dirs = []
    for i in range(num_shards):
        shard_dir = Path(str(base_output_dir) + f"_shard{i}")
        if shard_dir.exists():
            shard_dirs.append(shard_dir)
        else:
            print(f"  WARNING: Shard {i} not found at {shard_dir}")

    if not shard_dirs:
        print("No shard directories found!")
        return

    print(f"Found {len(shard_dirs)} shard directories")
    print(f"Layers to merge: {layers}")
    print(f"Max embeddings per token: {MAX_CAPTIONS_PER_TOKEN}")

    # Prepare arguments for parallel processing
    merge_args = [(layer_idx, shard_dirs, merged_dir, MAX_CAPTIONS_PER_TOKEN) for layer_idx in layers]

    # Merge layers in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(merge_single_layer, args): args[0] for args in merge_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging layers"):
            layer_idx = futures[future]
            layer_idx, num_tokens, num_embeddings = future.result()
            results.append((layer_idx, num_tokens, num_embeddings))
            print(f"  ✓ Layer {layer_idx}: {num_tokens} tokens, {num_embeddings} embeddings")

    # Create merged metadata
    total_captions = 0
    for shard_dir in shard_dirs:
        meta_file = shard_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                shard_meta = json.load(f)
                total_captions += shard_meta.get('num_captions_processed', 0)

    merged_metadata = {
        'model_type': 'Molmo (LLM backbone only, text-only)',
        'layers_extracted': layers,
        'num_captions_processed': total_captions,
        'max_captions_per_token': MAX_CAPTIONS_PER_TOKEN,
        'dataset': 'vg',
        'data_source': 'Visual Genome phrases',
        'position_filtering': {
            'enabled': False,
            'description': 'All token positions included'
        },
        'merged_from_shards': num_shards
    }

    with open(merged_dir / "metadata.json", 'w') as f:
        json.dump(merged_metadata, f, indent=2)

    print(f"\n✓ Merge complete! Output: {merged_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create contextual embeddings from Molmo checkpoint's LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to Molmo checkpoint directory (containing model.pt and config.yaml)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: molmo_data/contextual_llm_embeddings_vg/<ckpt_name>)")
    parser.add_argument("--num-captions", type=int, default=-1,
                        help="Number of captions to process (-1 for all)")
    parser.add_argument("--layers", type=int, nargs='+', default=[1, 2, 4, 8, 16, 24, 30, 31],
                        help="Layers to extract (default: 1 2 4 8 16 24 30 31 for OLMo)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--embedding-dtype", type=str, default=EMBEDDING_DTYPE,
                        choices=['float32', 'float16', 'float8'],
                        help=f"Embedding storage dtype (default: {EMBEDDING_DTYPE})")
    parser.add_argument("--vg-file", type=str, default=VG_FILE,
                        help=f"Visual Genome phrases file (default: {VG_FILE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reservoir sampling")

    # Sharding for multi-GPU
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0 to num_shards-1)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards")
    parser.add_argument("--merge-shards", action="store_true",
                        help="Merge sharded outputs instead of extracting")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine output directory
    ckpt_path = Path(args.ckpt_path)
    ckpt_name = ckpt_path.parent.name if ckpt_path.name.startswith("step") else ckpt_path.name

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path("molmo_data/contextual_llm_embeddings_vg") / ckpt_name

    # Handle merging
    if args.merge_shards:
        if args.num_shards is None:
            raise ValueError("Must specify --num-shards when merging")
        merge_shards(base_output_dir, args.num_shards, args.layers)
        return

    # Adjust output for sharding
    if args.shard is not None:
        output_dir = Path(str(base_output_dir) + f"_shard{args.shard}")
    else:
        output_dir = base_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"CONTEXTUAL EMBEDDING EXTRACTION FROM MOLMO CHECKPOINT")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output: {output_dir}")
    print(f"Layers: {args.layers}")
    print(f"Embedding dtype: {args.embedding_dtype}")
    if args.shard is not None:
        print(f"Shard: {args.shard}/{args.num_shards}")
    print()

    # ===== LOAD MODEL =====
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    load_start = time.time()

    # Load config
    config_path = ckpt_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = TrainConfig.load(str(config_path))
    cfg.model.init_device = "cpu"

    # Get tokenizer info from config
    tokenizer_id = cfg.model.tokenizer.identifier
    print(f"  Tokenizer: {tokenizer_id}")

    # Create model
    print(f"  Creating Molmo model...")
    model = Molmo(cfg.model)

    # Load weights
    model_file = ckpt_path / "model.pt"
    model_size_gb = model_file.stat().st_size / (1024**3)
    print(f"  Loading weights from {model_file} ({model_size_gb:.1f} GB)...")

    weights = torch.load(model_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Moving to {device} (fp16)...")
    model = model.half().to(device).eval()
    torch.cuda.empty_cache()

    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")

    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")

    # ===== LOAD DATA =====
    print()
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    num_captions = None if args.num_captions == -1 else args.num_captions
    captions = load_vg_phrases(args.vg_file, num_captions, args.shard, args.num_shards)

    # ===== SETUP LAYER DIRECTORIES =====
    layers_to_extract = args.layers
    layer_dirs = {}
    for layer_idx in layers_to_extract:
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

    # Check for existing progress
    progress = load_progress(output_dir)
    start_offset = 0
    if progress is not None:
        start_offset = progress['captions_processed']
        print(f"\nResuming from {start_offset} captions...")

        # Load existing token embeddings
        for layer_idx, layer_info in layer_dirs.items():
            layer_dir = layer_info['layer_dir']
            json_file = layer_dir / "token_embeddings.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    layer_info['token_dict'] = json.load(f)
            if str(layer_idx) in progress['layer_counters']:
                layer_info['counter'] = progress['layer_counters'][str(layer_idx)]

    # ===== EXTRACT EMBEDDINGS =====
    print()
    print("=" * 70)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 70)

    extract_start = time.time()

    total_embeddings, captions_processed = extract_contextual_embeddings_molmo(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        layers_to_extract=layers_to_extract,
        batch_size=args.batch_size,
        layer_dirs=layer_dirs,
        device=device,
        output_dir=output_dir,
        start_offset=start_offset,
        max_captions_per_token=MAX_CAPTIONS_PER_TOKEN,
        embedding_dtype=args.embedding_dtype,
        apply_position_filter=False  # VG: all positions
    )

    extract_time = time.time() - extract_start
    print(f"\n✓ Extraction completed in {extract_time/60:.1f} minutes")

    # ===== FINALIZE =====
    print()
    print("=" * 70)
    print("FINALIZING")
    print("=" * 70)

    # Combine preferred and fallback
    finalize_token_embeddings(layer_dirs, MAX_CAPTIONS_PER_TOKEN)

    # Save final token embeddings
    for layer_idx, layer_info in layer_dirs.items():
        layer_dir = layer_info['layer_dir']
        token_dict = layer_info['token_dict']
        output_file = layer_dir / "token_embeddings.json"

        with open(output_file, 'w') as f:
            json.dump(token_dict, f, indent=2)

        print(f"  Layer {layer_idx}: saved {len(token_dict)} tokens")

    # Save metadata
    save_metadata(
        output_dir=output_dir,
        model_name=ckpt_name,
        ckpt_path=ckpt_path,
        layers_extracted=layers_to_extract,
        num_captions=captions_processed,
        embedding_dtype=args.embedding_dtype,
        dataset='vg',
        shard=args.shard,
        num_shards=args.num_shards
    )

    print(f"\n{'='*70}")
    print(f"✓ COMPLETE!")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Captions processed: {captions_processed}")

    if args.shard is not None:
        print(f"\nNOTE: This is shard {args.shard}/{args.num_shards}.")
        print(f"After all shards complete, run with --merge-shards to combine.")


if __name__ == "__main__":
    main()
