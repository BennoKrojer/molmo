#!/usr/bin/env python3
"""
Create contextual embeddings from Qwen2-VL's LLM backbone (text-only, no vision).

This script is similar to create_contextual_embeddings.py but specifically for Qwen2-VL.
Since Qwen2-VL uses a finetuned version of Qwen2, we need to extract embeddings from
the actual Qwen2-VL model to get properly aligned contextual embeddings.

Key differences from the vanilla Qwen2 version:
1. Uses Qwen2VLForConditionalGeneration model
2. Processes pure text without any vision tokens
3. Accesses the LLM backbone directly

Supports multi-GPU processing via data sharding (--shard and --num-shards).

Usage:
    # Test with small number of captions (single GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py \
        --num-captions 1000 --layers 1 2 4 8 16 24 26 27

    # Full extraction with 8 GPUs in parallel (run each in separate terminal):
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py \
            --dataset vg --num-captions -1 --shard $i --num-shards 8 --embedding-dtype float8 &
    done
    
    # Then merge shards AND build caches in one step:
    python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py --merge-shards --num-shards 8 --dataset vg
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

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Import ml_dtypes for float8 support
try:
    import ml_dtypes
    HAVE_ML_DTYPES = True
except ImportError:
    HAVE_ML_DTYPES = False
    print("Warning: ml_dtypes not available. Float8 embeddings will not work.")

# Constants
BATCH_SIZE = 32  # Batch size for processing (smaller than vanilla due to model size)
SAVE_FREQUENCY = 1000  # Save progress every N captions
MAX_CAPTIONS_PER_TOKEN = 20  # Maximum embeddings per token (reservoir sampling)
EMBEDDING_DTYPE = 'float16'  # Default dtype for embeddings


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


def extract_contextual_embeddings_qwen2vl(
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
    apply_position_filter=True
):
    """
    Extract contextual embeddings from Qwen2-VL's LLM backbone (text-only).
    
    This processes pure text through Qwen2-VL without any vision inputs,
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
            max_length=512
        )
        
        # Move to device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Forward pass through LLM only (no vision)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Access the LLM directly through model.model
                # This bypasses vision processing entirely
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        
        all_hidden_states = outputs.hidden_states
        
        # Process each caption in the batch
        for sent_idx, caption in enumerate(batch_captions):
            # Get valid token positions
            valid_positions = torch.where(attention_mask[sent_idx] == 1)[0].cpu().numpy()
            token_ids = input_ids[sent_idx][valid_positions]
            
            # Process each token position
            for pos_idx, token_pos in enumerate(valid_positions):
                # Position filtering: skip positions 0 and 1 (only for CC dataset)
                if apply_position_filter and pos_idx < 2:
                    continue
                
                token_id = token_ids[pos_idx].item()
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
                        hidden_state = all_hidden_states[layer_idx][sent_idx, token_pos].cpu().numpy()
                        
                        # Check for NaN
                        if np.isnan(hidden_state).any():
                            raise ValueError(f"NaN detected in hidden state! Layer {layer_idx}, token '{token_str}'")
                        
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


def merge_single_layer(args):
    """Merge a single layer from all shards. Helper for parallel processing."""
    import shutil
    layer_name, shard_dirs, output_dir, max_per_token = args
    
    # Create merged layer directory
    merged_layer_dir = output_dir / layer_name
    merged_embeddings_dir = merged_layer_dir / "embeddings"
    merged_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all token embeddings
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
            
            # Copy embeddings and update paths
            for entry in entries:
                if len(merged_tokens[token_str]) >= max_per_token:
                    break
                
                # Copy embedding file
                src_path = shard_layer_dir / entry['embedding_path']
                if src_path.exists():
                    new_filename = f"emb_{embedding_counter:08d}.npy"
                    dst_path = merged_embeddings_dir / new_filename
                    shutil.copy2(src_path, dst_path)
                    
                    # Update entry with new path
                    new_entry = entry.copy()
                    new_entry['embedding_path'] = f"embeddings/{new_filename}"
                    merged_tokens[token_str].append(new_entry)
                    embedding_counter += 1
    
    # Save merged token embeddings
    with open(merged_layer_dir / "token_embeddings.json", 'w') as f:
        json.dump(merged_tokens, f, indent=2)
    
    return layer_name, len(merged_tokens), embedding_counter


def merge_shards(output_dir, num_shards, max_per_token=MAX_CAPTIONS_PER_TOKEN, build_cache=True, num_workers=8):
    """Merge shard outputs into a single directory and optionally build caches."""
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    output_dir = Path(output_dir)
    print(f"Merging {num_shards} shards into {output_dir}")
    
    # Find all shard directories
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
    
    # Get layers from first shard
    first_shard = shard_dirs[0]
    layer_dirs_to_merge = [d.name for d in first_shard.iterdir() if d.is_dir() and d.name.startswith("layer_")]
    print(f"Layers to merge: {layer_dirs_to_merge}")
    print(f"Merging {len(layer_dirs_to_merge)} layers in parallel with {num_workers} workers...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for parallel processing
    merge_args = [(layer_name, shard_dirs, output_dir, max_per_token) for layer_name in layer_dirs_to_merge]
    
    # Merge layers in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(merge_single_layer, args): args[0] for args in merge_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging layers"):
            layer_name = futures[future]
            layer_name, num_tokens, num_embeddings = future.result()
            results.append((layer_name, num_tokens, num_embeddings))
            print(f"  ✓ {layer_name}: {num_tokens} tokens, {num_embeddings} embeddings")
    
    print(f"\nAll layers merged!")
    
    # Merge metadata from first shard
    first_metadata_file = shard_dirs[0] / "metadata.json"
    if first_metadata_file.exists():
        with open(first_metadata_file, 'r') as f:
            metadata = json.load(f)
        metadata['merged_from_shards'] = len(shard_dirs)
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Merge complete! Output: {output_dir}")
    
    # Build caches for each layer
    if build_cache:
        print("\n" + "=" * 70)
        print("BUILDING CACHES")
        print("=" * 70)
        
        # Import cache building function
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from precompute_contextual_caches import build_cache_for_layer
        
        for layer_name in layer_dirs_to_merge:
            layer_dir = output_dir / layer_name
            print(f"\nBuilding cache for {layer_name}...")
            result = build_cache_for_layer((layer_dir, max_per_token, True))
            print(f"  {result[2]}")
        
        print(f"\n✓ Caches built!")
    
    print("\nYou can now delete the shard directories if desired.")


def main():
    parser = argparse.ArgumentParser(description="Create contextual embeddings from Qwen2-VL LLM backbone")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="Qwen2-VL model name")
    parser.add_argument("--num-captions", type=int, default=10000,
                       help="Number of captions to process (-1 for all)")
    parser.add_argument("--layers", type=int, nargs='+', default=[1, 2, 4, 8, 16, 24, 26, 27],
                       help="Layers to extract embeddings from (default: 1 2 4 8 16 24 26 27)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--max-per-token", type=int, default=MAX_CAPTIONS_PER_TOKEN,
                       help=f"Max embeddings per token (default: {MAX_CAPTIONS_PER_TOKEN})")
    parser.add_argument("--embedding-dtype", type=str, default=EMBEDDING_DTYPE,
                       choices=['float32', 'float16', 'float8'],
                       help=f"Embedding storage dtype (default: {EMBEDDING_DTYPE})")
    parser.add_argument("--dataset", type=str, default="vg", choices=["vg", "cc"],
                       help="Dataset to use: 'vg' for Visual Genome phrases (default), 'cc' for Conceptual Captions")
    parser.add_argument("--vg-file", type=str, default="vg_phrases.txt",
                       help="Path to VG phrases file (default: vg_phrases.txt)")
    parser.add_argument("--tsv-file", type=str, default="Train_GCC-training.tsv",
                       help="Path to TSV file with captions (for CC dataset)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    # Multi-GPU sharding
    parser.add_argument("--shard", type=int, default=None,
                       help="Shard index (0 to num-shards-1) for multi-GPU processing")
    parser.add_argument("--num-shards", type=int, default=1,
                       help="Total number of shards for multi-GPU processing")
    parser.add_argument("--merge-shards", action="store_true",
                       help="Merge shard outputs into final directory (run after all shards complete)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Skip building cache files after merging (only with --merge-shards)")
    args = parser.parse_args()
    
    # Handle merge-shards mode
    if args.merge_shards:
        if args.output_dir is None:
            model_safe = args.model_name.replace("/", "_")
            base_dir = "molmo_data/contextual_llm_embeddings_vg" if args.dataset == "vg" else "molmo_data/contextual_llm_embeddings"
            output_dir = Path(base_dir) / model_safe
        else:
            output_dir = Path(args.output_dir)
        merge_shards(output_dir, args.num_shards, args.max_per_token, build_cache=not args.no_cache)
        return
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    num_captions = None if args.num_captions == -1 else args.num_captions
    
    # Determine position filtering based on dataset (VG: no filtering, CC: filter positions 0-1)
    apply_position_filter = (args.dataset == "cc")
    
    # Sharding info
    shard_info = ""
    if args.shard is not None:
        shard_info = f" [Shard {args.shard}/{args.num_shards}]"
    
    print("=" * 70)
    print(f"QWEN2-VL CONTEXTUAL EMBEDDINGS EXTRACTION{shard_info}")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset.upper()} ({'Visual Genome phrases' if args.dataset == 'vg' else 'Conceptual Captions'})")
    print(f"Layers: {args.layers}")
    print(f"Captions/Phrases: {'ALL' if num_captions is None else num_captions}")
    if args.shard is not None:
        print(f"Sharding: shard {args.shard} of {args.num_shards}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max per token: {args.max_per_token}")
    print(f"Embedding dtype: {args.embedding_dtype}")
    print(f"Position filtering: {'enabled (skip pos 0-1)' if apply_position_filter else 'disabled (all positions)'}")
    print()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_safe = args.model_name.replace("/", "_")
        base_dir = "molmo_data/contextual_llm_embeddings_vg" if args.dataset == "vg" else "molmo_data/contextual_llm_embeddings"
        output_dir = Path(base_dir) / model_safe
    
    # Add shard suffix if sharding
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Get tokenizer from processor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded in {time.time() - load_start:.1f}s")
    
    # Get number of layers
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} transformer layers")
    
    # Validate layer indices
    for layer_idx in args.layers:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} out of range [0, {num_layers-1}]")
    
    print()
    
    # Load captions/phrases
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    if args.dataset == "vg":
        captions = load_vg_phrases(args.vg_file, num_captions)
    else:
        captions = load_tsv_captions(args.tsv_file, num_captions)
    
    # Shard the data if using multi-GPU
    if args.shard is not None:
        total_captions = len(captions)
        shard_size = (total_captions + args.num_shards - 1) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total_captions)
        captions = captions[start_idx:end_idx]
        print(f"Shard {args.shard}: processing captions {start_idx} to {end_idx-1} ({len(captions)} captions)")
    print()
    
    # Create layer directories
    layer_dirs = {}
    for layer_idx in args.layers:
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
    if progress:
        start_offset = progress['captions_processed']
        print(f"Resuming from {start_offset} captions processed")
        
        # Restore counters and token dicts
        for layer_idx in args.layers:
            if str(layer_idx) in progress['layer_counters']:
                layer_dirs[layer_idx]['counter'] = progress['layer_counters'][str(layer_idx)]
            
            token_file = layer_dirs[layer_idx]['layer_dir'] / "token_embeddings.json"
            if token_file.exists():
                with open(token_file, 'r') as f:
                    layer_dirs[layer_idx]['token_dict'] = json.load(f)
    
    # Extract embeddings
    print("=" * 70)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 70)
    extract_start = time.time()
    
    total_embeddings, captions_processed = extract_contextual_embeddings_qwen2vl(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        layers_to_extract=args.layers,
        batch_size=args.batch_size,
        layer_dirs=layer_dirs,
        device=device,
        output_dir=output_dir,
        start_offset=start_offset,
        max_captions_per_token=args.max_per_token,
        embedding_dtype=args.embedding_dtype,
        apply_position_filter=apply_position_filter
    )
    
    extract_time = time.time() - extract_start
    print(f"\n✓ Extraction completed in {extract_time:.1f}s ({extract_time/60:.1f} min)")
    print()
    
    # Finalize and save
    print("=" * 70)
    print("FINALIZING AND SAVING")
    print("=" * 70)
    
    finalize_token_embeddings(layer_dirs, args.max_per_token)
    
    # Save final token embeddings
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
        'model_name': args.model_name,
        'model_type': 'Qwen2-VL (LLM backbone only, no vision)',
        'layers_extracted': args.layers,
        'num_captions_processed': captions_processed,
        'max_captions_per_token': args.max_per_token,
        'embedding_dtype': args.embedding_dtype,
        'dataset': args.dataset,
        'data_source': 'Visual Genome phrases' if args.dataset == 'vg' else 'Conceptual Captions (TSV)',
        'position_filtering': {
            'enabled': apply_position_filter,
            'description': 'Only tokens at position >= 2' if apply_position_filter else 'All token positions included',
            'skip_positions': [0, 1] if apply_position_filter else [],
            'preferred_positions': '>= 10' if apply_position_filter else 'all',
            'fallback_positions': '2-9' if apply_position_filter else 'N/A'
        }
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 70)
    print("✓ DONE!")
    print(f"  Total time: {time.time() - load_start:.1f}s")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

