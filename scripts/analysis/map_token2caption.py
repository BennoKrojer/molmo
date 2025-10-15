#!/usr/bin/env python3
"""
Create a fast lookup table mapping token IDs to captions they appear in.

This is a simpler, faster version of create_contextual_embeddings.py that only
creates a token -> captions mapping without any LLM inference.
"""

from transformers import AutoTokenizer
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
import random

def load_tsv_captions(tsv_file_path, num_captions=None):
    """Load captions from TSV file."""
    print(f"Loading captions from {tsv_file_path}...")
    
    captions = []
    with open(tsv_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_captions and i >= num_captions:
                break
            
            # Split by tab to get caption and URL
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                caption = parts[0].strip()
                if caption:  # Only add non-empty captions
                    captions.append(caption)
            
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1} captions...")
    
    print(f"Loaded {len(captions)} captions from TSV file")
    return captions

def create_token_to_caption_map(tokenizer, captions, batch_size=1024, max_captions_per_token=100):
    """
    Create a mapping from token ID to captions using reservoir sampling for unbiased selection.
    
    Two-tier filtering by position:
    - Only considers tokens at position >= 2 (skips positions 0 and 1)
    - Prefers tokens at position >= 10, but falls back to position >= 2 if needed to fill max_captions_per_token
    
    Args:
        tokenizer: HuggingFace tokenizer
        captions: List of caption strings
        batch_size: Batch size for tokenization
        max_captions_per_token: Maximum number of captions to store per token (default: 100)
    
    Returns:
        dict: token_id -> list of (caption_idx, caption_text, position_in_caption)
    """
    
    print(f"\nCreating token -> caption mapping with reservoir sampling...")
    print(f"Processing {len(captions)} captions in batches of {batch_size}")
    print(f"Filtering: position >= 2, preferring position >= 10")
    
    # Separate reservoirs for preferred (pos >= 10) and fallback (2 <= pos < 10)
    token_to_preferred = defaultdict(list)  # Position >= 10
    token_to_fallback = defaultdict(list)   # 2 <= Position < 10
    
    # Track how many times each token has been seen in each category (for reservoir sampling)
    token_preferred_count = defaultdict(int)
    token_fallback_count = defaultdict(int)
    
    # Statistics
    total_tokens = 0
    filtered_tokens = 0  # Tokens at position >= 2
    
    # Process captions in batches
    for batch_start in tqdm(range(0, len(captions), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(captions))
        batch_captions = captions[batch_start:batch_end]
        
        # Tokenize batch
        try:
            tokenized = tokenizer(
                batch_captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
        except Exception as e:
            print(f"Error tokenizing batch at {batch_start}: {e}")
            continue
        
        # Process each caption in the batch
        for local_idx, caption in enumerate(batch_captions):
            caption_idx = batch_start + local_idx
            
            # Get valid token positions (exclude padding)
            valid_positions = (attention_mask[local_idx] == 1).nonzero(as_tuple=True)[0]
            token_ids = input_ids[local_idx][valid_positions]
            
            # Track total tokens
            total_tokens += len(token_ids)
            
            # For each token in this caption
            for pos_idx, token_id in enumerate(token_ids):
                # Skip positions 0 and 1
                if pos_idx < 2:
                    continue
                
                filtered_tokens += 1
                token_id_int = token_id.item()
                
                caption_entry = {
                    'caption_idx': caption_idx,
                    'caption': caption,
                    'position': pos_idx
                }
                
                # Determine if this is preferred (pos >= 10) or fallback (2 <= pos < 10)
                if pos_idx >= 10:
                    # Preferred category
                    token_preferred_count[token_id_int] += 1
                    
                    # Reservoir sampling for preferred
                    if len(token_to_preferred[token_id_int]) < max_captions_per_token:
                        token_to_preferred[token_id_int].append(caption_entry)
                    else:
                        j = random.randint(0, token_preferred_count[token_id_int] - 1)
                        if j < max_captions_per_token:
                            token_to_preferred[token_id_int][j] = caption_entry
                else:
                    # Fallback category (2 <= pos < 10)
                    token_fallback_count[token_id_int] += 1
                    
                    # Reservoir sampling for fallback
                    if len(token_to_fallback[token_id_int]) < max_captions_per_token:
                        token_to_fallback[token_id_int].append(caption_entry)
                    else:
                        j = random.randint(0, token_fallback_count[token_id_int] - 1)
                        if j < max_captions_per_token:
                            token_to_fallback[token_id_int][j] = caption_entry
    
    # Combine preferred and fallback: prioritize preferred, supplement with fallback if needed
    print(f"\nCombining preferred (pos >= 10) and fallback (2 <= pos < 10) entries...")
    token_to_captions = {}
    
    all_tokens = set(token_to_preferred.keys()) | set(token_to_fallback.keys())
    for token_id in all_tokens:
        preferred = token_to_preferred.get(token_id, [])
        fallback = token_to_fallback.get(token_id, [])
        
        # Start with preferred entries
        combined = preferred.copy()
        
        # If we need more to reach max_captions_per_token, add fallback
        if len(combined) < max_captions_per_token:
            needed = max_captions_per_token - len(combined)
            combined.extend(fallback[:needed])
        
        token_to_captions[token_id] = combined
    
    print(f"\n✓ Mapping complete!")
    print(f"  Total tokens processed: {total_tokens:,}")
    print(f"  Tokens at position >= 2: {filtered_tokens:,}")
    print(f"  Unique tokens: {len(token_to_captions):,}")
    
    return token_to_captions, total_tokens

def save_token_map(token_to_captions, tokenizer, output_file, total_tokens, total_captions):
    """Save the token -> caption mapping to JSON file."""
    
    print(f"\nPreparing data for saving...")
    
    # Convert to a more readable format with token text
    token_map = {}
    for token_id, caption_list in tqdm(token_to_captions.items(), desc="Converting to readable format"):
        token_text = tokenizer.decode([token_id])
        
        token_map[str(token_id)] = {
            'token_text': token_text,
            'frequency': len(caption_list),
            'captions': caption_list
        }
    
    # Create metadata
    metadata = {
        'total_captions': total_captions,
        'total_tokens_processed': total_tokens,
        'unique_tokens': len(token_map),
        'tokenizer': tokenizer.name_or_path,
        'description': 'Mapping from token ID to captions where the token appears',
        'token_map': token_map
    }
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\n💾 Saved token map to: {output_file}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Unique tokens: {len(token_map):,}")
    print(f"   Total captions: {total_captions:,}")

def main():
    parser = argparse.ArgumentParser(description="Create token ID to caption mapping")
    parser.add_argument("--tsv-file", type=str, default="Train_GCC-training.tsv",
                       help="Path to TSV file with captions")
    parser.add_argument("--num-captions", type=int, default=None,
                       help="Number of captions to process (default: None = all captions in TSV)")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for tokenization (default: 1024)")
    parser.add_argument("--max-captions-per-token", type=int, default=100,
                       help="Maximum number of captions to store per token (default: 100)")
    parser.add_argument("--tokenizer", type=str, default="allenai/OLMo-7B-1024-preview",
                       help="HuggingFace tokenizer to use (default: allenai/OLMo-7B-1024-preview)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/token_to_caption_map",
                       help="Output directory (default: analysis_results/token_to_caption_map)")
    args = parser.parse_args()
    
    print("="*80)
    print("TOKEN ID -> CAPTION MAPPING")
    print("="*80)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"TSV file: {args.tsv_file}")
    print(f"Number of captions: {'ALL' if args.num_captions is None else f'{args.num_captions:,}'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max captions per token: {args.max_captions_per_token}")
    print("="*80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Load captions
    captions = load_tsv_captions(args.tsv_file, num_captions=args.num_captions)
    
    # Create token -> caption mapping
    token_to_captions, total_tokens = create_token_to_caption_map(
        tokenizer, 
        captions, 
        batch_size=args.batch_size,
        max_captions_per_token=args.max_captions_per_token
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the mapping
    num_captions_str = "all" if args.num_captions is None else str(args.num_captions)
    output_file = output_dir / f"token_map_{num_captions_str}_captions.json"
    save_token_map(token_to_captions, tokenizer, output_file, total_tokens, len(captions))
    
    print(f"\n✅ Complete! Token -> caption mapping saved to {output_file}")
    print(f"\nYou can now quickly look up which captions contain any token ID.")
    print(f"Example usage in Python:")
    print(f"  import json")
    print(f"  with open('{output_file}', 'r') as f:")
    print(f"      data = json.load(f)")
    print(f"  token_map = data['token_map']")
    print(f"  # Look up token ID 12345")
    print(f"  print(token_map['12345'])")

if __name__ == "__main__":
    main()

