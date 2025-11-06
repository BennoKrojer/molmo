#!/usr/bin/env python3
"""
Trim contextual embeddings to a target number of captions.

This script safely removes embeddings that were extracted from captions beyond a target count.
It operates in dry-run mode first to show what would be deleted.

Usage:
    # Dry run (safe, just shows what would happen)
    python scripts/analysis/trim_contextual_embeddings.py --model-dir molmo_data/contextual_llm_embeddings/meta-llama_Meta-Llama-3-8B --target-captions 1000000 --dry-run
    
    # Actually trim
    python scripts/analysis/trim_contextual_embeddings.py --model-dir molmo_data/contextual_llm_embeddings/meta-llama_Meta-Llama-3-8B --target-captions 1000000
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict


def load_caption_order(tsv_file: str, num_captions: int) -> Set[str]:
    """Load the first N captions from TSV to identify which ones to keep."""
    print(f"Loading first {num_captions:,} captions from {tsv_file}...")
    captions_to_keep = set()
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_captions:
                break
            
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                caption = parts[0].strip()
                if caption:
                    captions_to_keep.add(caption)
            
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1:,} captions...")
    
    print(f"Loaded {len(captions_to_keep):,} unique captions to keep\n")
    return captions_to_keep


def analyze_trimming(model_dir: Path, target_captions: int, tsv_file: str) -> Dict:
    """Analyze what would be trimmed without actually deleting."""
    
    # Load progress
    progress_file = model_dir / "progress.json"
    if not progress_file.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_file}")
    
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    current_captions = progress['captions_processed']
    
    print(f"Model directory: {model_dir}")
    print(f"Current captions processed: {current_captions:,}")
    print(f"Target captions: {target_captions:,}")
    print(f"Captions to remove: {current_captions - target_captions:,}\n")
    
    if current_captions <= target_captions:
        print("✓ Already at or below target caption count. Nothing to do!")
        return None
    
    # Load the set of captions to keep
    captions_to_keep = load_caption_order(tsv_file, target_captions)
    
    # Analyze each layer
    layer_stats = {}
    total_embeddings_to_delete = 0
    total_embeddings_to_keep = 0
    
    for layer_dir in sorted(model_dir.glob("layer_*")):
        layer_name = layer_dir.name
        token_file = layer_dir / "token_embeddings.json"
        
        if not token_file.exists():
            continue
        
        print(f"Analyzing {layer_name}...")
        
        with open(token_file, 'r') as f:
            token_dict = json.load(f)
        
        embeddings_to_delete = []
        embeddings_to_keep_count = 0
        tokens_affected = 0
        tokens_completely_removed = 0
        
        for token_str, embeddings in token_dict.items():
            # Handle both list and dict formats
            if isinstance(embeddings, dict):
                embeddings_list = embeddings.get('preferred', []) + embeddings.get('fallback', [])
            else:
                embeddings_list = embeddings
            
            token_had_deletions = False
            for emb in embeddings_list:
                if not isinstance(emb, dict):
                    continue
                
                caption = emb.get('caption', '')
                
                if caption not in captions_to_keep:
                    # This embedding should be deleted
                    embedding_path = emb.get('embedding_path')
                    if embedding_path:
                        embeddings_to_delete.append(layer_dir / embedding_path)
                    token_had_deletions = True
                else:
                    embeddings_to_keep_count += 1
            
            if token_had_deletions:
                tokens_affected += 1
                # Check if ALL embeddings for this token are deleted
                remaining = sum(1 for emb in embeddings_list 
                              if isinstance(emb, dict) and emb.get('caption', '') in captions_to_keep)
                if remaining == 0:
                    tokens_completely_removed += 1
        
        layer_stats[layer_name] = {
            'embeddings_to_delete': len(embeddings_to_delete),
            'embeddings_to_keep': embeddings_to_keep_count,
            'tokens_affected': tokens_affected,
            'tokens_completely_removed': tokens_completely_removed,
            'files_to_delete': embeddings_to_delete
        }
        
        total_embeddings_to_delete += len(embeddings_to_delete)
        total_embeddings_to_keep += embeddings_to_keep_count
        
        print(f"  Embeddings to delete: {len(embeddings_to_delete):,}")
        print(f"  Embeddings to keep: {embeddings_to_keep_count:,}")
        print(f"  Tokens affected: {tokens_affected:,}")
        print(f"  Tokens completely removed: {tokens_completely_removed:,}\n")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total embeddings to delete: {total_embeddings_to_delete:,}")
    print(f"Total embeddings to keep: {total_embeddings_to_keep:,}")
    print(f"Percentage to delete: {100 * total_embeddings_to_delete / (total_embeddings_to_delete + total_embeddings_to_keep):.1f}%")
    print(f"{'='*80}\n")
    
    return {
        'model_dir': model_dir,
        'current_captions': current_captions,
        'target_captions': target_captions,
        'captions_to_keep': captions_to_keep,
        'layer_stats': layer_stats,
        'total_embeddings_to_delete': total_embeddings_to_delete,
        'total_embeddings_to_keep': total_embeddings_to_keep,
    }


def perform_trimming(analysis_results: Dict, create_backup: bool = True):
    """Actually perform the trimming based on analysis results."""
    
    model_dir = analysis_results['model_dir']
    target_captions = analysis_results['target_captions']
    captions_to_keep = analysis_results['captions_to_keep']
    layer_stats = analysis_results['layer_stats']
    
    print(f"\n{'='*80}")
    print(f"PERFORMING TRIMMING")
    print(f"{'='*80}\n")
    
    # Create backup if requested
    if create_backup:
        backup_dir = model_dir.parent / f"{model_dir.name}_backup_json_only"
        if backup_dir.exists():
            print(f"⚠️  Backup already exists: {backup_dir}")
            response = input("Overwrite existing backup? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborting.")
                return
            shutil.rmtree(backup_dir)
        
        print(f"Creating backup (JSON files only - much faster): {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Only backup JSON files, not the massive number of .npy embeddings
        for json_file in ['progress.json', 'metadata.json']:
            src = model_dir / json_file
            if src.exists():
                shutil.copy2(src, backup_dir / json_file)
        
        # Backup token_embeddings.json for each layer
        for layer_dir in model_dir.glob("layer_*"):
            backup_layer_dir = backup_dir / layer_dir.name
            backup_layer_dir.mkdir(exist_ok=True)
            
            token_file = layer_dir / "token_embeddings.json"
            if token_file.exists():
                shutil.copy2(token_file, backup_layer_dir / "token_embeddings.json")
        
        print(f"✓ JSON backup created (embedding files not backed up - too large)\n")
    
    # Process each layer
    for layer_dir in sorted(model_dir.glob("layer_*")):
        layer_name = layer_dir.name
        
        if layer_name not in layer_stats:
            continue
        
        print(f"Processing {layer_name}...")
        
        token_file = layer_dir / "token_embeddings.json"
        
        with open(token_file, 'r') as f:
            token_dict = json.load(f)
        
        # Create new token dict with only kept embeddings
        new_token_dict = {}
        deleted_files = 0
        
        for token_str, embeddings in token_dict.items():
            # Handle both list and dict formats
            if isinstance(embeddings, dict):
                # Dict format with preferred/fallback
                new_preferred = []
                new_fallback = []
                
                for emb in embeddings.get('preferred', []):
                    if isinstance(emb, dict):
                        caption = emb.get('caption', '')
                        if caption in captions_to_keep:
                            new_preferred.append(emb)
                        else:
                            # Delete the embedding file
                            embedding_path = emb.get('embedding_path')
                            if embedding_path:
                                file_path = layer_dir / embedding_path
                                if file_path.exists():
                                    file_path.unlink()
                                    deleted_files += 1
                
                for emb in embeddings.get('fallback', []):
                    if isinstance(emb, dict):
                        caption = emb.get('caption', '')
                        if caption in captions_to_keep:
                            new_fallback.append(emb)
                        else:
                            # Delete the embedding file
                            embedding_path = emb.get('embedding_path')
                            if embedding_path:
                                file_path = layer_dir / embedding_path
                                if file_path.exists():
                                    file_path.unlink()
                                    deleted_files += 1
                
                # Only keep token if it has remaining embeddings
                if new_preferred or new_fallback:
                    new_token_dict[token_str] = {
                        'preferred': new_preferred,
                        'fallback': new_fallback,
                        'combined': []
                    }
            
            else:
                # List format
                new_embeddings = []
                
                for emb in embeddings:
                    if isinstance(emb, dict):
                        caption = emb.get('caption', '')
                        if caption in captions_to_keep:
                            new_embeddings.append(emb)
                        else:
                            # Delete the embedding file
                            embedding_path = emb.get('embedding_path')
                            if embedding_path:
                                file_path = layer_dir / embedding_path
                                if file_path.exists():
                                    file_path.unlink()
                                    deleted_files += 1
                
                # Only keep token if it has remaining embeddings
                if new_embeddings:
                    new_token_dict[token_str] = new_embeddings
        
        # Save updated token dict
        with open(token_file, 'w') as f:
            json.dump(new_token_dict, f, indent=2)
        
        print(f"  Deleted {deleted_files:,} embedding files")
        print(f"  Kept {len(new_token_dict):,} tokens (removed {len(token_dict) - len(new_token_dict):,} tokens)")
    
    # Update progress.json
    progress_file = model_dir / "progress.json"
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    progress['captions_processed'] = target_captions
    progress['total_captions'] = target_captions
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"\n✓ Updated progress.json: captions_processed = {target_captions:,}")
    
    # Update metadata.json if it exists
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['num_captions_processed'] = target_captions
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Updated metadata.json")
    
    print(f"\n{'='*80}")
    print(f"TRIMMING COMPLETE!")
    print(f"{'='*80}\n")
    if create_backup:
        print(f"Backup saved to: {backup_dir}")
        print(f"If everything looks good, you can delete the backup to save space.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Trim contextual embeddings to target caption count",
        epilog="""
Examples:
  # Trim by model directory path
  python %(prog)s --model-dir molmo_data/contextual_llm_embeddings/meta-llama_Meta-Llama-3-8B --target-captions 1000000
  
  # Trim using shorthand model name
  python %(prog)s --model meta-llama/Meta-Llama-3-8B --target-captions 1000000
  
  # Dry run to see what would be deleted
  python %(prog)s --model meta-llama/Meta-Llama-3-8B --dry-run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", type=str,
                       help="Path to model directory (e.g., molmo_data/contextual_llm_embeddings/meta-llama_Meta-Llama-3-8B)")
    parser.add_argument("--model", type=str,
                       help="Model name (e.g., meta-llama/Meta-Llama-3-8B) - will be converted to directory path")
    parser.add_argument("--base-dir", type=str, default="molmo_data/contextual_llm_embeddings",
                       help="Base directory for embeddings (default: molmo_data/contextual_llm_embeddings)")
    parser.add_argument("--target-captions", type=int, default=1000000,
                       help="Target number of captions to keep (default: 1000000)")
    parser.add_argument("--tsv-file", type=str, default="Train_GCC-training.tsv",
                       help="TSV file with captions (to determine order)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run: analyze but don't delete anything")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup before trimming (not recommended)")
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif args.model:
        # Convert model name to directory path
        # e.g., "meta-llama/Meta-Llama-3-8B" -> "meta-llama_Meta-Llama-3-8B"
        model_dir_name = args.model.replace("/", "_")
        model_dir = Path(args.base_dir) / model_dir_name
    else:
        parser.error("Must specify either --model-dir or --model")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Analyze what would be trimmed
    analysis_results = analyze_trimming(model_dir, args.target_captions, args.tsv_file)
    
    if analysis_results is None:
        return
    
    if args.dry_run:
        print("✓ DRY RUN COMPLETE - No files were modified")
        print("\nTo actually perform the trimming, run without --dry-run flag")
    else:
        print("\n⚠️  WARNING: This will permanently delete embeddings!")
        print(f"   {analysis_results['total_embeddings_to_delete']:,} embedding files will be deleted")
        
        response = input("\nContinue with trimming? (yes/no): ")
        if response.lower() == 'yes':
            perform_trimming(analysis_results, create_backup=not args.no_backup)
        else:
            print("Aborting.")


if __name__ == "__main__":
    main()

