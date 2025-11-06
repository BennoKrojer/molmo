#!/usr/bin/env python3
"""
Strip frozen LLM and ViT weights from checkpoints to save disk space.

For checkpoints where ft_llm=false and ft_vit=false, we only need the connector weights.
This script removes the frozen weights from model.pt and optim.pt to drastically reduce
checkpoint size (from ~50-60GB to ~1GB).

Usage:
    python scripts/strip_frozen_weights.py <checkpoint_dir>
    python scripts/strip_frozen_weights.py /path/to/checkpoints --dry-run
"""

import argparse
import shutil
from pathlib import Path
import torch
import sys


# Connector parameters that we want to KEEP
CONNECTOR_PARAMS = [
    "vision_backbone.image_pooling_2d",
    "vision_backbone.image_projector", 
    "vision_backbone.cls_projector",
    "vision_backbone.pad_embed",
    "transformer.wte.new_embedding",
]

# Frozen parameters that we want to REMOVE
FROZEN_PARAMS = [
    "vision_backbone.image_vit",  # ViT encoder
    "transformer.wte.embedding",  # LLM embeddings
    "transformer.wte.weight",
    "transformer.wpe",
    "transformer.blocks",
    "transformer.block_groups", 
    "transformer.ln_f",
    "transformer.ff_out",
]


def is_connector_param(param_name: str) -> bool:
    """Check if parameter is part of the connector (should be kept)."""
    return any(conn in param_name for conn in CONNECTOR_PARAMS)


def is_frozen_param(param_name: str) -> bool:
    """Check if parameter is frozen (should be removed)."""
    return any(frozen in param_name for frozen in FROZEN_PARAMS)


def strip_model_state(model_state: dict, dry_run: bool = False) -> dict:
    """Remove frozen parameters from model state dict."""
    total_params = len(model_state)
    kept_params = {}
    removed_params = []
    
    kept_size_mb = 0
    removed_size_mb = 0
    
    for name, param in model_state.items():
        param_size_mb = param.numel() * param.element_size() / (1024 * 1024)
        
        if is_connector_param(name):
            kept_params[name] = param
            kept_size_mb += param_size_mb
        elif is_frozen_param(name):
            removed_params.append(name)
            removed_size_mb += param_size_mb
        else:
            # Unknown parameter - be conservative and keep it
            print(f"  WARNING: Unknown parameter '{name}' - keeping it")
            kept_params[name] = param
            kept_size_mb += param_size_mb
    
    print(f"  Model state: {total_params} params → {len(kept_params)} params")
    print(f"  Kept: {kept_size_mb:.1f} MB, Removed: {removed_size_mb:.1f} MB")
    print(f"  Space savings: {removed_size_mb:.1f} MB ({100*removed_size_mb/(kept_size_mb+removed_size_mb):.1f}%)")
    
    if not dry_run:
        return kept_params
    return model_state


def strip_optim_state(optim_state: dict, dry_run: bool = False) -> dict:
    """Remove optimizer state for frozen parameters."""
    if 'state' not in optim_state:
        return optim_state
    
    total_states = len(optim_state['state'])
    kept_state = {}
    removed_count = 0
    
    # Build mapping of param_id to param_name
    id_to_name = {}
    if 'param_groups' in optim_state:
        for group in optim_state['param_groups']:
            if 'param_names' in group and 'params' in group:
                for name, id in zip(group['param_names'], group['params']):
                    id_to_name[id] = name
    
    kept_size_mb = 0
    removed_size_mb = 0
    
    for param_id, state in optim_state['state'].items():
        param_name = id_to_name.get(param_id, "unknown")
        
        # Calculate size
        state_size_mb = 0
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state_size_mb += value.numel() * value.element_size() / (1024 * 1024)
        
        if is_connector_param(param_name):
            kept_state[param_id] = state
            kept_size_mb += state_size_mb
        elif is_frozen_param(param_name):
            removed_count += 1
            removed_size_mb += state_size_mb
        else:
            # Unknown - keep it
            kept_state[param_id] = state
            kept_size_mb += state_size_mb
    
    print(f"  Optim state: {total_states} param states → {len(kept_state)} param states")
    print(f"  Kept: {kept_size_mb:.1f} MB, Removed: {removed_size_mb:.1f} MB")
    
    if not dry_run:
        new_optim_state = dict(optim_state)
        new_optim_state['state'] = kept_state
        return new_optim_state
    return optim_state


def strip_checkpoint(checkpoint_dir: Path, backup: bool = True, dry_run: bool = False):
    """Strip frozen weights from a checkpoint directory."""
    model_path = checkpoint_dir / "model.pt"
    optim_path = checkpoint_dir / "optim.pt"
    
    if not model_path.exists():
        print(f"⚠️  No model.pt found in {checkpoint_dir}, skipping")
        return
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing {checkpoint_dir.name}...")
    
    # Backup if requested
    if backup and not dry_run:
        backup_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.backup"
        if not backup_dir.exists():
            print(f"  Creating backup at {backup_dir}")
            shutil.copytree(checkpoint_dir, backup_dir)
    
    # Process model.pt
    print("  Loading model.pt...")
    model_state = torch.load(model_path, map_location='cpu')
    orig_model_size = model_path.stat().st_size / (1024 * 1024 * 1024)
    
    stripped_model = strip_model_state(model_state, dry_run)
    
    if not dry_run:
        print("  Saving stripped model.pt...")
        torch.save(stripped_model, model_path)
        new_model_size = model_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"  Model size: {orig_model_size:.2f} GB → {new_model_size:.2f} GB")
    
    # Process optim.pt if it exists
    if optim_path.exists():
        print("  Loading optim.pt...")
        optim_state = torch.load(optim_path, map_location='cpu')
        orig_optim_size = optim_path.stat().st_size / (1024 * 1024 * 1024)
        
        stripped_optim = strip_optim_state(optim_state, dry_run)
        
        if not dry_run:
            print("  Saving stripped optim.pt...")
            torch.save(stripped_optim, optim_path)
            new_optim_size = optim_path.stat().st_size / (1024 * 1024 * 1024)
            print(f"  Optim size: {orig_optim_size:.2f} GB → {new_optim_size:.2f} GB")
    
    if not dry_run:
        total_saved = (orig_model_size + (orig_optim_size if optim_path.exists() else 0)) - \
                     (new_model_size + (new_optim_size if optim_path.exists() else 0))
        print(f"  ✅ Total saved: {total_saved:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Strip frozen weights from checkpoints")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory or parent directory containing multiple checkpoints")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup before stripping")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--pattern", default="step*", help="Pattern to match checkpoint directories (default: step*)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    # Check if this is a single checkpoint or directory containing checkpoints
    if (checkpoint_path / "model.pt").exists():
        # Single checkpoint
        checkpoints = [checkpoint_path]
    else:
        # Directory containing checkpoints
        checkpoints = sorted(checkpoint_path.glob(args.pattern))
        checkpoints = [c for c in checkpoints if c.is_dir() and (c / "model.pt").exists()]
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_path}")
        print(f"   Looking for directories matching '{args.pattern}' with model.pt inside")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoint(s) to process")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")
    
    total_before = 0
    total_after = 0
    
    for checkpoint_dir in checkpoints:
        try:
            strip_checkpoint(
                checkpoint_dir, 
                backup=not args.no_backup,
                dry_run=args.dry_run
            )
        except Exception as e:
            print(f"❌ Error processing {checkpoint_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'Would process' if args.dry_run else 'Processed'} {len(checkpoints)} checkpoint(s)")


if __name__ == "__main__":
    main()

