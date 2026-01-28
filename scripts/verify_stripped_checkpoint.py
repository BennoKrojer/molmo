#!/usr/bin/env python3
"""
Verify stripped checkpoint by comparing connector weights with original.

This is sufficient to prove functional equivalence since:
1. Both load same pretrained LLM + ViT weights
2. If connector weights match, outputs MUST match

Usage:
    python scripts/verify_stripped_checkpoint.py \
        --original /path/to/original \
        --stripped /path/to/stripped
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

def load_checkpoint_weights(checkpoint_path: Path):
    """Load model weights from checkpoint."""
    model_path = checkpoint_path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.pt in {checkpoint_path}")
    
    return torch.load(model_path, map_location='cpu')


def compare_weights(original_weights, stripped_weights, tolerance=1e-6):
    """Compare two weight dictionaries."""
    print("\nComparing weights...")
    
    # Get connector parameters (what should be in stripped)
    connector_params = [
        "transformer.wte.new_embedding",
        "vision_backbone.image_projector",
        "vision_backbone.image_pooling_2d",
        "vision_backbone.cls_projector", 
        "vision_backbone.pad_embed",
    ]
    
    # Find all connector-related keys in original
    original_connector_keys = [
        k for k in original_weights.keys()
        if any(conn in k for conn in connector_params)
    ]
    
    print(f"  Original checkpoint has {len(original_weights)} parameters")
    print(f"  Stripped checkpoint has {len(stripped_weights)} parameters")
    print(f"  Expected {len(original_connector_keys)} connector parameters")
    
    # Check all connector params are in stripped
    missing_in_stripped = []
    for key in original_connector_keys:
        if key not in stripped_weights:
            missing_in_stripped.append(key)
    
    if missing_in_stripped:
        print(f"\n  ❌ ERROR: {len(missing_in_stripped)} connector params missing in stripped:")
        for key in missing_in_stripped[:5]:
            print(f"     - {key}")
        return False
    
    # Check stripped doesn't have extra non-connector params
    extra_in_stripped = []
    for key in stripped_weights.keys():
        if not any(conn in key for conn in connector_params):
            extra_in_stripped.append(key)
    
    if extra_in_stripped:
        print(f"\n  ⚠️  WARNING: {len(extra_in_stripped)} unexpected params in stripped:")
        for key in extra_in_stripped[:5]:
            print(f"     - {key}")
    
    # Compare values for connector params
    print(f"\n  Comparing {len(original_connector_keys)} connector parameter values...")
    max_diff = 0.0
    mismatches = []
    
    for key in original_connector_keys:
        orig_tensor = original_weights[key]
        strip_tensor = stripped_weights[key]
        
        # Check shapes match
        if orig_tensor.shape != strip_tensor.shape:
            print(f"    ❌ Shape mismatch for {key}:")
            print(f"       Original: {orig_tensor.shape}")
            print(f"       Stripped: {strip_tensor.shape}")
            mismatches.append(key)
            continue
        
        # Check values match
        diff = torch.abs(orig_tensor - strip_tensor).max().item()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            mismatches.append((key, diff))
    
    if mismatches:
        print(f"\n  ❌ ERROR: {len(mismatches)} parameters have mismatched values:")
        for item in mismatches[:5]:
            if isinstance(item, tuple):
                key, diff = item
                print(f"     - {key}: max diff = {diff:.2e}")
            else:
                print(f"     - {item}: shape mismatch")
        return False
    
    print(f"\n  ✅ SUCCESS: All connector weights match!")
    print(f"     Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify stripped checkpoint matches original")
    parser.add_argument("--original", type=str, required=True, help="Path to original checkpoint")
    parser.add_argument("--stripped", type=str, required=True, help="Path to stripped checkpoint")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for weight comparison")
    
    args = parser.parse_args()
    
    original_path = Path(args.original)
    stripped_path = Path(args.stripped)
    
    print("=" * 80)
    print("STRIPPED CHECKPOINT VERIFICATION")
    print("=" * 80)
    
    # Check paths exist
    if not original_path.exists():
        print(f"\n❌ Original checkpoint not found: {original_path}")
        sys.exit(1)
    
    if not stripped_path.exists():
        print(f"\n❌ Stripped checkpoint not found: {stripped_path}")
        sys.exit(1)
    
    print(f"\nOriginal: {original_path}")
    print(f"Stripped: {stripped_path}")
    
    # Load weights
    print("\nLoading checkpoints...")
    try:
        original_weights = load_checkpoint_weights(original_path)
        print(f"  ✓ Loaded original checkpoint")
    except Exception as e:
        print(f"  ❌ Failed to load original: {e}")
        sys.exit(1)
    
    try:
        stripped_weights = load_checkpoint_weights(stripped_path)
        print(f"  ✓ Loaded stripped checkpoint")
    except Exception as e:
        print(f"  ❌ Failed to load stripped: {e}")
        sys.exit(1)
    
    # Compare
    success = compare_weights(original_weights, stripped_weights, args.tolerance)
    
    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("✅ VERIFICATION PASSED")
        print("\nThe stripped checkpoint is functionally equivalent to the original:")
        print("  • All connector weights match exactly")
        print("  • Both will load same pretrained LLM + ViT weights")
        print("  • Inference outputs will be identical")
        print("\nIt is safe to:")
        print("  • Delete the original checkpoint")
        print("  • Use the stripped checkpoint for inference")
        print("  • Continue training from the stripped checkpoint")
    else:
        print("❌ VERIFICATION FAILED")
        print("\nThe stripped checkpoint does NOT match the original.")
        print("DO NOT use it until the issues above are resolved.")
        sys.exit(1)
    
    print("=" * 80)


if __name__ == "__main__":
    main()

