"""Count how many valid left/right examples exist with current thresholds.

Usage:
    python scripts/count_left_right_examples.py
"""
import numpy as np
from tqdm import tqdm

from olmo.data.pixmo_datasets import PixMoPointsLeftRight


def count_valid_examples(split="train", max_examples=None):
    """Count valid left/right examples in the dataset."""
    dataset = PixMoPointsLeftRight(split=split, kind="basic")
    
    total_raw = len(dataset.data)
    valid_count = 0
    left_count = 0
    right_count = 0
    middle_count = 0
    no_objects_count = 0
    
    # Check how many examples have valid left/right objects
    max_check = max_examples if max_examples else total_raw
    
    print(f"Checking {max_check} examples from {split} split...")
    
    for i in tqdm(range(max_check)):
        ex = dataset.data[i]
        has_valid = False
        
        for label, points in zip(ex["label"], ex["points"]):
            if len(points) == 0:
                continue
            
            points_array = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
            position = PixMoPointsLeftRight.classify_position(points_array)
            
            if position == 'left':
                left_count += 1
                has_valid = True
            elif position == 'right':
                right_count += 1
                has_valid = True
            elif position == 'middle':
                middle_count += 1
        
        if has_valid:
            valid_count += 1
        else:
            no_objects_count += 1
    
    return {
        'total_raw': total_raw,
        'checked': max_check,
        'valid_examples': valid_count,
        'left_objects': left_count,
        'right_objects': right_count,
        'middle_objects': middle_count,
        'no_valid_objects': no_objects_count
    }


def main():
    print("="*80)
    print("Counting Left/Right Dataset Examples")
    print("="*80)
    
    # Count train split
    print("\n[TRAIN SPLIT]")
    train_stats = count_valid_examples("train", max_examples=10000)  # Sample first 10k
    
    print(f"\nResults (sampled first {train_stats['checked']} examples):")
    print(f"  Valid examples (have ≥1 left/right object): {train_stats['valid_examples']:,}")
    print(f"  Examples with no valid objects: {train_stats['no_valid_objects']:,}")
    print(f"  Percentage valid: {train_stats['valid_examples']/train_stats['checked']*100:.1f}%")
    
    print(f"\nObject counts:")
    print(f"  Left objects: {train_stats['left_objects']:,}")
    print(f"  Right objects: {train_stats['right_objects']:,}")
    print(f"  Middle objects (skipped): {train_stats['middle_objects']:,}")
    
    # Extrapolate to full dataset
    if train_stats['checked'] < train_stats['total_raw']:
        valid_ratio = train_stats['valid_examples'] / train_stats['checked']
        estimated_valid = int(train_stats['total_raw'] * valid_ratio)
        print(f"\n  Estimated valid examples in full train split: ~{estimated_valid:,}")
    
    # Count validation split
    print("\n" + "="*80)
    print("[VALIDATION SPLIT]")
    val_stats = count_valid_examples("validation")  # Check all validation examples
    
    print(f"\nResults (all {val_stats['checked']} examples):")
    print(f"  Valid examples (have ≥1 left/right object): {val_stats['valid_examples']:,}")
    print(f"  Examples with no valid objects: {val_stats['no_valid_objects']:,}")
    print(f"  Percentage valid: {val_stats['valid_examples']/val_stats['checked']*100:.1f}%")
    
    print(f"\nObject counts:")
    print(f"  Left objects: {val_stats['left_objects']:,}")
    print(f"  Right objects: {val_stats['right_objects']:,}")
    print(f"  Middle objects (skipped): {val_stats['middle_objects']:,}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

