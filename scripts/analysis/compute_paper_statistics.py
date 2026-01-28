#!/usr/bin/env python3
"""
Compute statistics for paper placeholders.
All numbers in the paper should be traceable to this script.

Usage:
    python scripts/analysis/compute_paper_statistics.py
"""

from collections import Counter
from pathlib import Path

def analyze_vg_phrases():
    """Analyze Visual Genome phrases file for uniqueness."""
    vg_file = Path(__file__).parent.parent.parent / "vg_phrases.txt"

    print("=" * 60)
    print("VISUAL GENOME PHRASES ANALYSIS")
    print("=" * 60)
    print(f"File: {vg_file}")

    with open(vg_file, 'r') as f:
        phrases = [line.strip() for line in f if line.strip()]

    print(f"\nTotal non-empty lines: {len(phrases)}")
    print(f"Unique phrases: {len(set(phrases))}")
    print(f"Difference: {len(phrases) - len(set(phrases))}")

    # Count duplicates
    counts = Counter(phrases)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    print(f"\nNumber of phrases that appear more than once: {len(duplicates)}")
    print(f"Total extra occurrences: {sum(v-1 for v in duplicates.values())}")

    print(f"\nTop 10 most duplicated phrases:")
    for phrase, count in sorted(duplicates.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count}x: \"{phrase}\"")

    return {
        'total_lines': len(phrases),
        'unique': len(set(phrases)),
        'num_duplicated_phrases': len(duplicates),
        'total_extra_occurrences': sum(v-1 for v in duplicates.values())
    }


def analyze_pixmo_cap():
    """Analyze PixMo-Cap dataset statistics."""
    from datasets import load_dataset
    import numpy as np

    print("\n" + "=" * 60)
    print("PIXMO-CAP DATASET ANALYSIS")
    print("=" * 60)

    print("Loading dataset from HuggingFace...")
    ds = load_dataset('allenai/pixmo-cap', split='train')

    print(f"\nTotal rows: {len(ds)}")

    # Count unique images
    image_urls = [row['image_url'] for row in ds]
    unique_images = len(set(image_urls))
    print(f"Unique images: {unique_images}")

    # Analyze captions
    captions = [row['caption'] for row in ds]
    word_counts = [len(c.split()) for c in captions]

    # Count sentences (split by period, filter empty)
    sentence_counts = [len([s for s in c.split('.') if s.strip()]) for c in captions]

    print(f"\nCaption statistics:")
    print(f"  Total captions: {len(captions)}")
    print(f"  Avg words per caption: {np.mean(word_counts):.1f} (std: {np.std(word_counts):.1f})")
    print(f"  Median words: {np.median(word_counts):.1f}")
    print(f"  Min/Max words: {min(word_counts)} / {max(word_counts)}")
    print(f"  Avg sentences per caption: {np.mean(sentence_counts):.1f} (std: {np.std(sentence_counts):.1f})")
    print(f"  Median sentences: {np.median(sentence_counts):.1f}")

    return {
        'total_rows': len(ds),
        'unique_images': unique_images,
        'avg_words': np.mean(word_counts),
        'std_words': np.std(word_counts),
        'median_words': np.median(word_counts),
        'avg_sentences': np.mean(sentence_counts),
        'std_sentences': np.std(sentence_counts),
        'median_sentences': np.median(sentence_counts)
    }


if __name__ == "__main__":
    print("Computing paper statistics...\n")

    # VG analysis (no dependencies)
    vg_stats = analyze_vg_phrases()

    # PixMo-Cap analysis (requires datasets library)
    try:
        pixmo_stats = analyze_pixmo_cap()
    except ImportError as e:
        print(f"\nSkipping PixMo-Cap analysis: {e}")
        print("Install with: pip install datasets")
        pixmo_stats = None

    # Summary for paper
    print("\n" + "=" * 60)
    print("SUGGESTED PAPER TEXT")
    print("=" * 60)

    print(f"\nVisual Genome:")
    print(f"  '{vg_stats['total_lines']:,} phrases ({vg_stats['unique']:,} unique)'")

    if pixmo_stats:
        print(f"\nPixMo-Cap:")
        print(f"  '{pixmo_stats['total_rows']:,} image-caption pairs "
              f"({pixmo_stats['unique_images']:,} unique images) "
              f"with captions averaging {pixmo_stats['avg_words']:.0f} words "
              f"and {pixmo_stats['avg_sentences']:.0f} sentences'")
