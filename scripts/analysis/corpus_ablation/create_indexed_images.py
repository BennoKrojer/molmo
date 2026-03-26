#!/usr/bin/env python3
"""Create symlinks for PixMoCap validation images as indexed files (00000.jpg, etc.).

This is needed by the release repo's evaluate_interpretability.py which expects
images named by index rather than hash.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from olmo.data.pixmo_datasets import PixMoCap


def main():
    output_dir = Path("analysis_results/pixmo_cap_validation_indexed")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_images = 100
    ds = PixMoCap(split="validation", mode="captions")

    for i in range(num_images):
        ex = ds.get(i, np.random)
        src = ex["image"]
        # Determine extension
        ext = Path(src).suffix if Path(src).suffix else ".jpg"
        if not ext:
            ext = ".jpg"
        dst = output_dir / f"{i:05d}{ext}"
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst)
        if i < 3:
            print(f"  {dst.name} -> {src}")

    print(f"Created {num_images} symlinks in {output_dir}")


if __name__ == "__main__":
    main()
