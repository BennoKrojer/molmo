#!/usr/bin/env python3
"""
Plot corpus size ablation: X = corpus size %, Y = avg interpretable tokens % across layers.
One line per model.

Usage:
    python scripts/analysis/corpus_ablation/plot_corpus_ablation.py \
        --eval-dir analysis_results/corpus_ablation/evaluation \
        --output analysis_results/corpus_ablation/corpus_ablation_plot.pdf
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODEL_DISPLAY = {
    'olmo-vit': 'OLMo + CLIP',
    'llama-siglip': 'LLaMA + SigLIP',
    'qwen-dino': 'Qwen2 + DINOv2',
}

MODEL_COLORS = {
    'olmo-vit': '#1f77b4',
    'llama-siglip': '#ff7f0e',
    'qwen-dino': '#2ca02c',
}

CORPUS_SIZES = [0.1, 1, 10, 100]

# Consistent layers to use for averaging (must be present across all pcts)
TARGET_LAYERS = {
    'olmo-vit': {0, 8, 16, 31},
    'llama-siglip': {0, 8, 16, 31},
    'qwen-dino': {0, 8, 16, 27},
}


def parse_pct(pct_str):
    """Convert dir suffix to float: '0pct'->0.1, '1pct'->1.0, '10pct'->10.0, '100pct'->100.0"""
    n = int(pct_str.replace("pct", ""))
    return 0.1 if n == 0 else float(n)


def load_eval_results(eval_dir):
    """Load evaluation results. Returns {model_key: {pct_float: avg_interpretability}}."""
    eval_dir = Path(eval_dir)
    results = {}

    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue

        results_file = subdir / "evaluation_results.json"
        if not results_file.exists():
            continue

        # Parse dir name: e.g., "olmo-vit_10pct"
        name = subdir.name
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].endswith("pct"):
            continue

        model_key = parts[0]
        pct = parse_pct(parts[1])

        with open(results_file) as f:
            data = json.load(f)

        # Filter to consistent target layers only
        target = TARGET_LAYERS.get(model_key)
        fractions = [entry["interpretable_fraction"] for entry in data
                    if "interpretable_fraction" in entry
                    and (target is None or entry["layer"] in target)]

        if not fractions:
            continue

        avg = np.mean(fractions) * 100  # Convert to percentage
        layer_details = {str(entry["layer"]): round(entry["interpretable_fraction"] * 100, 1)
                        for entry in data if "interpretable_fraction" in entry
                        and (target is None or entry["layer"] in target)}

        if model_key not in results:
            results[model_key] = {}
        results[model_key][pct] = avg

        print(f"  {name} ({pct}%): avg={avg:.1f}% across {len(fractions)} layers")
        print(f"    Per-layer: {layer_details}")

    return results


def plot(results, output_path):
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for model_key in sorted(results.keys()):
        pcts = sorted(results[model_key].keys())
        vals = [results[model_key][p] for p in pcts]

        label = MODEL_DISPLAY.get(model_key, model_key)
        color = MODEL_COLORS.get(model_key, None)

        ax.plot(pcts, vals, 'o-', label=label, color=color, markersize=6, linewidth=2)

    ax.set_xlabel("Corpus Size (%)", fontsize=11)
    ax.set_ylabel("Interpretable Tokens (%)\n(avg. across layers)", fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks(CORPUS_SIZES)
    ax.set_xticklabels([f"{s}%" for s in CORPUS_SIZES])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    # Also save PNG
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\nSaved: {output_path}")
    print(f"Saved: {png_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", default="analysis_results/corpus_ablation/corpus_ablation_plot.pdf")
    args = parser.parse_args()

    print("Loading evaluation results...")
    results = load_eval_results(args.eval_dir)

    if not results:
        print("ERROR: No results found!")
        return

    print(f"\nFound {len(results)} models:")
    for model_key, pcts in sorted(results.items()):
        print(f"  {model_key}: {sorted(pcts.keys())}")

    plot(results, args.output)


if __name__ == "__main__":
    main()
