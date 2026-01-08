#!/usr/bin/env python3
"""
Generate individual max token embedding plots (one per model) for LaTeX tabular.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("paper_plots/paper_figures_output/l2norm_plots/individual")
output_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path("analysis_results/max_token_embeddings")

models = [
    ("olmo-7b", "vit-l-14-336", ""),
    ("olmo-7b", "dinov2-large-336", ""),
    ("olmo-7b", "siglip", ""),
    ("llama3-8b", "vit-l-14-336", ""),
    ("llama3-8b", "dinov2-large-336", ""),
    ("llama3-8b", "siglip", ""),
    ("qwen2-7b", "vit-l-14-336", "_seed10"),
    ("qwen2-7b", "dinov2-large-336", ""),
    ("qwen2-7b", "siglip", ""),
]

LLM_DISPLAY = {'olmo-7b': 'OLMo-7B', 'llama3-8b': 'LLaMA3-8B', 'qwen2-7b': 'Qwen2-7B'}
ENC_DISPLAY = {'vit-l-14-336': 'CLIP', 'dinov2-large-336': 'DINOv2', 'siglip': 'SigLIP'}

for llm, enc, suffix in models:
    ckpt = f"train_mlp-only_pixmo_cap_resize_{llm}_{enc}{suffix}_step12000-unsharded"
    json_path = data_dir / f"{ckpt}_max_token.json"

    if not json_path.exists():
        print(f"SKIP: {json_path} not found")
        continue

    with open(json_path) as f:
        data = json.load(f)

    values = np.array(data['max_embedding_values'])
    l2_norm = data['max_info']['l2_norm']
    layer = data['max_info']['layer']
    stats = data['embedding_stats']

    # Create individual figure
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    ax.hist(values, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Mark percentiles
    ax.axvline(x=stats['percentiles']['p1'], color='red', linestyle=':', alpha=0.7)
    ax.axvline(x=stats['percentiles']['p99'], color='red', linestyle=':', alpha=0.7)

    ax.set_title(f"{LLM_DISPLAY[llm]} + {ENC_DISPLAY[enc]}\nL2={l2_norm:.0f}, layer={layer}",
                fontsize=11, fontweight='bold')
    ax.text(0.95, 0.95, f"mean={stats['mean']:.2f}\nstd={stats['std']:.2f}\nmax={stats['max']:.2f}",
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Embedding Value', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_name = f"max_token_{llm}_{enc}"
    plt.savefig(output_dir / f"{out_name}.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"{out_name}.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / out_name}.pdf")
    plt.close()

print(f"\nDone! Individual plots saved to {output_dir}")
