#!/usr/bin/env python3
"""
Create off-the-shelf VLM plots.

Generates the unified 3-subplot figure for all off-the-shelf VLMs:
Qwen2-VL-7B, Molmo-7B-D, LLaVA-1.5-7B.

Layout matches create_lineplot_unified.py exactly:
- 3 subplots side-by-side (one per method)
- 3 model lines per subplot
- Shared legend on the right
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_JSON = SCRIPT_DIR / "data.json"
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "offtheshelf"


def load_data():
    """Load data from data.json."""
    with open(DATA_JSON, 'r') as f:
        return json.load(f)


def create_unified_plot(models_data):
    """
    Create 3-subplot figure matching create_lineplot_unified.py exactly.

    models_data: dict of {model_key: {'nn': {layer: val}, 'logitlens': ..., 'contextual': ...}}
    """
    # Model display names and visual encoding (matches main figure style)
    model_configs = [
        ('qwen2vl',        'Qwen2-VL-7B',      'tab:blue',   'o'),
        ('qwen2.5-vl-32b', 'Qwen2.5-VL-32B',   'tab:red',    'D'),
        ('molmo-7b',       'Molmo-7B-D',        'tab:orange', 's'),
        ('llava-1.5',      'LLaVA-1.5-7B',     'tab:green',  '^'),
    ]

    # X-axis: union of layers across all models
    # 7B models: [0,1,2,4,8,16,24,26,27,30,31]  32B: [0,1,2,4,8,16,32,48,56,62,63]
    all_layers = sorted({0, 1, 2, 4, 8, 16, 24, 26, 27, 30, 31, 32, 48, 56, 62, 63})
    clean_ticks = [0, 8, 16, 32, 48, 63]  # updated for 64-layer model

    # 3-subplot layout matching main figure exactly
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    sns.set_style("whitegrid")

    subplot_configs = [
        {'ax': axes[0], 'method': 'nn',         'title': 'a) EmbeddingLens',    'show_ylabel': True},
        {'ax': axes[1], 'method': 'logitlens',   'title': 'b) LogitLens',        'show_ylabel': False},
        {'ax': axes[2], 'method': 'contextual',  'title': 'c) LatentLens (Ours)','show_ylabel': False},
    ]

    handles_dict = {}

    for cfg in subplot_configs:
        ax = cfg['ax']
        method = cfg['method']

        for model_key, display_name, color, marker in model_configs:
            if model_key not in models_data:
                continue
            layer_data = {int(k): v for k, v in models_data[model_key].get(method, {}).items()}
            if not layer_data:
                continue

            layers = sorted(layer_data.keys())
            values = [layer_data[l] for l in layers]

            line, = ax.plot(layers, values, marker=marker, color=color,
                            linewidth=2, markersize=8)
            if display_name not in handles_dict:
                handles_dict[display_name] = line

        # Axes formatting — matches create_lineplot_unified.py exactly
        ax.set_xlabel('Layer', fontsize=16)
        if cfg['show_ylabel']:
            ax.set_ylabel('% of interpretable tokens', fontsize=14)
        ax.set_title(cfg['title'], fontsize=20, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)
        ax.set_xticks(clean_ticks)
        ax.set_xticklabels([str(t) for t in clean_ticks])
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

    # Shared legend on the right — matches create_lineplot_unified.py
    ordered_handles = [handles_dict[name] for _, name, _, _ in model_configs if name in handles_dict]
    ordered_labels  = [name for _, name, _, _ in model_configs if name in handles_dict]
    fig.legend(ordered_handles, ordered_labels,
               loc='center left',
               bbox_to_anchor=(0.88, 0.5),
               ncol=1,
               fontsize=12,
               framealpha=0.9,
               handlelength=2.0,
               handletextpad=0.5)

    plt.tight_layout()
    plt.subplots_adjust(right=0.87, wspace=0.18)
    return fig


def create_individual_plots(nn_data, logitlens_data, contextual_data):
    """Create individual plots for each method."""
    figs = {}

    methods = {
        'nn': ('EmbeddingLens', nn_data, 'tab:blue'),
        'logitlens': ('LogitLens', logitlens_data, 'tab:orange'),
        'contextual': ('LatentLens (Ours)', contextual_data, 'tab:green')
    }

    for method_key, (method_name, data, color) in methods.items():
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        layers = sorted(data.keys())
        values = [data[l] for l in layers]
        ax.plot(layers, values, marker='o', color=color, linewidth=3,
               markersize=12, label='Qwen2-VL-7B-Instruct')

        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Interpretability %', fontsize=14)
        ax.set_title(f'Qwen2-VL: {method_name}', fontsize=16, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13, framealpha=0.95)

        plt.tight_layout()
        figs[method_key] = fig

    return figs


def main():
    parser = argparse.ArgumentParser(description='Generate off-the-shelf VLM unified plot')
    args = parser.parse_args()

    # Load data from single source of truth
    print("Loading data from data.json...")
    data = load_data()

    MODEL_KEYS = ['qwen2vl', 'qwen2.5-vl-32b', 'molmo-7b', 'llava-1.5']
    models_data = {}
    for key in MODEL_KEYS:
        if key in data:
            models_data[key] = data[key]
            m = data[key]
            print(f"  {key}: nn={len(m.get('nn',{}))}, logitlens={len(m.get('logitlens',{}))}, contextual={len(m.get('contextual',{}))} layers")
        else:
            print(f"  WARNING: {key} not in data.json — skipping")

    if not models_data:
        print("ERROR: No off-the-shelf model data found in data.json")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating unified 3-subplot figure...")
    fig = create_unified_plot(models_data)
    for ext in ['pdf', 'png']:
        dpi = 300 if ext == 'pdf' else 150
        output_file = OUTPUT_DIR / f'offtheshelf_unified.{ext}'
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved {output_file}")
    plt.close(fig)

    print(f"\n✓ Done. Output: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
