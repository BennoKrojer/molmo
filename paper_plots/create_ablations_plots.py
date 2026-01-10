#!/usr/bin/env python3
"""
Create ablation comparison plots.

Generates plots comparing all ablations against the baseline olmo-7b + vit-l-14-336.
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_JSON = SCRIPT_DIR / "data.json"
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "ablations"

# Ablation display names and groupings
BASELINE = "olmo-7b_vit-l-14-336"

ABLATION_GROUPS = {
    'caption_style': {
        'title': 'Caption Style',
        'models': ['first-sentence_olmo-7b_vit-l-14-336'],
        'labels': ['First-Sentence Only']
    },
    'vit_layers': {
        'title': 'ViT Layer',
        'models': [
            'olmo-7b_vit-l-14-336_earlier-vit-layers-6',
            'olmo-7b_vit-l-14-336_earlier-vit-layers-10'
        ],
        'labels': ['Layer 6', 'Layer 10']
    },
    'connector': {
        'title': 'Connector Type',
        'models': ['olmo-7b_vit-l-14-336_linear'],
        'labels': ['Linear']
    },
    'seeds': {
        'title': 'Random Seeds',
        'models': [
            'olmo-7b_vit-l-14-336_seed10',
            'olmo-7b_vit-l-14-336_seed11'
        ],
        'labels': ['Seed 10', 'Seed 11']
    },
    'llm_frozen': {
        'title': 'LLM Frozen vs Unfrozen',
        'models': ['olmo-7b_vit-l-14-336_unfreeze'],
        'labels': ['Unfrozen LLM']
    },
    'task': {
        'title': 'Training Task',
        'models': [
            'train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336',
            'train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm'
        ],
        'labels': ['TopBottom (frozen)', 'TopBottom (unfrozen)']
    }
}

# All ablations for mega plot (excluding baseline)
ALL_ABLATIONS = {
    'first-sentence_olmo-7b_vit-l-14-336': 'First-Sentence',
    'olmo-7b_vit-l-14-336_earlier-vit-layers-6': 'ViT Layer 6',
    'olmo-7b_vit-l-14-336_earlier-vit-layers-10': 'ViT Layer 10',
    'olmo-7b_vit-l-14-336_linear': 'Linear',
    'olmo-7b_vit-l-14-336_seed10': 'Seed 10',
    'olmo-7b_vit-l-14-336_seed11': 'Seed 11',
    'olmo-7b_vit-l-14-336_unfreeze': 'Unfrozen LLM',
    'olmo-7b_vit-l-14-336_unfreeze-llm': 'Unfrozen LLM (alt)',
    'train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336': 'TopBottom (frozen)',
    'train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm': 'TopBottom (unfrozen)'
}


def load_data():
    """Load data from data.json."""
    with open(DATA_JSON, 'r') as f:
        return json.load(f)


def create_mega_plot(ablations_data, method_name):
    """Create mega plot with all ablations vs baseline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Plot baseline (bold black line)
    if BASELINE in ablations_data:
        layers = sorted(ablations_data[BASELINE].keys())
        values = [ablations_data[BASELINE][l] for l in layers]
        ax.plot(layers, values, 'k-', linewidth=3.5, label='Baseline (OLMo-7B + ViT-L/14)', zorder=100)

    # Plot ablations (colored, thinner lines)
    colors = plt.cm.tab20(np.linspace(0, 1, len(ALL_ABLATIONS)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']

    for (model_key, label), color, marker in zip(ALL_ABLATIONS.items(), colors, markers):
        if model_key in ablations_data:
            layers = sorted(ablations_data[model_key].keys())
            values = [ablations_data[model_key][l] for l in layers]
            ax.plot(layers, values, marker=marker, color=color, linewidth=2,
                   markersize=8, label=label, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=16, fontweight='bold')
    ax.set_ylabel('Interpretability %', fontsize=16, fontweight='bold')
    ax.set_title(f'Ablations: {method_name}', fontsize=18, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=13)

    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.95)

    plt.tight_layout()
    return fig


def create_grouped_plots(ablations_data, method_name):
    """Create separate plots for each ablation group."""
    figs = {}

    for group_name, group_info in ABLATION_GROUPS.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot baseline
        if BASELINE in ablations_data:
            layers = sorted(ablations_data[BASELINE].keys())
            values = [ablations_data[BASELINE][l] for l in layers]
            ax.plot(layers, values, 'k-', linewidth=3, label='Baseline', zorder=100)

        # Plot group ablations
        colors = plt.cm.Set2(np.linspace(0, 1, len(group_info['models'])))
        markers = ['o', 's', '^', 'v', 'D']

        for model, label, color, marker in zip(group_info['models'], group_info['labels'],
                                               colors, markers):
            if model in ablations_data:
                layers = sorted(ablations_data[model].keys())
                values = [ablations_data[model][l] for l in layers]
                ax.plot(layers, values, marker=marker, color=color, linewidth=2.5,
                       markersize=9, label=label, alpha=0.9)

        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Interpretability %', fontsize=14, fontweight='bold')
        ax.set_title(f'{group_info["title"]}: {method_name}', fontsize=16, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12, framealpha=0.95)

        plt.tight_layout()
        figs[group_name] = fig

    return figs


def main():
    parser = argparse.ArgumentParser(description='Generate ablation plots')
    parser.add_argument('--mega-only', action='store_true', help='Only generate mega plot')
    parser.add_argument('--grouped-only', action='store_true', help='Only generate grouped plots')
    args = parser.parse_args()

    # Load data
    print("Loading data from data.json...")
    data = load_data()

    if 'ablations' not in data:
        print("ERROR: No ablations data in data.json")
        print("Run: python update_data.py")
        return

    ablations = data['ablations']
    print(f"Found ablations: {len(ablations.get('contextual', {}))} models")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def convert_keys_to_int(method_data):
        """Convert string keys to int for each model's layer data (JSON stores keys as strings)."""
        return {model: {int(k): v for k, v in layers.items()}
                for model, layers in method_data.items()}

    # Get baseline from MAIN model data (not ablations - ablations folder has corrupted baseline)
    # The olmo-7b+vit-l-14-336 from main models is the correct baseline
    baseline_key = 'olmo-7b+vit-l-14-336'
    baseline_data = {}
    for method in ['nn', 'contextual']:
        main_method_data = data.get(method, {})
        if baseline_key in main_method_data:
            baseline_data[method] = {int(k): v for k, v in main_method_data[baseline_key].items()}
            print(f"  Using main model baseline for {method}: {len(baseline_data[method])} layers")

    # Generate plots for each method
    methods = {
        'nn': ('Input Emb.', convert_keys_to_int(ablations.get('nn', {}))),
        'logitlens': ('LogitLens', convert_keys_to_int(ablations.get('logitlens', {}))),
        'contextual': ('LN-Lens', convert_keys_to_int(ablations.get('contextual', {})))
    }

    # Inject baseline into ablations data for plotting
    for method_key in methods:
        if method_key in baseline_data:
            methods[method_key][1][BASELINE] = baseline_data[method_key]

    for method_key, (method_name, method_data) in methods.items():
        if not method_data:
            print(f"  Skipping {method_name} (no data)")
            continue

        print(f"\nGenerating plots for {method_name}...")

        # Mega plot
        if not args.grouped_only:
            print("  Creating mega plot...")
            fig = create_mega_plot(method_data, method_name)
            for ext in ['pdf', 'png']:
                output_file = OUTPUT_DIR / f'ablations_mega_{method_key}.{ext}'
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    ✓ Saved {output_file}")
            plt.close(fig)

        # Grouped plots
        if not args.mega_only:
            print("  Creating grouped plots...")
            grouped_figs = create_grouped_plots(method_data, method_name)
            for group_name, fig in grouped_figs.items():
                for ext in ['pdf', 'png']:
                    output_file = OUTPUT_DIR / f'ablations_{group_name}_{method_key}.{ext}'
                    fig.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    ✓ Saved {group_name}")
                plt.close(fig)

    print(f"\n✓ All plots saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
