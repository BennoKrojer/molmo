#!/usr/bin/env python3
"""
Regenerate the L2 norm analysis plots with border lines for the paper appendix.
Adds border lines to make the 3x3 grid structure clearer.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import os

# ============================================================================
# Plot 1: Max Token Embedding Values (3x3 grid)
# ============================================================================

def create_max_token_embedding_plot():
    """Create the max token embedding distribution plot with borders."""
    output_dir = Path("analysis_results/max_token_embeddings")
    plot_dir = Path("paper_plots/paper_figures_output/l2norm_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

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

    LLM_DISPLAY = {'olmo-7b': 'OLMo-7B', 'llama3-8b': 'Llama3-8B', 'qwen2-7b': 'Qwen2-7B'}
    ENC_DISPLAY = {'vit-l-14-336': 'CLIP', 'dinov2-large-336': 'DINOv2', 'siglip': 'SigLIP'}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for idx, (llm, enc, suffix) in enumerate(models):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        ckpt = f"train_mlp-only_pixmo_cap_resize_{llm}_{enc}{suffix}_step12000-unsharded"
        json_path = output_dir / f"{ckpt}_max_token.json"

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)

            values = np.array(data['max_embedding_values'])
            l2_norm = data['max_info']['l2_norm']
            layer = data['max_info']['layer']
            stats = data['embedding_stats']

            ax.hist(values, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            # Mark percentiles
            ax.axvline(x=stats['percentiles']['p1'], color='red', linestyle=':', alpha=0.7, label='p1/p99')
            ax.axvline(x=stats['percentiles']['p99'], color='red', linestyle=':', alpha=0.7)

            ax.set_title(f"{LLM_DISPLAY[llm]} + {ENC_DISPLAY[enc]}\nL2={l2_norm:.0f}, layer={layer}",
                        fontsize=11, fontweight='bold')
            ax.text(0.95, 0.95, f"mean={stats['mean']:.2f}\nstd={stats['std']:.2f}\nmax={stats['max']:.2f}",
                    transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{LLM_DISPLAY[llm]} + {ENC_DISPLAY[enc]}", fontsize=11, fontweight='bold')

        ax.set_xlabel('Embedding Value', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Max L2 Norm Vision Token: Distribution of Embedding Dimensions\n(Is high L2 from few large values or all values?)',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add border lines around each subplot (3x3 grid)
    fig.canvas.draw()
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            bbox = ax.get_position()
            x0 = bbox.x0 - 0.012
            y0 = bbox.y0 - 0.015
            width = bbox.width + 0.024
            height = bbox.height + 0.045
            rect = Rectangle((x0, y0), width, height,
                            fill=False, edgecolor='black', linewidth=1.5,
                            transform=fig.transFigure, clip_on=False)
            fig.patches.append(rect)

    plt.savefig(plot_dir / 'max_token_embedding_values_3x3_bordered.png', dpi=200, bbox_inches='tight')
    plt.savefig(plot_dir / 'max_token_embedding_values_3x3_bordered.pdf', dpi=200, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'max_token_embedding_values_3x3_bordered.pdf'}")
    plt.close()


# ============================================================================
# Plot 2: L2 Norm 3x3 Grid (Vision vs Text side-by-side)
# ============================================================================

def load_l2norm_raw_data(base_path, ckpt_name, modality):
    """Load raw L2 norm values from JSON file."""
    if modality == 'vision':
        raw_file = base_path / ckpt_name / "l2norm_raw_values.json"
    else:
        raw_file = base_path / ckpt_name / "text_l2norm_raw_values.json"

    if not raw_file.exists():
        return None

    with open(raw_file) as f:
        data = json.load(f)
    return {int(k): np.array(v) for k, v in data['raw_norms'].items()}


def create_l2norm_3x3_grid():
    """Create the L2 norm 3x3 grid with borders."""
    plot_dir = Path("paper_plots/paper_figures_output/l2norm_plots")
    vision_base = Path("analysis_results/sameToken_acrossLayers_l2norm")
    text_base = Path("analysis_results/sameToken_acrossLayers_text_l2norm")

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

    LLM_DISPLAY = {'olmo-7b': 'OLMo-7B', 'llama3-8b': 'Llama3-8B', 'qwen2-7b': 'Qwen2-7B'}
    ENC_DISPLAY = {'vit-l-14-336': 'CLIP', 'dinov2-large-336': 'DINOv2', 'siglip': 'SigLIP'}

    # Load data from JSON files
    vision_data = {}
    text_data = {}

    for llm, enc, suffix in models:
        key = f"{llm}+{enc}"
        ckpt = f"train_mlp-only_pixmo_cap_resize_{llm}_{enc}{suffix}_step12000-unsharded"

        vd = load_l2norm_raw_data(vision_base, ckpt, 'vision')
        td = load_l2norm_raw_data(text_base, ckpt, 'text')

        if vd:
            vision_data[key] = vd
        if td:
            text_data[key] = td

    if not vision_data:
        print("No L2 norm data found! Skipping 3x3 grid plot.")
        return

    print(f"  Loaded vision data: {len(vision_data)} models")
    print(f"  Loaded text data: {len(text_data)} models")

    # Create 3x3 grid with histograms for each layer
    # Use log scale as in the original notebook
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # Color map for layers
    cmap = plt.cm.viridis

    for idx, (llm, enc, suffix) in enumerate(models):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        key = f"{llm}+{enc}"

        if key in vision_data:
            vd = vision_data[key]
            td = text_data.get(key, {})

            layers = sorted(vd.keys())

            # Compute stats per layer
            vision_means = []
            vision_stds = []
            vision_p99 = []
            vision_maxes = []
            text_means = []
            text_stds = []
            text_p99 = []
            text_maxes = []

            for layer in layers:
                v_norms = vd[layer]
                vision_means.append(np.mean(v_norms))
                vision_stds.append(np.std(v_norms))
                vision_p99.append(np.percentile(v_norms, 99))
                vision_maxes.append(np.max(v_norms))

                if layer in td:
                    t_norms = td[layer]
                    text_means.append(np.mean(t_norms))
                    text_stds.append(np.std(t_norms))
                    text_p99.append(np.percentile(t_norms, 99))
                    text_maxes.append(np.max(t_norms))
                else:
                    text_means.append(np.nan)
                    text_stds.append(np.nan)
                    text_p99.append(np.nan)
                    text_maxes.append(np.nan)

            # Plot vision
            ax.plot(layers, vision_means, 'b-', linewidth=2, label='Vision (mean)', marker='o', markersize=5)
            ax.fill_between(layers,
                          np.array(vision_means) - np.array(vision_stds),
                          np.array(vision_means) + np.array(vision_stds),
                          alpha=0.2, color='blue')
            ax.plot(layers, vision_p99, 'b--', linewidth=1, alpha=0.7, label='Vision (p99)')
            ax.plot(layers, vision_maxes, 'b:', linewidth=1, alpha=0.5, label='Vision (max)')

            # Plot text
            valid_mask = ~np.isnan(text_means)
            if np.any(valid_mask):
                valid_layers = np.array(layers)[valid_mask]
                ax.plot(valid_layers, np.array(text_means)[valid_mask], 'r-', linewidth=2, label='Text (mean)', marker='s', markersize=5)
                ax.fill_between(valid_layers,
                              np.array(text_means)[valid_mask] - np.array(text_stds)[valid_mask],
                              np.array(text_means)[valid_mask] + np.array(text_stds)[valid_mask],
                              alpha=0.2, color='red')
                ax.plot(valid_layers, np.array(text_p99)[valid_mask], 'r--', linewidth=1, alpha=0.7, label='Text (p99)')
                ax.plot(valid_layers, np.array(text_maxes)[valid_mask], 'r:', linewidth=1, alpha=0.5, label='Text (max)')
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f"{LLM_DISPLAY[llm]} + {ENC_DISPLAY[enc]}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('L2 Norm', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    fig.suptitle('L2 Norm Distribution: Vision vs Text Tokens Across LLM Layers\n(mean ± std, p99, max)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add border lines around each subplot
    fig.canvas.draw()
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            bbox = ax.get_position()
            x0 = bbox.x0 - 0.012
            y0 = bbox.y0 - 0.015
            width = bbox.width + 0.024
            height = bbox.height + 0.045
            rect = Rectangle((x0, y0), width, height,
                            fill=False, edgecolor='black', linewidth=1.5,
                            transform=fig.transFigure, clip_on=False)
            fig.patches.append(rect)

    plt.savefig(plot_dir / 'l2norm_3x3_grid_log_bordered.png', dpi=200, bbox_inches='tight')
    plt.savefig(plot_dir / 'l2norm_3x3_grid_log_bordered.pdf', dpi=200, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'l2norm_3x3_grid_log_bordered.pdf'}")
    plt.close()


if __name__ == "__main__":
    print("Regenerating plots with border lines...")
    print()

    # Plot 1: Max token embedding values
    print("Creating max token embedding plot...")
    create_max_token_embedding_plot()
    print()

    # Plot 2: L2 norm 3x3 grid
    print("Creating L2 norm 3x3 grid plot...")
    create_l2norm_3x3_grid()
    print()

    print("Done!")
