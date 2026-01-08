#!/usr/bin/env python3
"""
L2 Norm Histogram Plots.

Shows the distribution of L2 norms for vision tokens and text tokens
across different LLM layers for each model combination.

For each model combination (3x3 grid):
- X-axis: L2 Norm
- Y-axis: Density
- Multiple overlaid histograms for different layers (0, 4, 8, 24)
- Vision tokens (solid) vs Text tokens (dashed)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_JSON = SCRIPT_DIR / "data.json"
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "l2norm_plots"

# Display names
LLM_DISPLAY = {
    'olmo-7b': 'OLMo-7B',
    'llama3-8b': 'Llama3-8B',
    'qwen2-7b': 'Qwen2-7B'
}
ENC_DISPLAY = {
    'vit-l-14-336': 'CLIP',
    'siglip': 'SigLIP',
    'dinov2-large-336': 'DINOv2'
}

# Layer colors
LAYER_COLORS = {
    0: '#1f77b4',   # Blue
    4: '#ff7f0e',   # Orange
    8: '#2ca02c',   # Green
    24: '#d62728', # Red
}


def load_l2norm_data():
    """
    Load L2 norm data from data.json.

    Returns:
        Dictionary: {'vision': {model_key: {layer: histogram_data}},
                     'text': {model_key: {layer: histogram_data}}}
    """
    with open(DATA_JSON) as f:
        data = json.load(f)

    l2norm = data.get('l2norm', {})

    # Convert string layer keys to integers
    for modality in ['vision', 'text']:
        if modality in l2norm:
            for model_key in l2norm[modality]:
                l2norm[modality][model_key] = {
                    int(k): v for k, v in l2norm[modality][model_key].items()
                }

    return l2norm


def plot_histogram_from_bins(ax, counts, bin_edges, color, linestyle='-', label=None, alpha=0.3):
    """Plot a histogram from pre-computed counts and bin_edges."""
    # Normalize to density
    total = sum(counts)
    bin_widths = np.diff(bin_edges)
    density = np.array(counts) / (total * bin_widths)

    # Use step plot for cleaner visualization
    bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2

    # Plot as filled step
    ax.fill_between(bin_centers, 0, density, alpha=alpha, color=color, step='mid')
    ax.step(bin_centers, density, where='mid', color=color, linestyle=linestyle,
            linewidth=1.5, label=label)


def create_combined_3x3_plot(l2norm_data, output_path, target_layers=[0, 4, 8, 24]):
    """Create combined 3x3 histogram plot figure."""
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    enc_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    vision_data = l2norm_data.get('vision', {})
    text_data = l2norm_data.get('text', {})

    for row, llm in enumerate(llm_order):
        for col, encoder in enumerate(enc_order):
            ax = axes[row, col]
            key = f"{llm}+{encoder}"

            has_vision = key in vision_data
            has_text = key in text_data

            if not has_vision and not has_text:
                ax.set_visible(False)
                continue

            # Plot histograms for each layer
            for layer in target_layers:
                color = LAYER_COLORS.get(layer, '#333333')

                # Vision (solid line)
                if has_vision and layer in vision_data[key]:
                    hist = vision_data[key][layer]
                    plot_histogram_from_bins(
                        ax, hist['counts'], hist['bin_edges'],
                        color=color, linestyle='-',
                        label=f'Layer {layer} (Vision)' if row == 0 and col == 0 else None,
                        alpha=0.2
                    )

                # Text (dashed line)
                if has_text and layer in text_data[key]:
                    hist = text_data[key][layer]
                    plot_histogram_from_bins(
                        ax, hist['counts'], hist['bin_edges'],
                        color=color, linestyle='--',
                        label=f'Layer {layer} (Text)' if row == 0 and col == 0 else None,
                        alpha=0.1
                    )

            # Title
            title = f'{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

            # Labels (only on edges)
            if col == 0:
                ax.set_ylabel('Density', fontsize=13)
            if row == 2:
                ax.set_xlabel('L2 Norm', fontsize=13)

            # Styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=11)

    # Create legend with layer colors and vision/text distinction
    # Custom legend handles
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = []
    for layer in target_layers:
        color = LAYER_COLORS.get(layer, '#333333')
        legend_elements.append(
            Patch(facecolor=color, alpha=0.4, label=f'Layer {layer}')
        )
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='Vision'))
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Text'))

    # Adjust spacing to accommodate legend at bottom
    fig.subplots_adjust(right=0.98, hspace=0.35, wspace=0.25, top=0.88, bottom=0.15)

    # Create global legend
    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=12,
              framealpha=0.9, columnspacing=1.5)

    fig.suptitle('L2 Norm Distribution Across LLM Layers', fontsize=18, fontweight='bold', y=0.95)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def create_mean_lineplot(l2norm_data, output_path, target_layers=[0, 4, 8, 24]):
    """Create a line plot showing mean L2 norm across layers (similar to token similarity plot)."""
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    enc_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    vision_data = l2norm_data.get('vision', {})
    text_data = l2norm_data.get('text', {})

    for row, llm in enumerate(llm_order):
        for col, encoder in enumerate(enc_order):
            ax = axes[row, col]
            key = f"{llm}+{encoder}"

            has_vision = key in vision_data
            has_text = key in text_data

            if not has_vision and not has_text:
                ax.set_visible(False)
                continue

            # Extract mean L2 norms for each layer
            vision_layers = []
            vision_means = []
            text_layers = []
            text_means = []

            for layer in sorted(target_layers):
                if has_vision and layer in vision_data[key]:
                    vision_layers.append(layer)
                    vision_means.append(vision_data[key][layer]['mean'])

                if has_text and layer in text_data[key]:
                    text_layers.append(layer)
                    text_means.append(text_data[key][layer]['mean'])

            # Plot lines
            if vision_layers:
                ax.plot(vision_layers, vision_means,
                       marker='o', linewidth=2.5, markersize=8,
                       label='Vision tokens', color='#2E86AB', alpha=0.8)
            if text_layers:
                ax.plot(text_layers, text_means,
                       marker='s', linewidth=2.5, markersize=8,
                       label='Text tokens', color='#A23B72', alpha=0.8)

            # Title
            title = f'{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

            # Labels (only on edges)
            if col == 0:
                ax.set_ylabel('Mean L2 Norm', fontsize=13)
            if row == 2:
                ax.set_xlabel('LLM Layer', fontsize=13)

            # Styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=11)

            # Set x-axis to show target layers
            ax.set_xticks(target_layers)

    # Get handles and labels from first subplot for global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Adjust spacing
    fig.subplots_adjust(right=0.98, hspace=0.35, wspace=0.25, top=0.88, bottom=0.12)

    # Get center of middle column
    middle_subplot = axes[1, 1]
    bbox = middle_subplot.get_position()
    middle_col_center_x = bbox.x0 + bbox.width / 2

    # Create global legend
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(middle_col_center_x, 0.02),
              ncol=2, fontsize=15, framealpha=0.9, columnspacing=2.0, handlelength=2.0)

    fig.suptitle('Mean L2 Norm Across LLM Layers', fontsize=18, fontweight='bold', y=0.95)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def main():
    print("Creating L2 norm plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    # Load data from data.json
    l2norm_data = load_l2norm_data()

    if not l2norm_data.get('vision') and not l2norm_data.get('text'):
        print("ERROR: No l2norm data found in data.json!")
        print("  Run the analysis first:")
        print("    ./run_parallel_sameToken_l2norm.sh")
        print("  Then update data.json:")
        print("    python update_data.py")
        return

    # Print summary
    print("Available data:")
    print(f"  Vision models: {len(l2norm_data.get('vision', {}))}")
    print(f"  Text models: {len(l2norm_data.get('text', {}))}")

    for key in sorted(l2norm_data.get('vision', {}).keys()):
        layers = sorted(l2norm_data['vision'][key].keys())
        print(f"    {key}: layers {layers}")
    print()

    # Create combined 3x3 histogram plot
    print("Creating histogram plots...")
    histogram_path = OUTPUT_DIR / "l2norm_histogram_combined_3x3.png"
    create_combined_3x3_plot(l2norm_data, histogram_path)

    # Create mean line plot (similar to token similarity)
    print("Creating mean line plot...")
    lineplot_path = OUTPUT_DIR / "l2norm_mean_combined_3x3.png"
    create_mean_lineplot(l2norm_data, lineplot_path)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
