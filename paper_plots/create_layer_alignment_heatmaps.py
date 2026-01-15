#!/usr/bin/env python3
"""
Layer Alignment Heatmaps - Alternative visualization.

Shows vision layer (Y) vs LLM layer (X) alignment as heatmaps.
Color intensity = normalized proportion of top-5 NNs from that LLM layer.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_JSON = SCRIPT_DIR / "data.json"
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "layer_alignment_heatmaps"

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

# Fixed layer configurations per LLM (for consistent axes)
# Vision layers we sample at
VISION_LAYERS_DEFAULT = [0, 1, 2, 4, 8, 16, 24, 30, 31]  # OLMo, Llama
VISION_LAYERS_QWEN = [0, 1, 2, 4, 8, 16, 24, 26, 27]     # Qwen

# LLM embedding layers we compare against (layer 0 = Input Embedding Matrix)
LLM_LAYERS_DEFAULT = [0, 1, 2, 4, 8, 16, 24, 30, 31]  # OLMo, Llama (32 layers)
LLM_LAYERS_QWEN = [0, 1, 2, 4, 8, 16, 24, 26, 27]     # Qwen (28 layers)


def get_fixed_layers(llm):
    """Get fixed vision and LLM layers based on LLM type."""
    if 'qwen' in llm.lower():
        return VISION_LAYERS_QWEN, LLM_LAYERS_QWEN
    else:
        return VISION_LAYERS_DEFAULT, LLM_LAYERS_DEFAULT


def load_data():
    """Load layer alignment data from data.json."""
    with open(DATA_JSON) as f:
        data = json.load(f)
    return data.get('layer_alignment', {})


def create_single_heatmap(counts, llm, encoder, output_path):
    """Create a single heatmap for one model combination."""
    vision_layers, llm_layers = get_fixed_layers(llm)
    n_vision = len(vision_layers)
    n_llm = len(llm_layers)
    
    # Build the matrix: rows = vision layers (Y), cols = LLM layers (X)
    matrix = np.zeros((n_vision, n_llm))
    
    for i, vl in enumerate(vision_layers):
        vl_str = str(vl)
        layer_counts = counts.get(vl_str, {})
        total = sum(layer_counts.values())
        if total > 0:
            for j, ll in enumerate(llm_layers):
                ll_str = str(ll)
                matrix[i, j] = layer_counts.get(ll_str, 0) / total
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Heatmap with pcolormesh for cell borders
    # Fixed 0-1 color range for comparability
    im = ax.pcolormesh(matrix, cmap='viridis', vmin=0, vmax=1.0,
                       edgecolors='black', linewidth=0.5)
    ax.set_aspect('auto')
    
    # Labels
    ax.set_xlabel('LLM Layer', fontsize=18)
    ax.set_ylabel('Vision Layer', fontsize=18)
    ax.set_title(f'{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}',
                 fontsize=20, fontweight='bold', pad=15)
    
    # Y-axis: vision layers (ticks at cell centers: 0.5, 1.5, ...)
    ax.set_yticks([i + 0.5 for i in range(n_vision)])
    ax.set_yticklabels([str(vl) for vl in vision_layers], fontsize=15)
    
    # X-axis: LLM layers (ticks at cell centers)
    ax.set_xticks([i + 0.5 for i in range(n_llm)])
    ax.set_xticklabels([str(ll) for ll in llm_layers], fontsize=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Proportion of Top-5 NNs', fontsize=16)
    cbar.ax.tick_params(labelsize=15)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def create_combined_3x3_heatmap(all_counts, output_path):
    """Create combined 3x3 heatmap figure."""
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    enc_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    
    # Build matrices for all models
    all_matrices = {}
    
    for llm in llm_order:
        vision_layers, llm_layers = get_fixed_layers(llm)
        n_vision = len(vision_layers)
        n_llm = len(llm_layers)
        
        for encoder in enc_order:
            key = f"{llm}+{encoder}"
            counts = all_counts.get(key, {})
            if not counts:
                continue
            
            matrix = np.zeros((n_vision, n_llm))
            for i, vl in enumerate(vision_layers):
                vl_str = str(vl)
                layer_counts = counts.get(vl_str, {})
                total = sum(layer_counts.values())
                if total > 0:
                    for j, ll in enumerate(llm_layers):
                        ll_str = str(ll)
                        matrix[i, j] = layer_counts.get(ll_str, 0) / total
            
            all_matrices[key] = (matrix, vision_layers, llm_layers)
    
    images = []
    for row, llm in enumerate(llm_order):
        for col, encoder in enumerate(enc_order):
            ax = axes[row, col]
            key = f"{llm}+{encoder}"
            
            if key not in all_matrices:
                ax.set_visible(False)
                continue
            
            matrix, vision_layers, llm_layers = all_matrices[key]
            n_vision = len(vision_layers)
            n_llm = len(llm_layers)
            
            # Heatmap with pcolormesh for cell borders, fixed 0-1 color range
            im = ax.pcolormesh(matrix, cmap='viridis', vmin=0, vmax=1.0,
                              edgecolors='black', linewidth=0.3)
            ax.set_aspect('auto')
            images.append(im)
            
            # Title
            title = f'{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
            
            # Y-axis labels (ticks at cell centers, only leftmost column shows labels)
            ax.set_yticks([i + 0.5 for i in range(n_vision)])
            if col == 0:
                ax.set_yticklabels([str(vl) for vl in vision_layers], fontsize=12)
                ax.set_ylabel('Vision Layer', fontsize=15)
            else:
                ax.set_yticklabels([])
            
            # X-axis labels - show on ALL subplots since LLM layers differ per model
            ax.set_xticks([i + 0.5 for i in range(n_llm)])
            ax.set_xticklabels([str(ll) for ll in llm_layers], fontsize=12)
            if row == 2:
                ax.set_xlabel('LLM Layer', fontsize=15)
    
    # Single colorbar for all
    # Adjust spacing: more vertical space, lower top to accommodate suptitle, more bottom space for x-labels
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15, top=0.88, bottom=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label('Proportion of Top-5 NNs', fontsize=16)
    cbar.ax.tick_params(labelsize=15)
    
    fig.suptitle('Vision Token → LLM Layer Alignment', fontsize=20, fontweight='bold', y=0.95)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def create_qwen2vl_heatmap(counts, output_path):
    """Create a single heatmap for Qwen2-VL (off-the-shelf model)."""
    vision_layers = VISION_LAYERS_QWEN
    llm_layers = LLM_LAYERS_QWEN
    n_vision = len(vision_layers)
    n_llm = len(llm_layers)

    # Build the matrix: rows = vision layers (Y), cols = LLM layers (X)
    matrix = np.zeros((n_vision, n_llm))

    for i, vl in enumerate(vision_layers):
        # counts keys are ints for Qwen2-VL
        layer_counts = counts.get(vl, counts.get(str(vl), {}))
        total = sum(layer_counts.values())
        if total > 0:
            for j, ll in enumerate(llm_layers):
                matrix[i, j] = layer_counts.get(ll, layer_counts.get(str(ll), 0)) / total

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Heatmap with pcolormesh for cell borders
    # Fixed 0-1 color range for comparability
    im = ax.pcolormesh(matrix, cmap='viridis', vmin=0, vmax=1.0,
                       edgecolors='black', linewidth=0.5)
    ax.set_aspect('auto')

    # Labels
    ax.set_xlabel('LLM Layer', fontsize=18)
    ax.set_ylabel('Vision Layer', fontsize=18)
    ax.set_title('Qwen2-VL-7B-Instruct', fontsize=20, fontweight='bold', pad=15)

    # Y-axis: vision layers (ticks at cell centers: 0.5, 1.5, ...)
    ax.set_yticks([i + 0.5 for i in range(n_vision)])
    ax.set_yticklabels([str(vl) for vl in vision_layers], fontsize=15)

    # X-axis: LLM layers (ticks at cell centers)
    ax.set_xticks([i + 0.5 for i in range(n_llm)])
    ax.set_xticklabels([str(ll) for ll in llm_layers], fontsize=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Proportion of Top-5 NNs', fontsize=16)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def main():
    print("Creating layer alignment heatmaps...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Load data
    all_counts = load_data()
    if not all_counts:
        print("ERROR: No layer_alignment data in data.json")
        return
    print(f"✓ Loaded {len(all_counts)} model combinations\n")
    
    # Debug: show fixed layers per LLM
    print("Fixed layer configurations:")
    print(f"  OLMo/Llama: Vision={VISION_LAYERS_DEFAULT}, LLM={LLM_LAYERS_DEFAULT}")
    print(f"  Qwen:       Vision={VISION_LAYERS_QWEN}, LLM={LLM_LAYERS_QWEN}")
    print()
    
    # Create individual heatmaps
    llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    for llm in llms:
        for encoder in encoders:
            key = f"{llm}+{encoder}"
            counts = all_counts.get(key, {})
            if counts:
                output_path = OUTPUT_DIR / f"heatmap_{llm}_{encoder}.png"
                create_single_heatmap(counts, llm, encoder, output_path)
    
    # Create combined 3x3
    print("\n" + "=" * 50)
    print("Creating combined 3x3 heatmap...")
    combined_path = OUTPUT_DIR / "heatmap_combined_3x3.png"
    create_combined_3x3_heatmap(all_counts, combined_path)

    # Create Qwen2-VL heatmap (off-the-shelf model)
    print("\n" + "=" * 50)
    print("Creating Qwen2-VL heatmap...")
    with open(DATA_JSON) as f:
        data = json.load(f)
    qwen2vl_counts = data.get('qwen2vl_layer_alignment', {})
    if qwen2vl_counts:
        qwen2vl_output_dir = SCRIPT_DIR / "paper_figures_output" / "qwen2vl"
        qwen2vl_output_dir.mkdir(parents=True, exist_ok=True)
        qwen2vl_path = qwen2vl_output_dir / "qwen2vl_layer_alignment_heatmap.png"
        create_qwen2vl_heatmap(qwen2vl_counts, qwen2vl_path)
    else:
        print("  No qwen2vl_layer_alignment data found in data.json")

    print(f"\n✓ All heatmaps saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
