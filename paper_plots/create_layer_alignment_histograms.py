#!/usr/bin/env python3
"""
Create histograms showing which LLM embedding layers vision tokens align with.

For each vision layer, shows distribution of top-5 NN's source LLM layers.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
REPO_ROOT = Path(__file__).parent.parent
CONTEXTUAL_DIR = REPO_ROOT / "analysis_results" / "contextual_nearest_neighbors"
OUTPUT_DIR = Path(__file__).parent / "paper_figures_output" / "layer_alignment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configs
MODELS = [
    ("llama3-8b", "dinov2-large-336"),
    ("llama3-8b", "siglip"),
    ("llama3-8b", "vit-l-14-336"),
    ("olmo-7b", "dinov2-large-336"),
    ("olmo-7b", "siglip"),
    ("olmo-7b", "vit-l-14-336"),
    ("qwen2-7b", "dinov2-large-336"),
    ("qwen2-7b", "siglip"),
    ("qwen2-7b", "vit-l-14-336"),
]

# Display names
LLM_DISPLAY = {'llama3-8b': 'Llama3-8B', 'olmo-7b': 'OLMo-7B', 'qwen2-7b': 'Qwen2-7B'}
ENC_DISPLAY = {'vit-l-14-336': 'CLIP ViT-L/14', 'siglip': 'SigLIP', 'dinov2-large-336': 'DINOv2'}

# Layer configurations per LLM (Qwen has 28 layers, others have 32)
VISION_LAYERS_DEFAULT = [0, 1, 2, 4, 8, 16, 24, 30, 31]
VISION_LAYERS_QWEN = [0, 1, 2, 4, 8, 16, 24, 26, 27]
LLM_LAYERS_DEFAULT = [1, 2, 4, 8, 16, 24, 30, 31]
LLM_LAYERS_QWEN = [1, 2, 4, 8, 16, 24, 26, 27]

def get_vision_layers(llm):
    return VISION_LAYERS_QWEN if 'qwen' in llm else VISION_LAYERS_DEFAULT

def get_llm_layers(llm):
    return LLM_LAYERS_QWEN if 'qwen' in llm else LLM_LAYERS_DEFAULT


def find_model_dir(llm, encoder):
    """Find the model directory."""
    # Handle qwen2 seed10 case
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        pattern = f"*{llm}_{encoder}_seed10_step12000-unsharded"
    else:
        pattern = f"*{llm}_{encoder}_step12000-unsharded"
    
    matches = list(CONTEXTUAL_DIR.glob(pattern))
    # Filter out lite versions
    matches = [m for m in matches if 'lite' not in str(m)]
    
    if matches:
        return matches[0]
    return None


def load_layer_counts(model_dir, llm):
    """
    Load data and count which LLM layers the NNs come from for each vision layer.
    Returns: {vision_layer: {llm_layer: count}}
    """
    vision_layers = get_vision_layers(llm)
    counts = {vl: defaultdict(int) for vl in vision_layers}
    
    for vl in vision_layers:
        json_file = model_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
        if not json_file.exists():
            print(f"  Warning: {json_file.name} not found")
            continue
        
        print(f"  Loading visual{vl}...", end=" ", flush=True)
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Count NN source layers across all patches
        total_nns = 0
        for result in data.get('results', []):
            for chunk in result.get('chunks', []):
                for patch in chunk.get('patches', []):
                    for nn in patch.get('nearest_contextual_neighbors', []):
                        layer = nn.get('contextual_layer')
                        if layer is not None:
                            counts[vl][layer] += 1
                            total_nns += 1
        
        print(f"{total_nns} NNs")
    
    return counts


def create_histogram_figure(counts, llm, encoder, output_path):
    """Create figure with stacked horizontal histograms."""
    
    title = f"{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}"
    
    # Get model-specific layers
    vision_layers = get_vision_layers(llm)
    llm_layers = get_llm_layers(llm)
    
    # Create figure with subplots (one row per vision layer)
    n_rows = len(vision_layers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.5 * n_rows), sharex=True)
    
    # Color map for bars
    cmap = plt.cm.viridis
    colors = [cmap(i / len(llm_layers)) for i in range(len(llm_layers))]
    
    for idx, vl in enumerate(vision_layers):
        ax = axes[idx]
        
        # Get counts for this vision layer
        layer_counts = counts.get(vl, {})
        
        # Normalize to sum to 1
        total = sum(layer_counts.values())
        if total > 0:
            normalized = [layer_counts.get(ll, 0) / total for ll in llm_layers]
        else:
            normalized = [0] * len(llm_layers)
        
        # Create bar positions
        x = np.arange(len(llm_layers))
        
        # Plot bars
        bars = ax.bar(x, normalized, color=colors, edgecolor='black', linewidth=0.5)
        
        # Label on the right
        ax.set_ylabel(f"V{vl}", fontsize=10, rotation=0, labelpad=20, va='center')
        
        # Remove y-ticks but keep label
        ax.set_yticks([])
        ax.set_ylim(0, 1.0)  # Fixed y-axis for comparability across layers
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Only show x-ticks on bottom plot
        if idx == n_rows - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{ll}" for ll in llm_layers], fontsize=9)
            ax.set_xlabel("LLM Embedding Layer", fontsize=11, fontweight='bold')
    
    # Title
    fig.suptitle(f"Vision Token → LLM Layer Alignment\n{title}", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Left label
    fig.text(0.02, 0.5, "Vision Layer", va='center', rotation='vertical', 
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, hspace=0.1)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def create_combined_3x3_figure(all_counts, output_path):
    """
    Create a combined 3x3 figure with all 9 model combinations.
    Each cell looks like the individual figures (9 rows of bar charts).
    Layout: 3 model rows (LLMs) x 3 cols (encoders), each with 9 vision layer subplots.
    """
    import matplotlib.gridspec as gridspec
    
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    enc_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    n_vision_layers = 9
    
    # Create figure with nested gridspec for proper spacing
    fig = plt.figure(figsize=(16, 28))
    
    # Outer grid: 3 rows (LLMs) x 3 cols (encoders), with space between model blocks
    outer_grid = gridspec.GridSpec(3, 3, figure=fig, 
                                    hspace=0.15,  # space between LLM blocks
                                    wspace=0.12,  # space between encoder cols
                                    top=0.96, bottom=0.02, left=0.06, right=0.98)
    
    for model_row, llm in enumerate(llm_order):
        vision_layers = get_vision_layers(llm)
        llm_layers = get_llm_layers(llm)
        cmap = plt.cm.viridis
        colors = [cmap(i / len(llm_layers)) for i in range(len(llm_layers))]
        
        for col_idx, encoder in enumerate(enc_order):
            key = f"{llm}+{encoder}"
            counts = all_counts.get(key, {})
            
            # Inner grid for this model: 9 rows for vision layers
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                n_vision_layers, 1, 
                subplot_spec=outer_grid[model_row, col_idx],
                hspace=0.0  # no space between vision layer rows
            )
            
            for v_idx, vl in enumerate(vision_layers):
                ax = fig.add_subplot(inner_grid[v_idx])
                
                layer_counts = counts.get(vl, {})
                total = sum(layer_counts.values())
                if total > 0:
                    normalized = [layer_counts.get(ll, 0) / total for ll in llm_layers]
                else:
                    normalized = [0] * len(llm_layers)
                
                x = np.arange(len(llm_layers))
                ax.bar(x, normalized, color=colors, edgecolor='black', linewidth=0.3)
                ax.set_ylim(0, 1.0)
                ax.set_xlim(-0.5, len(llm_layers) - 0.5)
                ax.set_yticks([])
                ax.grid(axis='y', alpha=0.3, linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Vision layer label on left (only for leftmost column)
                if col_idx == 0:
                    ax.set_ylabel(f"V{vl}", fontsize=7, rotation=0, labelpad=10, va='center')
                else:
                    ax.set_ylabel('')
                
                # X-ticks only on bottom row of each model block
                if v_idx == n_vision_layers - 1:
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{ll}" for ll in llm_layers], fontsize=6)
                else:
                    ax.set_xticks([])
                
                # Title only on first row of each model block
                if v_idx == 0:
                    title = f"{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}"
                    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
                
                # Thin spines
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
    
    fig.suptitle('Vision Token → LLM Layer Alignment', 
                 fontsize=14, fontweight='bold')
    
    # Save PNG and PDF
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path.name}")
    
    plt.close()


def load_from_data_json():
    """Load pre-computed counts from data.json (fast!)."""
    data_json_path = Path(__file__).parent / "data.json"
    if not data_json_path.exists():
        return None
    
    with open(data_json_path) as f:
        data = json.load(f)
    
    if 'layer_alignment' not in data:
        return None
    
    # Convert string keys to int
    all_counts = {}
    for model_key, counts in data['layer_alignment'].items():
        all_counts[model_key] = {
            int(vl): {int(ll): c for ll, c in lc.items()}
            for vl, lc in counts.items()
        }
    return all_counts


def main():
    print("Creating layer alignment histograms...")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Try loading from data.json first (fast!)
    all_counts = load_from_data_json()
    if all_counts:
        print("✓ Loaded from data.json (fast mode)\n")
    else:
        print("Loading from raw files (slow mode)...\n")
        all_counts = {}
        for llm, encoder in MODELS:
            print(f"\n{llm} + {encoder}")
            print("-" * 40)
            
            model_dir = find_model_dir(llm, encoder)
            if not model_dir:
                print(f"  ERROR: Model directory not found")
                continue
            
            print(f"  Dir: {model_dir.name}")
            counts = load_layer_counts(model_dir, llm)
            all_counts[f"{llm}+{encoder}"] = counts
    
    # Create individual figures
    for model_key, counts in all_counts.items():
        llm, encoder = model_key.split('+')
        output_path = OUTPUT_DIR / f"layer_alignment_{llm}_{encoder}.png"
        create_histogram_figure(counts, llm, encoder, output_path)
    
    # Create combined 3x3 figure
    print("\n" + "=" * 50)
    print("Creating combined 3x3 figure...")
    combined_path = OUTPUT_DIR / "layer_alignment_combined_3x3.png"
    create_combined_3x3_figure(all_counts, combined_path)
    
    print(f"\n✓ All figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

