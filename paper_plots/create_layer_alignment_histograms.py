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
        ax.set_ylim(0, max(normalized) * 1.1 if max(normalized) > 0 else 1)
        
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


def main():
    print("Creating layer alignment histograms...")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    for llm, encoder in MODELS:
        print(f"\n{llm} + {encoder}")
        print("-" * 40)
        
        model_dir = find_model_dir(llm, encoder)
        if not model_dir:
            print(f"  ERROR: Model directory not found")
            continue
        
        print(f"  Dir: {model_dir.name}")
        
        # Load counts
        counts = load_layer_counts(model_dir, llm)
        
        # Create figure
        output_path = OUTPUT_DIR / f"layer_alignment_{llm}_{encoder}.png"
        create_histogram_figure(counts, llm, encoder, output_path)
    
    print(f"\n✓ All figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

