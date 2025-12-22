#!/usr/bin/env python3
"""
Create cosine similarity distribution histograms for top-1 nearest neighbors.

Shows the distribution of cosine similarities between vision tokens and their
nearest text embedding neighbors, for each model combination.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent.resolve()
CONTEXTUAL_NN_DIR = SCRIPT_DIR.parent / 'analysis_results/contextual_nearest_neighbors'
OUTPUT_DIR = SCRIPT_DIR / 'paper_figures_output/similarity_histograms'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = [
    ("olmo-7b", "vit-l-14-336"),
    ("olmo-7b", "siglip"),
    ("olmo-7b", "dinov2-large-336"),
    ("llama3-8b", "vit-l-14-336"),
    ("llama3-8b", "siglip"),
    ("llama3-8b", "dinov2-large-336"),
    ("qwen2-7b", "vit-l-14-336"),
    ("qwen2-7b", "siglip"),
    ("qwen2-7b", "dinov2-large-336"),
]

LLM_DISPLAY = {
    'olmo-7b': 'OLMo-7B',
    'llama3-8b': 'Llama3-8B',
    'qwen2-7b': 'Qwen2-7B'
}

ENCODER_DISPLAY = {
    'vit-l-14-336': 'CLIP ViT-L/14',
    'siglip': 'SigLIP',
    'dinov2-large-336': 'DINOv2'
}

def find_model_dir(llm, encoder):
    """Find the model directory for contextual nearest neighbor results."""
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        pattern = f"train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_seed10_step12000-unsharded"
    else:
        pattern = f"train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_step12000-unsharded"
    
    matches = list(CONTEXTUAL_NN_DIR.glob(pattern))
    matches = [m for m in matches if 'lite' not in str(m)]
    return matches[0] if matches else None


def load_similarities(model_dir, visual_layer=0):
    """
    Load top-1 NN cosine similarities from contextual NN JSON file.
    Returns list of similarity values.
    
    Args:
        model_dir: Path to model directory
        visual_layer: Which vision layer to load (0, 1, 2, 4, 8, 16, 24, 30, 31)
    """
    # Find the contextual neighbors JSON file for the specified visual layer
    json_file = model_dir / f"contextual_neighbors_visual{visual_layer}_allLayers.json"
    
    if not json_file.exists():
        print(f"  No visual{visual_layer} JSON found in {model_dir.name}")
        return []
    
    print(f"  Loading: {json_file.name}")
    
    similarities = []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract similarities from results
    for result in data.get('results', []):
        for chunk in result.get('chunks', []):
            for patch in chunk.get('patches', []):
                nns = patch.get('nearest_contextual_neighbors', [])
                if nns:
                    # Get top-1 similarity
                    top_sim = nns[0].get('similarity', 0)
                    similarities.append(top_sim)
    
    print(f"    Found {len(similarities):,} similarity values")
    return similarities


def create_single_histogram(similarities, llm, encoder, output_path=None, bins=50):
    """Create a single histogram for one model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    title = f"{LLM_DISPLAY.get(llm, llm)} + {ENCODER_DISPLAY.get(encoder, encoder)}"
    
    if similarities:
        ax.hist(similarities, bins=bins, edgecolor='black', linewidth=0.5, 
                color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Cosine Similarity (Top-1 NN)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    ax.set_title(f'Top-1 NN Similarity Distribution\n{title}', fontsize=20, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close(fig)
    return fig


def create_combined_3x3_histogram(all_similarities, output_path=None, bins=30, layer=0):
    """
    Create a combined 3x3 figure with histograms for all 9 model combinations.
    Rows: OLMo, Llama, Qwen
    Cols: CLIP, SigLIP, DINOv2
    """
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    enc_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25,
                           top=0.92, bottom=0.08, left=0.08, right=0.98)
    
    # Determine global x-axis range for consistency
    all_values = []
    for sims in all_similarities.values():
        all_values.extend(sims)
    if all_values:
        x_min = min(all_values) - 0.01
        x_max = max(all_values) + 0.01
    else:
        x_min, x_max = 0, 1
    
    for row_idx, llm in enumerate(llm_order):
        for col_idx, encoder in enumerate(enc_order):
            key = f"{llm}+{encoder}"
            similarities = all_similarities.get(key, [])
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            title = f"{LLM_DISPLAY.get(llm, llm)} + {ENCODER_DISPLAY.get(encoder, encoder)}"
            
            if similarities:
                ax.hist(similarities, bins=bins, edgecolor='black', linewidth=0.3, 
                        color='steelblue', alpha=0.8)
            
            ax.set_xlim(x_min, x_max)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3)
            
            # Labels only on edges
            if row_idx == 2:
                ax.set_xlabel('Cosine Similarity', fontsize=15)
            if col_idx == 0:
                ax.set_ylabel('Frequency', fontsize=15)
            
            ax.tick_params(axis='both', labelsize=12)
    
    fig.suptitle(f'Top-1 NN Cosine Similarity Distribution (Visual Layer {layer})', 
                 fontsize=20, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        pdf_path = Path(output_path).with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {pdf_path}")
    
    plt.close(fig)
    return fig


def main(visual_layer=0):
    print(f"\n{'='*60}")
    print(f"Creating Cosine Similarity Histograms (Visual Layer {visual_layer})")
    print(f"{'='*60}")
    
    all_similarities = {}
    
    for llm, encoder in MODELS:
        print(f"\n{llm} + {encoder}")
        print("-" * 40)
        
        model_dir = find_model_dir(llm, encoder)
        if not model_dir:
            print(f"  ERROR: Directory not found, skipping")
            continue
        
        similarities = load_similarities(model_dir, visual_layer=visual_layer)
        key = f"{llm}+{encoder}"
        all_similarities[key] = similarities
        
        # Create individual plot
        output_path = OUTPUT_DIR / f"similarity_hist_{llm}_{encoder}_visual{visual_layer}.png"
        create_single_histogram(similarities, llm, encoder, output_path)
    
    # Create combined 3x3 figure
    print(f"\n{'='*60}")
    print("Creating combined 3x3 figure...")
    print(f"{'='*60}")
    
    combined_path = OUTPUT_DIR / f"similarity_hist_combined_3x3_visual{visual_layer}.png"
    create_combined_3x3_histogram(all_similarities, combined_path, layer=visual_layer)
    
    print(f"\n✓ All similarity histograms complete!")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    visual_layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(visual_layer=visual_layer)

