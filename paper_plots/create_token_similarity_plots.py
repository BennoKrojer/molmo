#!/usr/bin/env python3
"""
Token Similarity Across Layers - Line plots.

Shows how vision tokens and text tokens evolve through LLM layers by plotting
cosine similarity to layer 0 (input layer) across all layers.

For each model combination (3x3 grid):
- X-axis: LLM Layer
- Y-axis: Cosine Similarity to Layer 0
- Two lines: Vision tokens (from vision similarity analysis) and Text tokens (from text similarity analysis)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Paths
SCRIPT_DIR = Path(__file__).parent
ANALYSIS_RESULTS_DIR = SCRIPT_DIR.parent / "analysis_results"
VISION_SIM_DIR = ANALYSIS_RESULTS_DIR / "sameToken_acrossLayers_similarity"
TEXT_SIM_DIR = ANALYSIS_RESULTS_DIR / "sameToken_acrossLayers_text_similarity"
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "token_similarity_plots"

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


def extract_model_from_checkpoint(checkpoint_path):
    """
    Extract LLM and encoder from checkpoint path.
    
    Example: 'train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded'
    Returns: ('olmo-7b', 'vit-l-14-336')
    """
    # Pattern: ..._llm_encoder_...
    llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    llm = None
    encoder = None
    
    for l in llms:
        if l in checkpoint_path:
            llm = l
            break
    
    for e in encoders:
        if e in checkpoint_path:
            encoder = e
            break
    
    return llm, encoder


def load_similarity_data(sim_dir, checkpoint_name):
    """
    Load similarity summary JSON for a given checkpoint.
    
    Args:
        sim_dir: Directory containing similarity results
        checkpoint_name: Name of checkpoint directory
    
    Returns:
        Dictionary with layer -> similarity data, or None if not found
    """
    # Determine filename based on directory name
    if 'text_similarity' in sim_dir.name:
        filename = "text_similarity_across_layers_summary.json"
    else:
        filename = "similarity_across_layers_summary.json"
    
    summary_file = sim_dir / checkpoint_name / filename
    
    if not summary_file.exists():
        return None
    
    with open(summary_file) as f:
        data = json.load(f)
    
    return data.get('global_averages', {})


def load_all_similarity_data():
    """
    Load all vision and text similarity data, organized by model combination.
    
    Returns:
        Dictionary: {(llm, encoder): {'vision': {...}, 'text': {...}}}
    """
    all_data = {}
    
    # Find all checkpoint directories
    vision_checkpoints = set()
    text_checkpoints = set()
    
    if VISION_SIM_DIR.exists():
        vision_checkpoints = {d.name for d in VISION_SIM_DIR.iterdir() if d.is_dir()}
    
    if TEXT_SIM_DIR.exists():
        text_checkpoints = {d.name for d in TEXT_SIM_DIR.iterdir() if d.is_dir()}
    
    # Find common checkpoints
    common_checkpoints = vision_checkpoints & text_checkpoints
    
    print(f"Found {len(common_checkpoints)} common checkpoints")
    
    for checkpoint_name in common_checkpoints:
        llm, encoder = extract_model_from_checkpoint(checkpoint_name)
        
        if llm is None or encoder is None:
            print(f"Warning: Could not extract model from {checkpoint_name}")
            continue
        
        key = (llm, encoder)
        
        # Load vision similarity data
        vision_data = load_similarity_data(VISION_SIM_DIR, checkpoint_name)
        
        # Load text similarity data
        text_data = load_similarity_data(TEXT_SIM_DIR, checkpoint_name)
        
        if vision_data is None or text_data is None:
            print(f"Warning: Missing data for {checkpoint_name}")
            continue
        
        all_data[key] = {
            'vision': vision_data,
            'text': text_data
        }
    
    return all_data


def create_combined_3x3_plot(all_data, output_path):
    """Create combined 3x3 line plot figure."""
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    enc_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    for row, llm in enumerate(llm_order):
        for col, encoder in enumerate(enc_order):
            ax = axes[row, col]
            key = (llm, encoder)
            
            if key not in all_data:
                ax.set_visible(False)
                continue
            
            vision_data = all_data[key]['vision']
            text_data = all_data[key]['text']
            
            # Extract layers and similarities
            # Vision tokens
            vision_layers = sorted([int(l) for l in vision_data.keys()])
            vision_similarities = [vision_data[str(l)]['same_token']['mean_similarity'] for l in vision_layers]
            
            # Text tokens
            text_layers = sorted([int(l) for l in text_data.keys()])
            text_similarities = [text_data[str(l)]['same_token']['mean_similarity'] for l in text_layers]
            
            # Plot lines
            ax.plot(vision_layers, vision_similarities, 
                   marker='o', linewidth=2.5, markersize=8, 
                   label='Vision tokens', color='#2E86AB', alpha=0.8)
            ax.plot(text_layers, text_similarities,
                   marker='s', linewidth=2.5, markersize=8,
                   label='Text tokens', color='#A23B72', alpha=0.8)
            
            # Title
            title = f'{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
            
            # Labels (only on edges)
            if col == 0:
                ax.set_ylabel('Cosine Similarity\nto Layer 0', fontsize=15)
            if row == 2:
                ax.set_xlabel('LLM Layer', fontsize=15)
            
            # Styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=12)
            
            # Set x-axis limits based on available layers
            all_layers = sorted(set(vision_layers + text_layers))
            if all_layers:
                ax.set_xlim(min(all_layers) - 1, max(all_layers) + 1)
                # Show subset of x-ticks if too many
                if len(all_layers) > 15:
                    step = max(1, len(all_layers) // 8)
                    shown_layers = all_layers[::step]
                    if all_layers[-1] not in shown_layers:
                        shown_layers.append(all_layers[-1])
                    ax.set_xticks(shown_layers)
                else:
                    ax.set_xticks(all_layers)
    
    # Get handles and labels from first subplot for global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Adjust spacing to accommodate legend at bottom
    fig.subplots_adjust(right=0.98, hspace=0.35, wspace=0.25, top=0.88, bottom=0.12)
    
    # Get the position of the middle column subplot to center legend on it
    # Middle column is column index 1, use middle row (row 1) for reference
    middle_subplot = axes[1, 1]
    bbox = middle_subplot.get_position()
    middle_col_center_x = bbox.x0 + bbox.width / 2
    
    # Create global legend centered on middle column
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(middle_col_center_x, 0.02),
              ncol=2, fontsize=17, framealpha=0.9, columnspacing=2.0, handlelength=2.0)
    
    fig.suptitle('Same-Token Similarity Across LLM Layers', fontsize=20, fontweight='bold', y=0.95)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def main():
    print("Creating token similarity plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Load data
    all_data = load_all_similarity_data()
    
    if not all_data:
        print("ERROR: No similarity data found!")
        print(f"  Vision dir: {VISION_SIM_DIR}")
        print(f"  Text dir: {TEXT_SIM_DIR}")
        return
    
    print(f"✓ Loaded data for {len(all_data)} model combinations\n")
    
    # Print summary
    print("Available model combinations:")
    for (llm, encoder) in sorted(all_data.keys()):
        vision_layers = sorted([int(l) for l in all_data[(llm, encoder)]['vision'].keys()])
        text_layers = sorted([int(l) for l in all_data[(llm, encoder)]['text'].keys()])
        print(f"  {LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}: "
              f"Vision layers {len(vision_layers)}, Text layers {len(text_layers)}")
    print()
    
    # Create combined 3x3 plot
    print("=" * 50)
    print("Creating combined 3x3 plot...")
    combined_path = OUTPUT_DIR / "token_similarity_combined_3x3.png"
    create_combined_3x3_plot(all_data, combined_path)
    
    print(f"\n✓ All plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

