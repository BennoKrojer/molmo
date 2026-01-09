#!/usr/bin/env python3
"""
Token Similarity Across Layers - Line plots.

Shows how vision tokens and text tokens evolve through LLM layers by plotting
cosine similarity to layer 0 (input layer) across all layers.

For each model combination (3x3 grid):
- X-axis: LLM Layer
- Y-axis: Cosine Similarity to Layer 0
- Two lines: Vision tokens and Text tokens
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


def load_token_similarity_data():
    """
    Load token similarity data from data.json.
    
    Returns:
        Dictionary: {(llm, encoder): {'vision': {layer: sim}, 'text': {layer: sim}}}
    """
    with open(DATA_JSON) as f:
        data = json.load(f)
    
    token_sim = data.get('token_similarity', {})
    vision_data = token_sim.get('vision', {})
    text_data = token_sim.get('text', {})
    
    # Convert to (llm, encoder) tuple keys and int layer keys
    all_data = {}
    
    for key in set(vision_data.keys()) | set(text_data.keys()):
        llm, encoder = key.split('+')
        
        vision = {int(l): v for l, v in vision_data.get(key, {}).items()}
        text = {int(l): v for l, v in text_data.get(key, {}).items()}
        
        if vision or text:
            all_data[(llm, encoder)] = {'vision': vision, 'text': text}
    
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
            
            # Extract layers and similarities (already in correct format from data.json)
            vision_layers = sorted(vision_data.keys())
            vision_similarities = [vision_data[l] for l in vision_layers]
            
            text_layers = sorted(text_data.keys())
            text_similarities = [text_data[l] for l in text_layers]
            
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


def create_qwen2vl_plot(vision_data, text_data, output_path):
    """Create a single token similarity plot for Qwen2-VL."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract layers and similarities
    vision_layers = sorted(vision_data.keys())
    vision_similarities = [vision_data[l] for l in vision_layers]

    # For Qwen2-VL, we only have vision tokens (off-the-shelf model, no text similarity data)
    ax.plot(vision_layers, vision_similarities,
           marker='o', linewidth=2.5, markersize=10,
           label='Vision tokens', color='#2E86AB')

    # If we have text data, plot it too
    if text_data:
        text_layers = sorted(text_data.keys())
        text_similarities = [text_data[l] for l in text_layers]
        ax.plot(text_layers, text_similarities,
               marker='s', linewidth=2.5, markersize=10,
               label='Text tokens', color='#A23B72')
        ax.legend(fontsize=12, framealpha=0.95)

    # Title and labels
    ax.set_title('Qwen2-VL-7B-Instruct', fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('LLM Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cosine Similarity to Layer 0', fontsize=14, fontweight='bold')

    # Styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=12)

    all_layers = sorted(set(vision_layers))
    if all_layers:
        ax.set_xlim(min(all_layers) - 1, max(all_layers) + 1)
        ax.set_xticks(all_layers)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def main():
    print("Creating token similarity plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    # Load data from data.json
    all_data = load_token_similarity_data()

    if not all_data:
        print("ERROR: No token_similarity data found in data.json!")
        print("  Run: python update_data.py")
        return

    print(f"✓ Loaded data for {len(all_data)} model combinations\n")

    # Print summary
    print("Available model combinations:")
    for (llm, encoder) in sorted(all_data.keys()):
        vision_layers = len(all_data[(llm, encoder)]['vision'])
        text_layers = len(all_data[(llm, encoder)]['text'])
        print(f"  {LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(encoder, encoder)}: "
              f"Vision layers {vision_layers}, Text layers {text_layers}")
    print()

    # Create combined 3x3 plot
    print("=" * 50)
    print("Creating combined 3x3 plot...")
    combined_path = OUTPUT_DIR / "token_similarity_combined_3x3.png"
    create_combined_3x3_plot(all_data, combined_path)

    # Create Qwen2-VL plot if data exists
    print()
    print("=" * 50)
    print("Creating Qwen2-VL token similarity plot...")
    with open(DATA_JSON) as f:
        data = json.load(f)
    qwen2vl_token_sim = data.get('qwen2vl_token_similarity', {})
    if qwen2vl_token_sim:
        qwen2vl_vision = {int(k): v for k, v in qwen2vl_token_sim.get('vision', {}).items()}
        qwen2vl_text = {int(k): v for k, v in qwen2vl_token_sim.get('text', {}).items()}
        if qwen2vl_vision:
            qwen2vl_output_dir = SCRIPT_DIR / "paper_figures_output" / "qwen2vl"
            qwen2vl_output_dir.mkdir(parents=True, exist_ok=True)
            qwen2vl_path = qwen2vl_output_dir / "qwen2vl_token_similarity.png"
            create_qwen2vl_plot(qwen2vl_vision, qwen2vl_text, qwen2vl_path)
    else:
        print("  No qwen2vl_token_similarity data found in data.json")

    print(f"\n✓ All plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

