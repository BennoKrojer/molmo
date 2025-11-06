#!/usr/bin/env python3
"""
Create a line plot visualization of LLM judge interpretability results for logit lens.
X-axis: Layer number, Y-axis: % of interpretable tokens (at least 1 out of 5 marked interpretable)
Lines: 9 different model combinations (3 LLMs × 3 vision encoders)
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def load_results(results_dir):
    """Load all results JSON files and extract interpretability percentages per layer."""
    results_dir = Path(results_dir)
    
    # Dictionary to store: {(llm, vision_encoder): {layer: interpretability_percentage}}
    data = defaultdict(lambda: defaultdict(dict))
    
    # Find all results JSON files
    for results_file in results_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            spresults = json.load(f)
        
        # Extract model name and layer from the path or from the JSON
        # Path format: llm_judge_{llm}_{vision_encoder}_{layer}_{suffix}/results_validation.json
        # or: llm_judge_{llm}_{vision_encoder}_seed{seed}_{layer}_{suffix}/results_validation.json
        path_str = str(results_file)
        
        # Extract layer from JSON first (more reliable)
        layer_str = spresults.get('layer', '')
        if not layer_str:
            # Fallback: extract from path
            match = re.search(r'_layer(\d+)_', path_str)
            if match:
                layer_str = f"layer{match.group(1)}"
        
        # Extract model info from JSON
        model_str = spresults.get('model', '')
        if not model_str:
            # Fallback: extract from path
            match = re.search(r'llm_judge_([^_]+)_([^_]+?)(?:_(?:seed\d+_)?layer\d+|_layer\d+)', path_str)
            if match:
                model_str = f"{match.group(1)}_{match.group(2)}"
        
        # Parse model string to get LLM and vision encoder
        # Format: {llm}_{vision_encoder} or {llm}_{vision_encoder}_seed{seed}
        model_parts = model_str.split('_')
        if len(model_parts) >= 2:
            # Handle seed10 case for qwen2-7b_vit-l-14-336
            if 'seed' in model_parts[1]:
                # Skip seed part
                llm = model_parts[0]
                vision_encoder = '_'.join(model_parts[2:]) if len(model_parts) > 2 else model_parts[1]
            else:
                llm = model_parts[0]
                # Vision encoder might have multiple parts (e.g., vit-l-14-336)
                # Check if it's a known encoder name
                if len(model_parts) == 2:
                    vision_encoder = model_parts[1]
                else:
                    # Try to match known encoders
                    if 'vit-l-14-336' in model_str or 'vit-l' in model_str:
                        vision_encoder = 'vit-l-14-336'
                    elif 'dinov2-large-336' in model_str or 'dinov2' in model_str:
                        vision_encoder = 'dinov2-large-336'
                    elif 'siglip' in model_str:
                        vision_encoder = 'siglip'
                    else:
                        vision_encoder = '_'.join(model_parts[1:])
        else:
            # Skip if we can't parse
            continue
        
        # Extract layer number
        if layer_str.startswith('layer'):
            layer_num = int(layer_str.replace('layer', ''))
        else:
            continue
        
        # Get accuracy/interpretability percentage
        accuracy = spresults.get('accuracy', 0.0)
        
        if llm and vision_encoder:
            # Store as percentage (multiply by 100 if it's a fraction)
            if accuracy <= 1.0:
                accuracy = accuracy * 100.0
            data[(llm, vision_encoder)][layer_num] = accuracy
    
    return data


def create_lineplot(data, output_path, title='Logit Lens Interpretability Across Layers (MLLM Judge)', 
                     ylabel='Interpretability %', print_table=True):
    """Create and save a line plot visualization."""
    if not data:
        print("No data found to visualize")
        return
    
    # Mapping from internal names to display names
    llm_display_names = {
        'llama3-8b': 'Llama3-8B',
        'olmo-7b': 'Olmo-7B',
        'qwen2-7b': 'Qwen2-7B'
    }
    
    encoder_display_names = {
        'vit-l-14-336': 'CLIP ViT-L/14',
        'siglip': 'SigLIP',
        'dinov2-large-336': 'DinoV2'
    }
    
    # Define exact order for LLMs and encoders
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoder_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    # Get all layers from all model combinations
    all_layers = set()
    for (llm, encoder), layer_data in data.items():
        all_layers.update(layer_data.keys())
    all_layers = sorted(list(all_layers))
    
    # Create figure with good sizing
    plt.figure(figsize=(14, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Generate colors for 9 lines
    # Use a colormap that provides distinct colors
    # Use tab20 for more distinct colors across 9 lines
    # Create color scheme where each LLM gets a base color, and encoders get different shades
    # Define base colors for each LLM
    llm_base_colors = {
        'olmo-7b': plt.cm.Blues,
        'llama3-8b': plt.cm.Greens,
        'qwen2-7b': plt.cm.Reds
    }
    
    # For each LLM, we'll use 3 shades (one per encoder)
    # Use indices 0.4, 0.6, 0.8 to get light to dark shades
    encoder_shade_indices = [0.5, 0.7, 0.9]
    
    # Define markers for each vision encoder (distinct shapes for clarity)
    # Matplotlib has many marker options: 'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h', 'H', '*', 'X', 'P', '+', 'x', '|', '_', etc.
    encoder_markers = {
        'vit-l-14-336': '*',       # star
        'siglip': 'o',             # circle (will be hollow)
        'dinov2-large-336': '^'    # triangle
    }
    
    # Define marker face colors (for hollow markers)
    encoder_marker_facecolors = {
        'vit-l-14-336': None,      # filled (use line color)
        'siglip': 'none',          # hollow/transparent
        'dinov2-large-336': None   # filled (use line color)
    }
    
    # Create color mapping for each combination
    color_map = {}
    for llm in llm_order:
        base_cmap = llm_base_colors[llm]
        for enc_idx, encoder in enumerate(encoder_order):
            color_map[(llm, encoder)] = base_cmap(encoder_shade_indices[enc_idx])
    
    # Plot lines for each model combination
    plotted_combinations = []
    
    for llm in llm_order:
        for encoder in encoder_order:
            key = (llm, encoder)
            if key not in data:
                continue
            
            layer_data = data[key]
            
            # Extract layers and values in sorted order
            layers = sorted([l for l in layer_data.keys() if l in all_layers])
            values = [layer_data[l] for l in layers]
            
            if len(layers) == 0:
                continue
            
            # Create label
            llm_label = llm_display_names.get(llm, llm)
            encoder_label = encoder_display_names.get(encoder, encoder)
            label = f"{llm_label} + {encoder_label}"
            
            # Get marker and marker properties for this encoder
            marker = encoder_markers.get(encoder, 'o')
            marker_facecolor = encoder_marker_facecolors.get(encoder, None)
            
            # Plot line with thinner lines
            if marker_facecolor is not None:
                plt.plot(layers, values, marker=marker, label=label, 
                        color=color_map[key], markerfacecolor=marker_facecolor,
                        markeredgewidth=1.5, linewidth=1.5, markersize=8)
            else:
                plt.plot(layers, values, marker=marker, label=label, 
                        color=color_map[key], linewidth=1.5, markersize=8)
            
            plotted_combinations.append((llm, encoder, label))
    
    # Customize plot
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    # Position legend outside the plot area if there are many layers
    legend_outside = len(all_layers) > 15
    if legend_outside:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    else:
        plt.legend(loc='best', fontsize=9, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all layers as integers
    if all_layers:
        plt.xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)
        plt.xticks(all_layers, rotation=0 if len(all_layers) <= 15 else 45, ha='right' if len(all_layers) > 15 else 'center')
    
    # Set y-axis range
    plt.ylim(0, 100)
    
    if legend_outside:
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Line plot saved to: {output_path}")
    
    # Also save as PNG with lower DPI for quick viewing
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Line plot also saved as PNG: {png_path}")
    
    plt.close()
    
    # Print the data table
    if print_table:
        print(f"\n{title}:")
        print("\n" + "="*80)
        
        # Print header
        print(f"{'Model Combination':<40}", end="")
        for layer in all_layers:
            print(f"{'L' + str(layer):>8}", end="")
        print()
        print("-" * (40 + len(all_layers) * 8))
        
        # Print data for each combination
        for llm, encoder, label in plotted_combinations:
            key = (llm, encoder)
            layer_data = data[key]
            print(f"{label:<40}", end="")
            for layer in all_layers:
                value = layer_data.get(layer, None)
                if value is not None:
                    print(f"{value:>7.1f}%", end="")
                else:
                    print(f"{'---':>8}", end="")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create line plot from LLM judge logit lens results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='analysis_results/llm_judge_logitlens',
        help='Directory containing LLM judge logit lens results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_results/llm_judge_logitlens/lineplot_interpretability.pdf',
        help='Output path for line plot (PDF or PNG)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    data = load_results(args.results_dir)
    
    if not data:
        print("ERROR: No results found!")
        return
    
    # Print summary
    print("\nFound results for:")
    for (llm, encoder), layer_data in sorted(data.items()):
        layers = sorted(layer_data.keys())
        print(f"  {llm} + {encoder}: {len(layers)} layers ({min(layers)}-{max(layers)})")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create interpretability line plot
    print("\n" + "="*60)
    print("Creating interpretability line plot...")
    print("="*60)
    create_lineplot(data, output_path, 
                   title='Logit Lens Interpretability Across Layers (MLLM Judge)',
                   ylabel='Interpretability %')
    
    print("\n" + "="*60)
    print("Line plot created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

