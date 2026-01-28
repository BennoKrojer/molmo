#!/usr/bin/env python3
"""
Create a line plot visualization of LLM judge interpretability results for contextual NN.
X-axis: Contextual layer number, Y-axis: % of interpretable patches
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

def load_results(results_dir, nn_results_dir=None):
    """Load all results JSON files and extract interpretability percentages per contextual layer.
    Also loads layer 0 from nearest neighbors results if nn_results_dir is provided."""
    results_dir = Path(results_dir)
    
    # Dictionary to store: {(llm, vision_encoder): {layer: interpretability_percentage}}
    data = defaultdict(lambda: defaultdict(dict))
    
    # Find all results JSON files
    for results_file in results_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            spresults = json.load(f)
        
        # Extract model name and layer from the path or from the JSON
        # Path format: llm_judge_{llm}_{vision_encoder}_contextual{layer}_{suffix}/results_validation.json
        # or: llm_judge_{llm}_{vision_encoder}_seed{seed}_contextual{layer}_{suffix}/results_validation.json
        path_str = str(results_file)
        
        # Extract layer from path (contextual NN uses contextual{N} format)
        layer_str = None
        match = re.search(r'_contextual(\d+)_', path_str)
        if match:
            layer_str = f"contextual{match.group(1)}"
        
        # Extract model info from JSON
        model_str = spresults.get('model', '')
        llm = spresults.get('llm', '')
        vision_encoder = spresults.get('vision_encoder', '')
        
        if not model_str:
            # Fallback: extract from path
            match = re.search(r'llm_judge_([^_]+)_([^_]+?)(?:_(?:seed\d+_)?contextual\d+|_contextual\d+)', path_str)
            if match:
                llm = llm or match.group(1)
                # Try to extract vision encoder from path
                path_parts = path_str.split('/')
                for part in path_parts:
                    if 'llm_judge_' in part:
                        model_part = part
                        # Extract encoder from model part
                        if 'vit-l-14-336' in model_part or 'vit-l' in model_part:
                            vision_encoder = vision_encoder or 'vit-l-14-336'
                        elif 'dinov2-large-336' in model_part or 'dinov2' in model_part:
                            vision_encoder = vision_encoder or 'dinov2-large-336'
                        elif 'siglip' in model_part:
                            vision_encoder = vision_encoder or 'siglip'
                        break
        
        # Parse model string to get LLM and vision encoder if not already extracted
        if not llm or not vision_encoder:
            if model_str:
                model_parts = model_str.split('_')
                if len(model_parts) >= 2:
                    # Handle seed10 case for qwen2-7b_vit-l-14-336
                    if 'seed' in model_parts[1]:
                        llm = llm or model_parts[0]
                        vision_encoder = vision_encoder or ('_'.join(model_parts[2:]) if len(model_parts) > 2 else model_parts[1])
                    else:
                        llm = llm or model_parts[0]
                        # Vision encoder might have multiple parts (e.g., vit-l-14-336)
                        if len(model_parts) == 2:
                            vision_encoder = vision_encoder or model_parts[1]
                        else:
                            # Try to match known encoders
                            if 'vit-l-14-336' in model_str or 'vit-l' in model_str:
                                vision_encoder = vision_encoder or 'vit-l-14-336'
                            elif 'dinov2-large-336' in model_str or 'dinov2' in model_str:
                                vision_encoder = vision_encoder or 'dinov2-large-336'
                            elif 'siglip' in model_str:
                                vision_encoder = vision_encoder or 'siglip'
                            else:
                                vision_encoder = vision_encoder or '_'.join(model_parts[1:])
        
        # Extract layer number
        if layer_str and layer_str.startswith('contextual'):
            layer_num = int(layer_str.replace('contextual', ''))
        else:
            continue
        
        # Only process if we have both llm and vision_encoder
        if not llm or not vision_encoder:
            continue
        
        # Calculate interpretability percentage from results
        # For contextual NN, we count interpretable patches
        results = spresults.get('results', [])
        if results:
            total = len(results)
            interpretable_count = sum(1 for r in results if r.get('interpretable', False))
            accuracy = (interpretable_count / total * 100.0) if total > 0 else 0.0
        else:
            # Fallback: try to get accuracy from JSON
            accuracy = spresults.get('accuracy', 0.0)
            # If accuracy is a fraction, convert to percentage
            if accuracy <= 1.0:
                accuracy = accuracy * 100.0
        
        data[(llm, vision_encoder)][layer_num] = accuracy
    
    # Also load layer 0 from nearest neighbors results if nn_results_dir is provided
    if nn_results_dir:
        nn_results_dir = Path(nn_results_dir)
        if nn_results_dir.exists():
            print(f"\nSearching for layer 0 results in {nn_results_dir}...")
            layer0_count = 0
            files_checked = 0
            # Find layer 0 results from NN directory
            for results_file in nn_results_dir.glob("**/results_*.json"):
                path_str = str(results_file)
                files_checked += 1
                
                # Extract layer from path - only process layer 0
                match = re.search(r'_layer(\d+)_', path_str)
                if not match:
                    continue
                layer_num = int(match.group(1))
                if layer_num != 0:
                    continue
                
                # Extract model info from path
                # Pattern: llm_judge_{llm}_{vision_encoder}(_seed10)?_layer0
                # Handle multi-part vision encoders like "vit-l-14-336"
                match = re.search(r'llm_judge_([^/]+?)_layer\d+', path_str)
                if not match:
                    print(f"  DEBUG: Could not match pattern in: {results_file.parent.name}")
                    continue
                
                full_model_part = match.group(1)
                print(f"  DEBUG: Found layer 0 file, model part: {full_model_part}")
                
                # Parse LLM and vision encoder from the full model part
                # Handle patterns like: olmo-7b_vit-l-14-336, qwen2-7b_vit-l-14-336_seed10, llama3-8b_dinov2-large-336
                parts = full_model_part.split('_')
                
                # First part is always the LLM
                llm = parts[0]
                
                # Extract vision encoder (everything after LLM, before seed if present)
                remaining = '_'.join(parts[1:])
                # Remove seed part if present
                remaining = re.sub(r'_seed\d+$', '', remaining)
                
                # Try to match known encoders
                if 'vit-l-14-336' in remaining or remaining.startswith('vit-l'):
                    vision_encoder = 'vit-l-14-336'
                elif 'dinov2-large-336' in remaining or remaining.startswith('dinov2'):
                    vision_encoder = 'dinov2-large-336'
                elif 'siglip' in remaining:
                    vision_encoder = 'siglip'
                else:
                    vision_encoder = remaining
                
                print(f"  DEBUG: Parsed as {llm} + {vision_encoder}")
                
                # Load JSON and get accuracy
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Get accuracy/interpretability percentage
                    accuracy = results.get('accuracy', 0.0)
                    
                    if llm and vision_encoder:
                        # Store as percentage (multiply by 100 if it's a fraction)
                        if accuracy <= 1.0:
                            accuracy = accuracy * 100.0
                        # Only add if not already present (prefer contextual results if they exist)
                        if 0 not in data.get((llm, vision_encoder), {}):
                            data[(llm, vision_encoder)][0] = accuracy
                            layer0_count += 1
                            print(f"  ✓ Loaded layer 0 for {llm} + {vision_encoder}: {accuracy:.1f}%")
                        else:
                            print(f"  - Skipped layer 0 for {llm} + {vision_encoder} (already exists)")
                except Exception as e:
                    print(f"  ERROR loading {results_file}: {e}")
            
            print(f"\nChecked {files_checked} JSON files")
            if layer0_count == 0:
                print(f"  Warning: No layer 0 results loaded from {nn_results_dir}")
            else:
                print(f"  Successfully loaded {layer0_count} layer 0 results from nearest neighbors")
    
    return data


def create_lineplot(data, output_path, title='Contextual NN Interpretability Across Layers (MLLM Judge)', 
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
            # Include all layers from layer_data (including layer 0 if present)
            layers = sorted(layer_data.keys())
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
    plt.xlabel('Contextual Layer', fontsize=12, fontweight='bold')
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
            print(f"{'C' + str(layer):>8}", end="")
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
    
    parser = argparse.ArgumentParser(description='Create line plot from LLM judge contextual NN results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='analysis_results/llm_judge_contextual_nn',
        help='Directory containing LLM judge contextual NN results'
    )
    parser.add_argument(
        '--nn-results-dir',
        type=str,
        default='analysis_results/llm_judge_nearest_neighbors',
        help='Directory containing LLM judge nearest neighbors results (for layer 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_results/llm_judge_contextual_nn/lineplot_interpretability.pdf',
        help='Output path for line plot (PDF or PNG)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading contextual results from: {args.results_dir}")
    if args.nn_results_dir:
        print(f"Loading layer 0 from nearest neighbors results: {args.nn_results_dir}")
    data = load_results(args.results_dir, nn_results_dir=args.nn_results_dir)
    
    if not data:
        print("ERROR: No results found!")
        return
    
    # Print summary
    print("\nFound results for:")
    for (llm, encoder), layer_data in sorted(data.items()):
        layers = sorted(layer_data.keys())
        if 0 in layers:
            print(f"  {llm} + {encoder}: {len(layers)} layers (0, {', '.join(str(l) for l in layers if l != 0)})")
        else:
            print(f"  {llm} + {encoder}: {len(layers)} contextual layers ({min(layers)}-{max(layers)})")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create interpretability line plot
    print("\n" + "="*60)
    print("Creating interpretability line plot...")
    print("="*60)
    create_lineplot(data, output_path, 
                   title='Contextual NN Interpretability Across Layers (MLLM Judge)',
                   ylabel='Interpretability %')
    
    print("\n" + "="*60)
    print("Line plot created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

