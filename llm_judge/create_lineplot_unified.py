#!/usr/bin/env python3
"""
Create a unified line plot visualization with 3 subplots side by side:
1. Nearest Neighbors Interpretability
2. Logit Lens Interpretability
3. Contextual NN Interpretability

Each subplot shows lines for 9 model combinations (3 LLMs × 3 vision encoders)
with a single shared legend at the bottom.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import sys


def load_nn_results(results_dir):
    """Load nearest neighbors results."""
    results_dir = Path(results_dir)
    data = defaultdict(lambda: defaultdict(dict))
    
    for results_file in results_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        path_str = str(results_file)
        
        # Extract layer from path
        match = re.search(r'_layer(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))
        
        # Extract model info from path
        match = re.search(r'llm_judge_([^_]+)_([^_]+)(?:_seed\d+)?_layer\d+', path_str)
        if not match:
            continue
        
        llm = match.group(1)
        vision_encoder = match.group(2)
        
        # Get accuracy
        accuracy = results.get('accuracy', 0.0)
        
        if llm and vision_encoder:
            if accuracy <= 1.0:
                accuracy = accuracy * 100.0
            data[(llm, vision_encoder)][layer_num] = accuracy
    
    return data


def load_logitlens_results(results_dir):
    """Load logit lens results."""
    results_dir = Path(results_dir)
    data = defaultdict(lambda: defaultdict(dict))
    
    for results_file in results_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            spresults = json.load(f)
        
        path_str = str(results_file)
        
        # Extract layer from JSON first
        layer_str = spresults.get('layer', '')
        if not layer_str:
            match = re.search(r'_layer(\d+)_', path_str)
            if match:
                layer_str = f"layer{match.group(1)}"
        
        # Extract model info from JSON
        model_str = spresults.get('model', '')
        if not model_str:
            match = re.search(r'llm_judge_([^_]+)_([^_]+?)(?:_(?:seed\d+_)?layer\d+|_layer\d+)', path_str)
            if match:
                model_str = f"{match.group(1)}_{match.group(2)}"
        
        # Parse model string
        model_parts = model_str.split('_')
        if len(model_parts) >= 2:
            if 'seed' in model_parts[1]:
                llm = model_parts[0]
                vision_encoder = '_'.join(model_parts[2:]) if len(model_parts) > 2 else model_parts[1]
            else:
                llm = model_parts[0]
                if len(model_parts) == 2:
                    vision_encoder = model_parts[1]
                else:
                    if 'vit-l-14-336' in model_str or 'vit-l' in model_str:
                        vision_encoder = 'vit-l-14-336'
                    elif 'dinov2-large-336' in model_str or 'dinov2' in model_str:
                        vision_encoder = 'dinov2-large-336'
                    elif 'siglip' in model_str:
                        vision_encoder = 'siglip'
                    else:
                        vision_encoder = '_'.join(model_parts[1:])
        else:
            continue
        
        # Extract layer number
        if layer_str.startswith('layer'):
            layer_num = int(layer_str.replace('layer', ''))
        else:
            continue
        
        # Get accuracy
        accuracy = spresults.get('accuracy', 0.0)
        
        if llm and vision_encoder:
            if accuracy <= 1.0:
                accuracy = accuracy * 100.0
            data[(llm, vision_encoder)][layer_num] = accuracy
    
    return data


def load_contextual_results(results_dir, nn_results_dir=None):
    """Load contextual NN results."""
    results_dir = Path(results_dir)
    data = defaultdict(lambda: defaultdict(dict))
    
    for results_file in results_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            spresults = json.load(f)
        
        path_str = str(results_file)
        
        # Extract layer from path
        layer_str = None
        match = re.search(r'_contextual(\d+)_', path_str)
        if match:
            layer_str = f"contextual{match.group(1)}"
        
        # Extract model info from JSON
        model_str = spresults.get('model', '')
        llm = spresults.get('llm', '')
        vision_encoder = spresults.get('vision_encoder', '')
        
        if not model_str:
            match = re.search(r'llm_judge_([^_]+)_([^_]+?)(?:_(?:seed\d+_)?contextual\d+|_contextual\d+)', path_str)
            if match:
                llm = llm or match.group(1)
                path_parts = path_str.split('/')
                for part in path_parts:
                    if 'llm_judge_' in part:
                        model_part = part
                        if 'vit-l-14-336' in model_part or 'vit-l' in model_part:
                            vision_encoder = vision_encoder or 'vit-l-14-336'
                        elif 'dinov2-large-336' in model_part or 'dinov2' in model_part:
                            vision_encoder = vision_encoder or 'dinov2-large-336'
                        elif 'siglip' in model_part:
                            vision_encoder = vision_encoder or 'siglip'
                        break
        
        # Parse model string if needed
        if not llm or not vision_encoder:
            if model_str:
                model_parts = model_str.split('_')
                if len(model_parts) >= 2:
                    if 'seed' in model_parts[1]:
                        llm = llm or model_parts[0]
                        vision_encoder = vision_encoder or ('_'.join(model_parts[2:]) if len(model_parts) > 2 else model_parts[1])
                    else:
                        llm = llm or model_parts[0]
                        if len(model_parts) == 2:
                            vision_encoder = vision_encoder or model_parts[1]
                        else:
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
        
        if not llm or not vision_encoder:
            continue
        
        # Calculate interpretability percentage
        results = spresults.get('results', [])
        if results:
            total = len(results)
            interpretable_count = sum(1 for r in results if r.get('interpretable', False))
            accuracy = (interpretable_count / total * 100.0) if total > 0 else 0.0
        else:
            accuracy = spresults.get('accuracy', 0.0)
            if accuracy <= 1.0:
                accuracy = accuracy * 100.0
        
        data[(llm, vision_encoder)][layer_num] = accuracy
    
    # Load layer 0 from nearest neighbors if provided
    if nn_results_dir:
        nn_results_dir = Path(nn_results_dir)
        if nn_results_dir.exists():
            for results_file in nn_results_dir.glob("**/results_*.json"):
                path_str = str(results_file)
                
                match = re.search(r'_layer(\d+)_', path_str)
                if not match:
                    continue
                layer_num = int(match.group(1))
                if layer_num != 0:
                    continue
                
                match = re.search(r'llm_judge_([^/]+?)_layer\d+', path_str)
                if not match:
                    continue
                
                full_model_part = match.group(1)
                parts = full_model_part.split('_')
                llm = parts[0]
                
                remaining = '_'.join(parts[1:])
                remaining = re.sub(r'_seed\d+$', '', remaining)
                
                if 'vit-l-14-336' in remaining or remaining.startswith('vit-l'):
                    vision_encoder = 'vit-l-14-336'
                elif 'dinov2-large-336' in remaining or remaining.startswith('dinov2'):
                    vision_encoder = 'dinov2-large-336'
                elif 'siglip' in remaining:
                    vision_encoder = 'siglip'
                else:
                    vision_encoder = remaining
                
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                accuracy = results.get('accuracy', 0.0)
                
                if llm and vision_encoder:
                    if accuracy <= 1.0:
                        accuracy = accuracy * 100.0
                    if 0 not in data.get((llm, vision_encoder), {}):
                        data[(llm, vision_encoder)][0] = accuracy
    
    return data


def create_unified_lineplot(nn_data, logitlens_data, contextual_data, output_path):
    """Create a unified figure with 3 subplots side by side and a shared legend."""
    
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
    
    # Define exact order
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoder_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    # Color scheme
    llm_base_colors = {
        'olmo-7b': plt.cm.Blues,
        'llama3-8b': plt.cm.Greens,
        'qwen2-7b': plt.cm.Reds
    }
    encoder_shade_indices = [0.5, 0.7, 0.9]
    
    # Markers
    encoder_markers = {
        'vit-l-14-336': '*',       # star
        'siglip': 'o',             # circle (hollow)
        'dinov2-large-336': '^'    # triangle
    }
    
    encoder_marker_facecolors = {
        'vit-l-14-336': None,
        'siglip': 'none',
        'dinov2-large-336': None
    }
    
    # Create color mapping
    color_map = {}
    for llm in llm_order:
        base_cmap = llm_base_colors[llm]
        for enc_idx, encoder in enumerate(encoder_order):
            color_map[(llm, encoder)] = base_cmap(encoder_shade_indices[enc_idx])
    
    # Create figure with 3 subplots side by side
    # Make it much wider to give plenty of room for x-axis labels
    fig, axes = plt.subplots(1, 3, figsize=(36, 8))
    sns.set_style("whitegrid")
    
    # Titles and data for each subplot
    subplot_configs = [
        {
            'ax': axes[0],
            'data': nn_data,
            'title': 'Nearest Neighbors',
            'xlabel': 'Layer'
        },
        {
            'ax': axes[1],
            'data': logitlens_data,
            'title': 'Logit Lens',
            'xlabel': 'Layer'
        },
        {
            'ax': axes[2],
            'data': contextual_data,
            'title': 'Contextual NN',
            'xlabel': 'Layer'
        }
    ]
    
    # Store handles and labels for shared legend
    handles_dict = {}
    
    for config in subplot_configs:
        ax = config['ax']
        data = config['data']
        
        if not data:
            continue
        
        # Get all layers
        all_layers = set()
        for (llm, encoder), layer_data in data.items():
            all_layers.update(layer_data.keys())
        all_layers = sorted(list(all_layers))
        
        # Plot lines for each model combination
        for llm in llm_order:
            for encoder in encoder_order:
                key = (llm, encoder)
                if key not in data:
                    continue
                
                layer_data = data[key]
                layers = sorted(layer_data.keys())
                values = [layer_data[l] for l in layers]
                
                if len(layers) == 0:
                    continue
                
                # Create label
                llm_label = llm_display_names.get(llm, llm)
                encoder_label = encoder_display_names.get(encoder, encoder)
                label = f"{llm_label} + {encoder_label}"
                
                # Get marker properties
                marker = encoder_markers.get(encoder, 'o')
                marker_facecolor = encoder_marker_facecolors.get(encoder, None)
                
                # Plot line
                if marker_facecolor is not None:
                    line, = ax.plot(layers, values, marker=marker, 
                                   color=color_map[key], markerfacecolor=marker_facecolor,
                                   markeredgewidth=2, linewidth=2.5, markersize=10)
                else:
                    line, = ax.plot(layers, values, marker=marker,
                                   color=color_map[key], linewidth=2.5, markersize=10)
                
                # Store handle for legend (only once)
                if label not in handles_dict:
                    handles_dict[label] = line
        
        # Customize subplot
        ax.set_xlabel(config['xlabel'], fontsize=16, fontweight='bold')
        ax.set_ylabel('Interpretability %', fontsize=16, fontweight='bold')
        ax.set_title(config['title'], fontsize=18, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Set x-axis
        if all_layers:
            ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)
            ax.set_xticks(all_layers)
            # Rotate labels if there are many layers to avoid overlap
            if len(all_layers) > 10:
                ax.tick_params(axis='both', labelsize=12)
                ax.set_xticklabels(all_layers, rotation=35, ha='right')
            elif len(all_layers) > 7:
                ax.tick_params(axis='both', labelsize=12)
                ax.set_xticklabels(all_layers, rotation=0)
            else:
                ax.tick_params(axis='both', labelsize=13)
    
    # Create single shared legend at the bottom
    # Get handles and labels in the desired order
    ordered_handles = []
    ordered_labels = []
    for llm in llm_order:
        for encoder in encoder_order:
            llm_label = llm_display_names.get(llm, llm)
            encoder_label = encoder_display_names.get(encoder, encoder)
            label = f"{llm_label} + {encoder_label}"
            if label in handles_dict:
                ordered_handles.append(handles_dict[label])
                ordered_labels.append(label)
    
    # Add legend well below the plots with larger font and more spacing
    fig.legend(ordered_handles, ordered_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.08),
              ncol=3, 
              fontsize=15, 
              framealpha=0.9,
              columnspacing=2.5,
              handlelength=2.5,
              handletextpad=1.2)
    
    # Adjust layout with proper spacing
    plt.tight_layout()
    # Add significant space at the bottom for the legend
    plt.subplots_adjust(bottom=0.20, wspace=0.28)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Unified line plot saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Unified line plot also saved as PNG: {png_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create unified line plot from all LLM judge results')
    parser.add_argument(
        '--nn-results-dir',
        type=str,
        default='analysis_results/llm_judge_nearest_neighbors',
        help='Directory containing nearest neighbors results'
    )
    parser.add_argument(
        '--logitlens-results-dir',
        type=str,
        default='analysis_results/llm_judge_logitlens',
        help='Directory containing logit lens results'
    )
    parser.add_argument(
        '--contextual-results-dir',
        type=str,
        default='analysis_results/llm_judge_contextual_nn',
        help='Directory containing contextual NN results'
    )
    parser.add_argument(
        '--nn-layer0-dir',
        type=str,
        default='analysis_results/llm_judge_nearest_neighbors',
        help='Directory containing layer 0 results for contextual NN'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_results/llm_judge/lineplot_unified.pdf',
        help='Output path for unified line plot'
    )
    
    args = parser.parse_args()
    
    # Load results from all three sources
    print("Loading nearest neighbors results...")
    nn_data = load_nn_results(args.nn_results_dir)
    print(f"  Found {len(nn_data)} model combinations")
    
    print("\nLoading logit lens results...")
    logitlens_data = load_logitlens_results(args.logitlens_results_dir)
    print(f"  Found {len(logitlens_data)} model combinations")
    
    print("\nLoading contextual NN results...")
    contextual_data = load_contextual_results(args.contextual_results_dir, args.nn_layer0_dir)
    print(f"  Found {len(contextual_data)} model combinations")
    
    if not nn_data and not logitlens_data and not contextual_data:
        print("ERROR: No results found!")
        return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create unified plot
    print("\n" + "="*60)
    print("Creating unified line plot...")
    print("="*60)
    create_unified_lineplot(nn_data, logitlens_data, contextual_data, output_path)
    
    print("\n" + "="*60)
    print("Unified line plot created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

