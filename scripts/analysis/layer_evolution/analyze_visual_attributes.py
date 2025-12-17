#!/usr/bin/env python3
"""
Analyze the occurrence of visual attribute words (colors, shapes, textures) in nearest neighbors across layers.

This script reads ALL nearest neighbor results (not just the sampled LLM judge results) and counts
how often concrete visual words appear, providing a simple proxy for visual grounding.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import argparse
from tqdm import tqdm
import pickle

from visual_attribute_words import VISUAL_ATTRIBUTES, get_attribute_type


def load_visual_attribute_counts(contextual_nn_dir, static_nn_dir, llm, vision_encoder):
    """
    Load all NN results and count occurrences of visual attribute words per layer.
    
    Returns:
        dict: {layer: {'color': count, 'shape': count, 'texture': count, 'total_words': count, 'total_tokens': count}}
    """
    contextual_nn_dir = Path(contextual_nn_dir)
    static_nn_dir = Path(static_nn_dir) if static_nn_dir else None
    
    data = defaultdict(lambda: {'color': 0, 'shape': 0, 'texture': 0, 'total_words': 0, 'total_tokens': 0})
    
    # Build checkpoint name
    if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_seed10_step12000-unsharded"
    else:
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_step12000-unsharded"
    
    print(f"\nSearching for checkpoint: {checkpoint_name}")
    
    # Load contextual nearest neighbors
    contextual_dir = contextual_nn_dir / checkpoint_name
    if contextual_dir.exists():
        print(f"\nLoading contextual NN results from {contextual_dir}...")
        
        # Find all contextual neighbor files
        nn_files = sorted(contextual_dir.glob("contextual_neighbors_visual*_contextual*_multi-gpu.json"))
        
        for nn_file in nn_files:
            # Extract contextual layer from filename
            match = re.search(r'_contextual(\d+)_multi-gpu\.json$', str(nn_file))
            if not match:
                continue
            
            layer_num = int(match.group(1))
            
            print(f"  Loading layer {layer_num}...")
            with open(nn_file, 'r') as f:
                nn_results = json.load(f)
            
            results_list = nn_results.get('results', [])
            
            for image_result in tqdm(results_list, desc=f"    Processing layer {layer_num}", leave=False):
                chunks = image_result.get('chunks', [])
                
                for chunk in chunks:
                    patches = chunk.get('patches', [])
                    
                    for patch in patches:
                        data[layer_num]['total_tokens'] += 1
                        
                        # Get nearest contextual neighbors
                        neighbors = patch.get('nearest_contextual_neighbors', [])
                        
                        # Check each neighbor's token
                        for neighbor in neighbors:
                            token_str = neighbor.get('token_str', '')
                            word_clean = token_str.lower().strip()
                            
                            data[layer_num]['total_words'] += 1
                            
                            attr_types = get_attribute_type(word_clean)
                            for attr_type in attr_types:
                                data[layer_num][attr_type] += 1
            
            print(f"    ✓ Layer {layer_num}: {data[layer_num]['total_tokens']} tokens")
    else:
        print(f"  WARNING: Contextual NN directory not found: {contextual_dir}")
    
    # Load static nearest neighbors (layer 0)
    if static_nn_dir:
        static_dir = static_nn_dir / checkpoint_name
        if static_dir.exists():
            print(f"\nLoading static NN results (layer 0) from {static_dir}...")
            
            # Find layer 0 file
            nn_file = static_dir / "nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json"
            
            if nn_file.exists():
                print(f"  Loading layer 0...")
                with open(nn_file, 'r') as f:
                    nn_results = json.load(f)
                
                # Get validation split
                validation_data = nn_results.get('splits', {}).get('validation', {})
                images = validation_data.get('images', [])
                
                for image_result in tqdm(images, desc="    Processing layer 0", leave=False):
                    chunks = image_result.get('chunks', [])
                    
                    for chunk in chunks:
                        patches = chunk.get('patches', [])
                        
                        for patch in patches:
                            data[0]['total_tokens'] += 1
                            
                            # Get nearest neighbors
                            neighbors = patch.get('nearest_neighbors', [])
                            
                            # Check each neighbor's token
                            for neighbor in neighbors:
                                token_str = neighbor.get('token', '')
                                word_clean = token_str.lower().strip()
                                
                                data[0]['total_words'] += 1
                                
                                attr_types = get_attribute_type(word_clean)
                                for attr_type in attr_types:
                                    data[0][attr_type] += 1
                
                print(f"    ✓ Layer 0: {data[0]['total_tokens']} tokens")
            else:
                print(f"  WARNING: Layer 0 file not found: {nn_file}")
        else:
            print(f"  WARNING: Static NN directory not found: {static_dir}")
    
    return dict(data)


def plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=False):
    """Plot a single subplot with visual attribute lines."""
    # Mapping from internal names to display names
    llm_display_names = {
        'llama3-8b': 'Llama3-8B',
        'olmo-7b': 'Olmo-7B',
        'qwen2-7b': 'Qwen2-7B',
        'average': 'Average'
    }
    
    encoder_display_names = {
        'vit-l-14-336': 'CLIP ViT-L/14',
        'siglip': 'SigLIP',
        'dinov2-large-336': 'DinoV2',
        'average': ''
    }
    
    llm_label = llm_display_names.get(llm, llm)
    encoder_label = encoder_display_names.get(vision_encoder, vision_encoder)
    
    # Handle average case
    if llm == 'average' and vision_encoder == 'average':
        title_text = 'Average'
    elif encoder_label:
        title_text = f'{llm_label} + {encoder_label}'
    else:
        title_text = llm_label
    
    # Sort layers
    layers = sorted(data.keys())
    
    # Calculate percentages of words that are visual attributes
    color_pct = []
    shape_pct = []
    texture_pct = []
    
    for layer in layers:
        total_words = data[layer]['total_words']
        if total_words > 0:
            # Percentage of all words that are this attribute type
            color_pct.append(data[layer]['color'] / total_words * 100)
            shape_pct.append(data[layer]['shape'] / total_words * 100)
            texture_pct.append(data[layer]['texture'] / total_words * 100)
        else:
            color_pct.append(0)
            shape_pct.append(0)
            texture_pct.append(0)
    
    # Define colors
    colors = {
        'color': '#E91E63',    # Pink/Red
        'shape': '#2196F3',    # Blue
        'texture': '#4CAF50'   # Green
    }
    
    # Plot lines
    line1 = ax.plot(layers, color_pct, marker='o', label='Color words', 
                    color=colors['color'], linewidth=2.5, markersize=7)
    line2 = ax.plot(layers, shape_pct, marker='s', label='Shape words',
                    color=colors['shape'], linewidth=2.5, markersize=7)
    line3 = ax.plot(layers, texture_pct, marker='^', label='Texture words',
                    color=colors['texture'], linewidth=2.5, markersize=7)
    
    # Customize plot - only add labels if not in combined mode
    if not is_combined:
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of NN Words', fontsize=12, fontweight='bold')
    
    # Add subplot title
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=8)
    
    # Set x-axis ticks
    ax.set_xticks(layers)
    if is_combined:
        ax.set_xticklabels([str(l) for l in layers], fontsize=11)
    else:
        ax.set_xticklabels([str(l) for l in layers], fontsize=11)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    if is_combined:
        ax.tick_params(axis='y', labelsize=11)
    else:
        ax.tick_params(axis='y', labelsize=11)
    
    # Add legend only if requested
    if show_legend:
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return line1[0], line2[0], line3[0]


def create_visualization(data, output_path, llm, vision_encoder):
    """Create line plot showing visual attribute occurrence across layers."""
    if not data:
        print("No data found to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot single subplot
    plot_single_subplot(ax, data, llm, vision_encoder, show_legend=True, is_combined=False)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Visualization also saved as PNG: {png_path}")
    
    plt.close()
    
    # Print data table
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
    llm_label = llm_display_names.get(llm, llm)
    encoder_label = encoder_display_names.get(vision_encoder, vision_encoder)
    
    layers = sorted(data.keys())
    color_pct = [data[layer]['color'] / data[layer]['total_words'] * 100 
                if data[layer]['total_words'] > 0 else 0 for layer in layers]
    shape_pct = [data[layer]['shape'] / data[layer]['total_words'] * 100 
                if data[layer]['total_words'] > 0 else 0 for layer in layers]
    texture_pct = [data[layer]['texture'] / data[layer]['total_words'] * 100 
                  if data[layer]['total_words'] > 0 else 0 for layer in layers]
    
    print(f"\n{llm_label} + {encoder_label}:")
    print("="*110)
    print(f"{'Layer':<10}{'Color %':<15}{'Shape %':<15}{'Texture %':<15}{'Total Words':<18}{'Total Tokens':<15}")
    print("-" * 110)
    for i, layer in enumerate(layers):
        print(f"{layer:<10}{color_pct[i]:<15.2f}{shape_pct[i]:<15.2f}"
              f"{texture_pct[i]:<15.2f}{data[layer]['total_words']:<18}{data[layer]['total_tokens']:<15}")
    print()


def compute_average_data(all_data_dict):
    """Compute average across all model combinations."""
    # Collect all layers from all combinations
    all_layers = set()
    for data in all_data_dict.values():
        all_layers.update(data.keys())
    all_layers = sorted(all_layers)
    
    if not all_layers:
        return {}
    
    # Initialize average data
    avg_data = defaultdict(lambda: {'color': 0, 'shape': 0, 'texture': 0, 'total_words': 0, 'total_tokens': 0})
    count = 0
    
    # Sum across all combinations
    for data in all_data_dict.values():
        if data:
            count += 1
            for layer in all_layers:
                if layer in data:
                    avg_data[layer]['color'] += data[layer]['color']
                    avg_data[layer]['shape'] += data[layer]['shape']
                    avg_data[layer]['texture'] += data[layer]['texture']
                    avg_data[layer]['total_words'] += data[layer]['total_words']
                    avg_data[layer]['total_tokens'] += data[layer]['total_tokens']
    
    # Average
    if count > 0:
        for layer in all_layers:
            avg_data[layer]['color'] /= count
            avg_data[layer]['shape'] /= count
            avg_data[layer]['texture'] /= count
            avg_data[layer]['total_words'] /= count
            avg_data[layer]['total_tokens'] /= count
    
    return dict(avg_data)


def create_average_plot(avg_data, output_path):
    """Create a single plot showing the average across all 9 combinations."""
    if not avg_data:
        print("No average data to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot average (using dummy labels since we'll add our own title)
    plot_single_subplot(ax, avg_data, 'average', 'average', show_legend=True, is_combined=False)
    
    # Update title
    ax.set_title('Visual Attribute Word Frequency Across Layers\n(averaged across all 9 model combinations)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAverage plot saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Average plot also saved as PNG: {png_path}")
    
    plt.close()


def create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders):
    """Create a 3x3 subplot grid with all 9 combinations."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    plot_idx = 0
    handles = None
    labels = None
    
    # Plot each combination
    for llm_idx, llm in enumerate(all_llms):
        for encoder_idx, vision_encoder in enumerate(all_vision_encoders):
            ax = axes_flat[plot_idx]
            
            # Get data for this combination
            data = all_data_dict.get((llm, vision_encoder), {})
            
            if data:
                # Plot the subplot
                line1, line2, line3 = plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=True)
                
                # Store handles and labels from first subplot for shared legend
                if plot_idx == 0:
                    handles = [line1, line2, line3]
                    labels = ['Color words', 'Shape words', 'Texture words']
            else:
                # No data - show empty plot with message
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plot_idx += 1
    
    # Add shared legend at the bottom center
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14, framealpha=0.9)
    
    # No figure-level labels (removed to avoid overlaying subplots)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space for legend
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Combined plot also saved as PNG: {png_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze visual attribute word occurrence across layers'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default='olmo-7b',
        choices=['olmo-7b', 'llama3-8b', 'qwen2-7b'],
        help='LLM model name'
    )
    parser.add_argument(
        '--vision-encoder',
        type=str,
        default='vit-l-14-336',
        choices=['vit-l-14-336', 'siglip', 'dinov2-large-336'],
        help='Vision encoder name'
    )
    parser.add_argument(
        '--contextual-nn-dir',
        type=str,
        default=None,
        help='Directory containing contextual nearest neighbors results'
    )
    parser.add_argument(
        '--static-nn-dir',
        type=str,
        default=None,
        help='Directory containing static nearest neighbors results (layer 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for visualization (PDF or PNG). Ignored if --combined is set.'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Create a single 3x3 combined plot with all 9 combinations'
    )
    
    args = parser.parse_args()
    
    # Set up paths relative to repository root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    
    # Set default input directories if not provided
    if args.contextual_nn_dir is None:
        args.contextual_nn_dir = str(repo_root / 'analysis_results' / 'contextual_nearest_neighbors')
    if args.static_nn_dir is None:
        args.static_nn_dir = str(repo_root / 'analysis_results' / 'nearest_neighbors')
    
    # Define all model combinations
    all_llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    all_vision_encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    # Set up output directory
    output_dir = repo_root / 'analysis_results' / 'layer_evolution'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up cache directory
    cache_dir = repo_root / 'analysis_results' / 'layer_evolution' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'visual_attributes_data.pkl'
    
    if args.combined:
        # Create combined 3x3 plot
        print(f"\n{'='*80}")
        print("Creating combined 3x3 plot for all 9 model combinations")
        print(f"{'='*80}")
        print(f"LLMs: {all_llms}")
        print(f"Vision encoders: {all_vision_encoders}")
        print(f"{'='*80}\n")
        
        # Try to load from cache
        all_data_dict = {}
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                all_data_dict = pickle.load(f)
            print(f"  ✓ Loaded {len(all_data_dict)} cached combinations")
        else:
            print(f"No cache found. Loading data from JSON files (this may take 10+ minutes)...")
        
        # Load missing combinations
        for llm in all_llms:
            for vision_encoder in all_vision_encoders:
                key = (llm, vision_encoder)
                if key not in all_data_dict:
                    print(f"Loading data for {llm} + {vision_encoder}...")
                    data = load_visual_attribute_counts(
                        args.contextual_nn_dir, 
                        args.static_nn_dir,
                        llm,
                        vision_encoder
                    )
                    if data:
                        all_data_dict[key] = data
                        layers = sorted(data.keys())
                        print(f"  ✓ Found {len(layers)} layers: {layers}")
                    else:
                        print(f"  ⚠ No data found")
        
        # Save to cache
        if all_data_dict:
            print(f"\nSaving data to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(all_data_dict, f)
            print(f"  ✓ Cache saved")
        
        # Create combined plot
        output_path = output_dir / 'visual_attributes_combined.pdf'
        print(f"\nCreating combined visualization...")
        create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders)
        
        # Create average plot
        avg_data = compute_average_data(all_data_dict)
        if avg_data:
            avg_output_path = output_dir / 'visual_attributes_average.pdf'
            print(f"\nCreating average visualization...")
            create_average_plot(avg_data, avg_output_path)
        
        print(f"\n{'='*80}")
        print(f"Combined plot created successfully!")
        print(f"{'='*80}")
    else:
        # Process single combination
        # Use defaults if not provided
        llm = args.llm if args.llm else 'olmo-7b'
        vision_encoder = args.vision_encoder if args.vision_encoder else 'vit-l-14-336'
        
        # Auto-generate output path if not provided
        if args.output is None:
            output_path = output_dir / f'visual_attributes_{llm}_{vision_encoder}.pdf'
        else:
            output_path = Path(args.output)
        
        # Load data
        print(f"Loading nearest neighbor results for {llm} + {vision_encoder}...")
        print(f"  Contextual NNs: {args.contextual_nn_dir}")
        print(f"  Static NNs: {args.static_nn_dir}")
        
        data = load_visual_attribute_counts(
            args.contextual_nn_dir, 
            args.static_nn_dir,
            llm,
            vision_encoder
        )
        
        if not data:
            print(f"ERROR: No results found for {llm} + {vision_encoder}!")
            return
        
        # Print summary
        layers = sorted(data.keys())
        print(f"\nFound data for {len(layers)} layers: {layers}")
        
        # Create visualization
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Creating visual attributes visualization...")
        print("="*60)
        
        create_visualization(data, output_path, llm, vision_encoder)
        
        print("\n" + "="*60)
        print("Visualization created successfully!")
        print("="*60)


if __name__ == '__main__':
    main()
