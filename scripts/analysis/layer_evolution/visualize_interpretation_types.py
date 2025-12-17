#!/usr/bin/env python3
"""
Visualize the evolution of interpretation types (concrete, abstract, global) across layers.

This script analyzes how the ratio of concrete/abstract/global interpretable tokens
changes across contextual layers for a specific model combination.
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
import pickle


def load_interpretation_types(results_dir, nn_results_dir, llm, vision_encoder):
    """
    Load results and count interpretation types (concrete, abstract, global) per layer.
    
    Returns:
        dict: {layer: {'concrete': count, 'abstract': count, 'global': count, 'total_interpretable': count}}
    """
    results_dir = Path(results_dir)
    nn_results_dir = Path(nn_results_dir) if nn_results_dir else None
    
    data = defaultdict(lambda: {'concrete': 0, 'abstract': 0, 'global': 0, 'total_interpretable': 0})
    
    # Pattern to match for the specific model
    # Handle seed10 case for qwen2
    if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
        model_pattern = f"{llm}_{vision_encoder}_seed10"
    else:
        model_pattern = f"{llm}_{vision_encoder}"
    
    print(f"\nSearching for results matching: {model_pattern}")
    
    # Load contextual results
    print(f"\nSearching contextual results in {results_dir}...")
    for results_file in results_dir.glob("**/results_*.json"):
        path_str = str(results_file)
        
        # Check if this file matches our model
        if model_pattern not in path_str:
            continue
        
        # Extract layer from path
        match = re.search(r'_contextual(\d+)_', path_str)
        if not match:
            continue
        
        layer_num = int(match.group(1))
        
        # Load JSON and process
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        results_list = results.get('results', [])
        
        for result in results_list:
            if not result.get('interpretable', False):
                continue
            
            data[layer_num]['total_interpretable'] += 1
            
            gpt_response = result.get('gpt_response', {})
            concrete_words = gpt_response.get('concrete_words', [])
            abstract_words = gpt_response.get('abstract_words', [])
            global_words = gpt_response.get('global_words', [])
            
            # Count as concrete if it has concrete words
            if concrete_words:
                data[layer_num]['concrete'] += 1
            # Count as abstract if it has abstract words (and no concrete)
            elif abstract_words:
                data[layer_num]['abstract'] += 1
            # Count as global if it has global words (and no concrete/abstract)
            elif global_words:
                data[layer_num]['global'] += 1
        
        print(f"  ✓ Loaded contextual layer {layer_num}: {len(results_list)} patches")
    
    # Load layer 0 from nearest neighbors results
    if nn_results_dir and nn_results_dir.exists():
        print(f"\nSearching for layer 0 in {nn_results_dir}...")
        # Collect all matching files, prioritizing non-ablation files
        matching_files = []
        for results_file in nn_results_dir.glob("**/results_*.json"):
            path_str = str(results_file)
            
            # Check if this file matches our model
            if model_pattern not in path_str:
                continue
            
            # Extract layer from path - only process layer 0
            match = re.search(r'_layer(\d+)_', path_str)
            if not match:
                continue
            layer_num = int(match.group(1))
            if layer_num != 0:
                continue
            
            # Skip ablation files - prioritize main results
            is_ablation = '/ablations/' in path_str
            matching_files.append((is_ablation, results_file))
        
        # Sort to put non-ablation files first
        matching_files.sort(key=lambda x: x[0])
        
        if not matching_files:
            print(f"  ⚠ No layer 0 results found for {model_pattern}")
        else:
            # Use the first (non-ablation) file
            _, results_file = matching_files[0]
            print(f"  Using file: {results_file}")
            
            # Load JSON and process
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Layer 0 results use 'responses' dict instead of 'results' list
            responses_dict = results.get('responses', {})
            
            # Flatten responses dict into a list
            results_list = []
            for image_path, patches in responses_dict.items():
                results_list.extend(patches)
            
            for result in results_list:
                gpt_response = result.get('gpt_response', {})
                interpretable = gpt_response.get('interpretable', False)
                
                if not interpretable:
                    continue
                
                data[0]['total_interpretable'] += 1
                
                concrete_words = gpt_response.get('concrete_words', [])
                abstract_words = gpt_response.get('abstract_words', [])
                global_words = gpt_response.get('global_words', [])
                
                # Count as concrete if it has concrete words
                if concrete_words:
                    data[0]['concrete'] += 1
                # Count as abstract if it has abstract words (and no concrete)
                elif abstract_words:
                    data[0]['abstract'] += 1
                # Count as global if it has global words (and no concrete/abstract)
                elif global_words:
                    data[0]['global'] += 1
            
            print(f"  ✓ Loaded layer 0: {len(results_list)} patches")
    
    return dict(data)


def plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=False):
    """Plot a single subplot with stacked bars."""
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
    
    # Calculate percentages
    concrete_pcts = []
    abstract_pcts = []
    global_pcts = []
    
    for layer in layers:
        total = data[layer]['total_interpretable']
        if total > 0:
            concrete_pcts.append(data[layer]['concrete'] / total * 100)
            abstract_pcts.append(data[layer]['abstract'] / total * 100)
            global_pcts.append(data[layer]['global'] / total * 100)
        else:
            concrete_pcts.append(0)
            abstract_pcts.append(0)
            global_pcts.append(0)
    
    # Define colors
    colors = {
        'concrete': '#2E7D32',   # Dark green
        'abstract': '#1976D2',   # Medium blue
        'global': '#F57C00'      # Orange
    }
    
    # Create stacked bars
    x = np.arange(len(layers))
    width = 0.6
    
    # Stack: concrete at bottom, abstract in middle, global on top
    p1 = ax.bar(x, concrete_pcts, width, label='Concrete', color=colors['concrete'])
    p2 = ax.bar(x, abstract_pcts, width, bottom=concrete_pcts, 
                label='Abstract', color=colors['abstract'])
    p3 = ax.bar(x, global_pcts, width, 
                bottom=np.array(concrete_pcts) + np.array(abstract_pcts),
                label='Global', color=colors['global'])
    
    # Customize plot - only add labels if not in combined mode
    if not is_combined:
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of Interpretable Tokens by NN Type', fontsize=12, fontweight='bold')
    
    # Add subplot title
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=8)
    
    # Set x-axis ticks
    ax.set_xticks(x)
    if is_combined:
        ax.set_xticklabels([f'{l}' for l in layers], fontsize=11)
    else:
        ax.set_xticklabels([f'{l}' for l in layers], fontsize=11)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    if is_combined:
        ax.tick_params(axis='y', labelsize=11)
    else:
        ax.tick_params(axis='y', labelsize=11)
    
    # Add legend only if requested
    if show_legend:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    return p1, p2, p3


def create_stacked_bar_plot(data, output_path, llm, vision_encoder, 
                             title='Evolution of Interpretation Types Across Layers'):
    """Create a stacked bar plot showing the ratio of interpretation types per layer."""
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
    
    llm_label = llm_display_names.get(llm, llm)
    encoder_label = encoder_display_names.get(vision_encoder, vision_encoder)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot single subplot
    plot_single_subplot(ax, data, llm, vision_encoder, show_legend=True, is_combined=False)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nStacked bar plot saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Stacked bar plot also saved as PNG: {png_path}")
    
    plt.close()
    
    # Print data table
    full_title = f'{llm_label} + {encoder_label}'
    layers = sorted(data.keys())
    concrete_pcts = [data[layer]['concrete'] / data[layer]['total_interpretable'] * 100 
                     if data[layer]['total_interpretable'] > 0 else 0 for layer in layers]
    abstract_pcts = [data[layer]['abstract'] / data[layer]['total_interpretable'] * 100 
                     if data[layer]['total_interpretable'] > 0 else 0 for layer in layers]
    global_pcts = [data[layer]['global'] / data[layer]['total_interpretable'] * 100 
                   if data[layer]['total_interpretable'] > 0 else 0 for layer in layers]
    
    print(f"\n{full_title}:")
    print("\n" + "="*80)
    print(f"{'Layer':<10}{'Concrete %':<15}{'Abstract %':<15}{'Global %':<15}{'Total Interp.':<15}")
    print("-" * 80)
    for i, layer in enumerate(layers):
        print(f"{layer:<10}{concrete_pcts[i]:<15.1f}{abstract_pcts[i]:<15.1f}"
              f"{global_pcts[i]:<15.1f}{data[layer]['total_interpretable']:<15}")
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
    avg_data = defaultdict(lambda: {'concrete': 0, 'abstract': 0, 'global': 0, 'total_interpretable': 0})
    count = 0
    
    # Sum across all combinations
    for data in all_data_dict.values():
        if data:
            count += 1
            for layer in all_layers:
                if layer in data:
                    avg_data[layer]['concrete'] += data[layer]['concrete']
                    avg_data[layer]['abstract'] += data[layer]['abstract']
                    avg_data[layer]['global'] += data[layer]['global']
                    avg_data[layer]['total_interpretable'] += data[layer]['total_interpretable']
    
    # Average
    if count > 0:
        for layer in all_layers:
            avg_data[layer]['concrete'] /= count
            avg_data[layer]['abstract'] /= count
            avg_data[layer]['global'] /= count
            avg_data[layer]['total_interpretable'] /= count
    
    return dict(avg_data)


def create_average_plot(avg_data, output_path):
    """Create a single plot showing the average across all 9 combinations."""
    if not avg_data:
        print("No average data to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot average (using dummy labels since we'll add our own title)
    plot_single_subplot(ax, avg_data, 'average', 'average', show_legend=True, is_combined=False)
    
    # Update title
    ax.set_title('Interpretation Types Across Layers\n(averaged across all 9 model combinations)', 
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
                p1, p2, p3 = plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=True)
                
                # Store handles and labels from first subplot for shared legend
                if plot_idx == 0:
                    handles = [p1, p2, p3]
                    labels = ['Concrete', 'Abstract', 'Global']
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


def process_single_combination(llm, vision_encoder, results_dir, nn_results_dir, output_dir, output_path=None):
    """Process a single LLM + vision encoder combination."""
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = output_dir / f'interpretation_types_{llm}_{vision_encoder}.pdf'
    else:
        output_path = Path(output_path)
    
    # Load data
    print(f"\n{'='*80}")
    print(f"Processing: {llm} + {vision_encoder}")
    print(f"{'='*80}")
    print(f"  Contextual results: {results_dir}")
    print(f"  Nearest neighbors: {nn_results_dir}")
    data = load_interpretation_types(
        results_dir, 
        nn_results_dir,
        llm,
        vision_encoder
    )
    
    if not data:
        print(f"  ⚠ WARNING: No results found for {llm} + {vision_encoder}!")
        return False
    
    # Print summary
    layers = sorted(data.keys())
    print(f"\n  Found data for {len(layers)} layers: {layers}")
    
    # Create visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Creating interpretation types visualization...")
    create_stacked_bar_plot(data, output_path, llm, vision_encoder)
    
    print(f"  ✓ Successfully created: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize evolution of interpretation types across layers'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default=None,
        choices=['olmo-7b', 'llama3-8b', 'qwen2-7b'],
        help='LLM model name (if not provided and --all is not set, defaults to olmo-7b)'
    )
    parser.add_argument(
        '--vision-encoder',
        type=str,
        default=None,
        choices=['vit-l-14-336', 'siglip', 'dinov2-large-336'],
        help='Vision encoder name (if not provided and --all is not set, defaults to vit-l-14-336)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all 9 combinations of LLMs and vision encoders (creates individual plots)'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Create a single 3x3 combined plot with all 9 combinations'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing LLM judge contextual NN results (default: analysis_results/llm_judge_contextual_nn)'
    )
    parser.add_argument(
        '--nn-results-dir',
        type=str,
        default=None,
        help='Directory containing LLM judge nearest neighbors results for layer 0 (default: analysis_results/llm_judge_nearest_neighbors)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for visualization (PDF or PNG). If not provided, auto-generated. Ignored if --all is set.'
    )
    
    args = parser.parse_args()
    
    # Set up paths relative to repository root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    
    # Set default input directories if not provided
    if args.results_dir is None:
        args.results_dir = str(repo_root / 'analysis_results' / 'llm_judge_contextual_nn')
    if args.nn_results_dir is None:
        args.nn_results_dir = str(repo_root / 'analysis_results' / 'llm_judge_nearest_neighbors')
    
    # Define all model combinations
    all_llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    all_vision_encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    # Set up output directory
    output_dir = repo_root / 'analysis_results' / 'layer_evolution'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up cache directory
    cache_dir = repo_root / 'analysis_results' / 'layer_evolution' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'interpretation_types_data.pkl'
    
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
            print(f"No cache found. Loading data from JSON files...")
        
        # Load missing combinations
        for llm in all_llms:
            for vision_encoder in all_vision_encoders:
                key = (llm, vision_encoder)
                if key not in all_data_dict:
                    print(f"Loading data for {llm} + {vision_encoder}...")
                    data = load_interpretation_types(
                        args.results_dir, 
                        args.nn_results_dir,
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
        output_path = output_dir / 'interpretation_types_combined.pdf'
        print(f"\nCreating combined visualization...")
        create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders)
        
        # Create average plot
        avg_data = compute_average_data(all_data_dict)
        if avg_data:
            avg_output_path = output_dir / 'interpretation_types_average.pdf'
            print(f"\nCreating average visualization...")
            create_average_plot(avg_data, avg_output_path)
        
        print(f"\n{'='*80}")
        print(f"Combined plot created successfully!")
        print(f"{'='*80}")
    elif args.all:
        # Process all 9 combinations (individual plots)
        print(f"\n{'='*80}")
        print("Processing all 9 model combinations")
        print(f"{'='*80}")
        print(f"LLMs: {all_llms}")
        print(f"Vision encoders: {all_vision_encoders}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
        
        success_count = 0
        for llm in all_llms:
            for vision_encoder in all_vision_encoders:
                success = process_single_combination(
                    llm, vision_encoder, args.results_dir, args.nn_results_dir, output_dir
                )
                if success:
                    success_count += 1
        
        print(f"\n{'='*80}")
        print(f"Completed: {success_count}/9 combinations processed successfully")
        print(f"{'='*80}")
    else:
        # Process single combination
        # Use defaults if not provided
        llm = args.llm if args.llm else 'olmo-7b'
        vision_encoder = args.vision_encoder if args.vision_encoder else 'vit-l-14-336'
        
        # Auto-generate output path if not provided
        if args.output is None:
            output_path = output_dir / f'interpretation_types_{llm}_{vision_encoder}.pdf'
        else:
            output_path = Path(args.output)
        
        print(f"Output directory: {output_dir}")
        
        success = process_single_combination(
            llm, vision_encoder, args.results_dir, args.nn_results_dir, output_dir, output_path
        )
        
        if not success:
            print(f"\nERROR: Failed to process {llm} + {vision_encoder}!")


if __name__ == '__main__':
    main()

