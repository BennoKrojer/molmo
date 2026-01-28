#!/usr/bin/env python3
"""
Compare interpretation types (concrete, abstract, global) across multiple models.

This script creates side-by-side visualizations comparing how the ratio of 
concrete/abstract/global interpretable tokens changes across contextual layers
for multiple model combinations.
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


def load_interpretation_types(results_dir, nn_results_dir, llm, vision_encoder):
    """
    Load results and count interpretation types (concrete, abstract, global) per layer.
    
    Now uses contextual NN results for ALL layers including layer 0.
    The nn_results_dir parameter is kept for backwards compatibility but not used.
    
    Returns:
        dict: {layer: {'concrete': count, 'abstract': count, 'global': count, 'total_interpretable': count}}
    """
    results_dir = Path(results_dir)
    
    data = defaultdict(lambda: {'concrete': 0, 'abstract': 0, 'global': 0, 'total_interpretable': 0})
    
    # Pattern to match for the specific model
    # Handle seed10 case for qwen2
    if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
        model_pattern = f"{llm}_{vision_encoder}_seed10"
    else:
        model_pattern = f"{llm}_{vision_encoder}"
    
    # Load all contextual results (including layer 0)
    for results_file in results_dir.glob("**/results_*.json"):
        path_str = str(results_file)

        # Check if this file matches our model
        if model_pattern not in path_str:
            continue

        # Skip ablations directory (CLAUDE.md rule: never use ** glob without excluding ablations)
        if '/ablations/' in path_str:
            continue

        # Extract layer from path (matches _contextual0_, _contextual1_, etc.)
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
    
    return dict(data)


def create_comparison_plot(all_data, output_path, title='Evolution of Interpretation Types Across Layers'):
    """Create a comparison plot with multiple subplots for different models."""
    if not all_data:
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
    
    # Define colors
    colors = {
        'concrete': '#2E7D32',   # Dark green
        'abstract': '#1976D2',   # Medium blue
        'global': '#F57C00'      # Orange
    }
    
    # Create figure with subplots
    n_models = len(all_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    
    # Handle case where there's only one model
    if n_models == 1:
        axes = [axes]
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot each model
    for idx, ((llm, vision_encoder), data) in enumerate(sorted(all_data.items())):
        ax = axes[idx]
        
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
        
        # Customize subplot
        ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('% of Interpretable Tokens\nby NN Type', fontsize=11, fontweight='bold')
        
        # Model-specific title
        llm_label = llm_display_names.get(llm, llm)
        encoder_label = encoder_display_names.get(vision_encoder, vision_encoder)
        ax.set_title(f'{llm_label} + {encoder_label}', fontsize=12, fontweight='bold', pad=10)
        
        # Set x-axis ticks
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l}' for l in layers], fontsize=9)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add legend only to the last subplot
        if idx == n_models - 1:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot also saved as PNG: {png_path}")
    
    plt.close()
    
    # Print data table for each model
    for (llm, vision_encoder), data in sorted(all_data.items()):
        llm_label = llm_display_names.get(llm, llm)
        encoder_label = encoder_display_names.get(vision_encoder, vision_encoder)
        
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
        
        print(f"\n{llm_label} + {encoder_label}:")
        print("="*80)
        print(f"{'Layer':<10}{'Concrete %':<15}{'Abstract %':<15}{'Global %':<15}{'Total Interp.':<15}")
        print("-" * 80)
        for i, layer in enumerate(layers):
            print(f"{layer:<10}{concrete_pcts[i]:<15.1f}{abstract_pcts[i]:<15.1f}"
                  f"{global_pcts[i]:<15.1f}{data[layer]['total_interpretable']:<15}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare interpretation types across multiple models'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Model combinations as "llm:encoder" (e.g., "olmo-7b:vit-l-14-336" "qwen2-7b:vit-l-14-336")'
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
        help='Output path for visualization (PDF or PNG). If not provided, auto-generated.'
    )
    
    args = parser.parse_args()
    
    # Set up paths relative to repository root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    
    # Set default input directories if not provided
    if args.results_dir is None:
        args.results_dir = str(repo_root / 'analysis_results' / 'llm_judge_contextual_nn')
    if args.nn_results_dir is None:
        args.nn_results_dir = str(repo_root / 'analysis_results' / 'llm_judge_nearest_neighbors')
    
    # Parse model specifications
    model_specs = []
    for model_spec in args.models:
        parts = model_spec.split(':')
        if len(parts) != 2:
            print(f"ERROR: Invalid model specification '{model_spec}'. Use format 'llm:encoder'")
            return
        llm, encoder = parts
        model_specs.append((llm, encoder))
    
    # Load data for all models
    all_data = {}
    print(f"Loading results for {len(model_specs)} model(s)...")
    print(f"  Contextual results: {args.results_dir}")
    print(f"  Nearest neighbors: {args.nn_results_dir}")
    
    for llm, vision_encoder in model_specs:
        print(f"\n{'='*60}")
        print(f"Loading: {llm} + {vision_encoder}")
        print('='*60)
        
        data = load_interpretation_types(
            args.results_dir, 
            args.nn_results_dir,
            llm,
            vision_encoder
        )
        
        if not data:
            print(f"  WARNING: No results found for {llm} + {vision_encoder}!")
            continue
        
        layers = sorted(data.keys())
        print(f"  Found data for {len(layers)} layers: {layers}")
        
        all_data[(llm, vision_encoder)] = data
    
    if not all_data:
        print("\nERROR: No data found for any model!")
        return
    
    # Auto-generate output path if not provided
    if args.output is None:
        output_dir = repo_root / 'analysis_results' / 'layer_evolution'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from model names
        model_names = '_vs_'.join([f"{llm}_{enc}" for llm, enc in model_specs])
        args.output = output_dir / f'compare_interpretation_types_{model_names}.pdf'
        print(f"\nOutput directory: {output_dir}")
    else:
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating comparison visualization...")
    print("="*60)
    
    create_comparison_plot(all_data, output_path)
    
    print("\n" + "="*60)
    print("Visualization created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

