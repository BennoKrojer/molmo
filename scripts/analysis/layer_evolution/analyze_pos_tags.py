#!/usr/bin/env python3
"""
Analyze parts of speech in nearest neighbors across layers.

This script samples nearest neighbor tokens/phrases and counts POS tags,
providing insights into how the linguistic structure evolves across layers.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import argparse
from tqdm import tqdm
import pickle
import random

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("WARNING: spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")


def load_pos_tag_counts(contextual_nn_dir, static_nn_dir, llm, vision_encoder, 
                        nlp, sample_size=500, seed=42):
    """
    Load NN results, sample tokens/phrases, and count POS tags per layer.
    
    Args:
        contextual_nn_dir: Directory with contextual NN results
        static_nn_dir: Directory with static NN results (layer 0)
        llm: LLM name
        vision_encoder: Vision encoder name
        nlp: spaCy language model
        sample_size: Number of items to sample per layer
        seed: Random seed for reproducibility
    
    Returns:
        dict: {layer: {pos_tag: count, ...}}
    """
    random.seed(seed)
    np.random.seed(seed)
    
    contextual_nn_dir = Path(contextual_nn_dir)
    static_nn_dir = Path(static_nn_dir) if static_nn_dir else None
    
    data = defaultdict(Counter)
    
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
            
            # Collect all phrases from this layer
            all_phrases = []
            for image_result in results_list:
                chunks = image_result.get('chunks', [])
                for chunk in chunks:
                    patches = chunk.get('patches', [])
                    for patch in patches:
                        neighbors = patch.get('nearest_contextual_neighbors', [])
                        for neighbor in neighbors:
                            token_str = neighbor.get('token_str', '').strip()
                            if token_str:
                                all_phrases.append(token_str)
            
            # Sample phrases
            if len(all_phrases) > sample_size:
                sampled_phrases = random.sample(all_phrases, sample_size)
            else:
                sampled_phrases = all_phrases
            
            print(f"    Sampling {len(sampled_phrases)} phrases from {len(all_phrases)} total...")
            
            # Tag POS for each phrase
            for phrase in tqdm(sampled_phrases, desc=f"    Tagging layer {layer_num}", leave=False):
                doc = nlp(phrase)
                for token in doc:
                    # Skip punctuation and spaces
                    if not token.is_punct and not token.is_space:
                        pos_tag = token.pos_
                        data[layer_num][pos_tag] += 1
            
            print(f"    ✓ Layer {layer_num}: {len(sampled_phrases)} phrases tagged")
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
                
                # Collect all tokens from layer 0
                all_tokens = []
                for image_result in images:
                    chunks = image_result.get('chunks', [])
                    for chunk in chunks:
                        patches = chunk.get('patches', [])
                        for patch in patches:
                            neighbors = patch.get('nearest_neighbors', [])
                            for neighbor in neighbors:
                                token_str = neighbor.get('token', '').strip()
                                if token_str:
                                    all_tokens.append(token_str)
                
                # Sample tokens
                if len(all_tokens) > sample_size:
                    sampled_tokens = random.sample(all_tokens, sample_size)
                else:
                    sampled_tokens = all_tokens
                
                print(f"    Sampling {len(sampled_tokens)} tokens from {len(all_tokens)} total...")
                
                # Tag POS for each token (single word)
                for token_str in tqdm(sampled_tokens, desc="    Tagging layer 0", leave=False):
                    doc = nlp(token_str)
                    for token in doc:
                        # Skip punctuation and spaces
                        if not token.is_punct and not token.is_space:
                            pos_tag = token.pos_
                            data[0][pos_tag] += 1
                
                print(f"    ✓ Layer 0: {len(sampled_tokens)} tokens tagged")
            else:
                print(f"  WARNING: Layer 0 file not found: {nn_file}")
        else:
            print(f"  WARNING: Static NN directory not found: {static_dir}")
    
    # Convert Counter to dict for consistency
    return {layer: dict(counter) for layer, counter in data.items()}


def plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=False, top_n_tags=8):
    """Plot a single subplot with POS tag percentages."""
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
    
    # Get all POS tags across all layers
    all_pos_tags = set()
    for layer_data in data.values():
        all_pos_tags.update(layer_data.keys())
    
    # Focus on most common POS tags for readability
    # Count total occurrences across all layers
    pos_tag_totals = Counter()
    for layer_data in data.values():
        for pos_tag, count in layer_data.items():
            pos_tag_totals[pos_tag] += count
    
    # Select top N most common POS tags
    top_pos_tags = [tag for tag, _ in pos_tag_totals.most_common(top_n_tags)]
    
    # Calculate percentages for each layer
    pos_data = {pos_tag: [] for pos_tag in top_pos_tags}
    
    for layer in layers:
        layer_data = data.get(layer, {})
        total_tokens = sum(layer_data.values())
        
        if total_tokens > 0:
            for pos_tag in top_pos_tags:
                count = layer_data.get(pos_tag, 0)
                pos_data[pos_tag].append(count / total_tokens * 100)
        else:
            for pos_tag in top_pos_tags:
                pos_data[pos_tag].append(0)
    
    # Define colors for different POS tags
    # Using a color palette that's distinct
    colors_list = sns.color_palette("husl", len(top_pos_tags))
    colors_dict = {tag: colors_list[i] for i, tag in enumerate(top_pos_tags)}
    
    # Plot lines for each POS tag
    lines = []
    for pos_tag in top_pos_tags:
        line = ax.plot(layers, pos_data[pos_tag], marker='o', label=pos_tag,
                      color=colors_dict[pos_tag], linewidth=2, markersize=6)
        lines.append(line[0])
    
    # Customize plot
    if not is_combined:
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of Tokens', fontsize=12, fontweight='bold')
    
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
        ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return lines


def create_visualization(data, output_path, llm, vision_encoder, top_n_tags=8):
    """Create line plot showing POS tag occurrence across layers."""
    if not data:
        print("No data found to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot single subplot
    plot_single_subplot(ax, data, llm, vision_encoder, show_legend=True, is_combined=False, top_n_tags=top_n_tags)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Visualization also saved as PNG: {png_path}")
    
    plt.close()


def compute_average_data(all_data_dict):
    """Compute average across all model combinations."""
    # Collect all layers from all combinations
    all_layers = set()
    for data in all_data_dict.values():
        all_layers.update(data.keys())
    all_layers = sorted(all_layers)
    
    if not all_layers:
        return {}
    
    # Collect all POS tags
    all_pos_tags = set()
    for data in all_data_dict.values():
        for layer_data in data.values():
            all_pos_tags.update(layer_data.keys())
    
    # Initialize average data
    avg_data = defaultdict(lambda: defaultdict(float))
    count = 0
    
    # Sum across all combinations
    for data in all_data_dict.values():
        if data:
            count += 1
            for layer in all_layers:
                if layer in data:
                    layer_data = data[layer]
                    for pos_tag in all_pos_tags:
                        avg_data[layer][pos_tag] += layer_data.get(pos_tag, 0)
    
    # Average
    if count > 0:
        for layer in all_layers:
            for pos_tag in all_pos_tags:
                avg_data[layer][pos_tag] /= count
    
    return {layer: dict(pos_dict) for layer, pos_dict in avg_data.items()}


def create_average_plot(avg_data, output_path, top_n_tags=8):
    """Create a single plot showing the average across all 9 combinations."""
    if not avg_data:
        print("No average data to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot average
    plot_single_subplot(ax, avg_data, 'average', 'average', show_legend=True, is_combined=False, top_n_tags=top_n_tags)
    
    # Update title
    ax.set_title('Parts of Speech Across Layers\n(averaged across all 9 model combinations)', 
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


def create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders, top_n_tags=8):
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
                lines = plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=True, top_n_tags=top_n_tags)
                
                # Store handles and labels from first subplot for shared legend
                if plot_idx == 0 and lines:
                    handles = lines
                    # Get labels from the plot
                    labels = [line.get_label() for line in lines]
            else:
                # No data - show empty plot with message
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plot_idx += 1
    
    # Add shared legend at the bottom center
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, framealpha=0.9)
    
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
    if not SPACY_AVAILABLE:
        print("ERROR: spaCy is required. Install with:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        return
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("ERROR: spaCy model 'en_core_web_sm' not found.")
        print("Download it with: python -m spacy download en_core_web_sm")
        return
    
    parser = argparse.ArgumentParser(
        description='Analyze parts of speech in nearest neighbors across layers'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default=None,
        choices=['olmo-7b', 'llama3-8b', 'qwen2-7b'],
        help='LLM model name (if not provided and --combined is not set, defaults to olmo-7b)'
    )
    parser.add_argument(
        '--vision-encoder',
        type=str,
        default=None,
        choices=['vit-l-14-336', 'siglip', 'dinov2-large-336'],
        help='Vision encoder name (if not provided and --combined is not set, defaults to vit-l-14-336)'
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
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='Number of tokens/phrases to sample per layer (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--top-n-tags',
        type=int,
        default=8,
        help='Number of top POS tags to display in plots (default: 8)'
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
    cache_file = cache_dir / f'pos_tags_data_s{args.sample_size}_seed{args.seed}.pkl'
    
    if args.combined:
        # Create combined 3x3 plot
        print(f"\n{'='*80}")
        print("Creating combined 3x3 plot for all 9 model combinations")
        print(f"{'='*80}")
        print(f"LLMs: {all_llms}")
        print(f"Vision encoders: {all_vision_encoders}")
        print(f"Sample size: {args.sample_size} per layer")
        print(f"Seed: {args.seed}")
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
                    data = load_pos_tag_counts(
                        args.contextual_nn_dir, 
                        args.static_nn_dir,
                        llm,
                        vision_encoder,
                        nlp,
                        sample_size=args.sample_size,
                        seed=args.seed
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
        output_path = output_dir / 'pos_tags_combined.pdf'
        print(f"\nCreating combined visualization...")
        create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders, top_n_tags=args.top_n_tags)
        
        # Create average plot
        avg_data = compute_average_data(all_data_dict)
        if avg_data:
            avg_output_path = output_dir / 'pos_tags_average.pdf'
            print(f"\nCreating average visualization...")
            create_average_plot(avg_data, avg_output_path, top_n_tags=args.top_n_tags)
        
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
            output_path = output_dir / f'pos_tags_{llm}_{vision_encoder}.pdf'
        else:
            output_path = Path(args.output)
        
        # Load data
        print(f"Loading nearest neighbor results for {llm} + {vision_encoder}...")
        print(f"  Contextual NNs: {args.contextual_nn_dir}")
        print(f"  Static NNs: {args.static_nn_dir}")
        print(f"  Sample size: {args.sample_size} per layer")
        
        data = load_pos_tag_counts(
            args.contextual_nn_dir, 
            args.static_nn_dir,
            llm,
            vision_encoder,
            nlp,
            sample_size=args.sample_size,
            seed=args.seed
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
        print("Creating POS tags visualization...")
        print("="*60)
        
        create_visualization(data, output_path, llm, vision_encoder, top_n_tags=args.top_n_tags)
        
        print("\n" + "="*60)
        print("Visualization created successfully!")
        print("="*60)


if __name__ == '__main__':
    main()

