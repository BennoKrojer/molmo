#!/usr/bin/env python3
"""
Analyze the average concreteness of nearest neighbor words across layers.

Uses the Brysbaert et al. (2014) concreteness ratings dataset to compute
the average concreteness of top-5 NN words for each vision token at each VISUAL layer.

NOTE: We group by VISUAL LAYER (from filename), not contextual layer.
Each visual layer file contains all neighbors for vision tokens from that layer.
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
import pandas as pd


def extract_full_word_from_token(sentence: str, token: str) -> str:
    """
    Extract the full word containing the token from the sentence.
    If the token is a subword within a larger word (e.g., "ing" in "rendering"),
    expand to return the entire containing word. Case-insensitive match.
    If not found, fall back to returning the token itself.
    
    Copied from llm_judge/run_single_model_with_viz_contextual.py
    """
    if not sentence:
        return token.strip() if token else ""

    # Strip whitespace from token (may have trailing/leading spaces from tokenizer)
    token = token.strip() if token else ""
    if not token:
        return ""

    low_sent = sentence.lower()
    low_tok = token.lower()
    if not low_tok:
        return token

    idx = low_sent.find(low_tok)
    if idx == -1:
        # No occurrence; return the token as-is
        return token

    # Expand to full word boundaries around the found occurrence
    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == '_'

    start = idx
    end = idx + len(low_tok)  # Use length of lowercase token for consistency

    # If token already contains whitespace, do not expand across words
    token_has_space = any(ch.isspace() for ch in token)

    if not token_has_space:
        # Check if token is embedded in a word (has word chars before or after)
        left_is_word = start > 0 and is_word_char(sentence[start - 1])
        right_is_word = end < len(sentence) and is_word_char(sentence[end])
        
        if left_is_word or right_is_word:
            # Expand to full word boundaries
            exp_start = start
            exp_end = end
            # Expand left to word boundary
            while exp_start > 0 and is_word_char(sentence[exp_start - 1]):
                exp_start -= 1
            # Expand right to word boundary
            while exp_end < len(sentence) and is_word_char(sentence[exp_end]):
                exp_end += 1
            
            expanded = sentence[exp_start:exp_end]
            # Only use expansion if it doesn't contain whitespace
            if not any(ch.isspace() for ch in expanded):
                return expanded

    # Default: return only the matched token range (no expansion)
    # Use original sentence slice to preserve case
    return sentence[start:end]


def load_concreteness_ratings(ratings_path):
    """
    Load concreteness ratings from the Brysbaert et al. Excel file.
    
    Returns:
        dict: {word: concreteness_rating}
    """
    df = pd.read_excel(ratings_path)
    
    # Create lookup dict (lowercase for matching)
    ratings = {}
    for _, row in df.iterrows():
        word = str(row['Word']).lower().strip()
        rating = row['Conc.M']
        ratings[word] = rating
    
    print(f"Loaded {len(ratings)} concreteness ratings")
    print(f"  Rating range: {min(ratings.values()):.2f} - {max(ratings.values()):.2f}")
    print(f"  Mean rating: {np.mean(list(ratings.values())):.2f}")
    
    return ratings


def load_concreteness_per_layer(contextual_nn_dir, static_nn_dir, llm, vision_encoder, concreteness_ratings):
    """
    Load all NN results and compute average concreteness per VISUAL layer.
    
    IMPORTANT: We group by visual layer (from filename), NOT contextual layer.
    Each file contextual_neighbors_visual{N}_allLayers.json contains all neighbors
    for vision tokens from visual layer N.
    
    Returns:
        dict: {visual_layer: {'sum_concreteness': float, 'count': int, 'matched_words': int, 'total_words': int}}
    """
    contextual_nn_dir = Path(contextual_nn_dir)
    static_nn_dir = Path(static_nn_dir) if static_nn_dir else None
    
    data = defaultdict(lambda: {'sum_concreteness': 0.0, 'count': 0, 'matched_words': 0, 'total_words': 0})
    
    # Build checkpoint name
    if llm == "qwen2-7b" and vision_encoder == "vit-l-14-336":
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_seed10_step12000-unsharded"
    else:
        checkpoint_name = f"train_mlp-only_pixmo_cap_resize_{llm}_{vision_encoder}_step12000-unsharded"
    
    print(f"\nSearching for checkpoint: {checkpoint_name}")
    
    # Load contextual nearest neighbors (allLayers format)
    contextual_dir = contextual_nn_dir / checkpoint_name
    if contextual_dir.exists():
        print(f"\nLoading contextual NN results from {contextual_dir}...")
        
        # Find all contextual neighbor files
        nn_files = sorted(contextual_dir.glob("contextual_neighbors_visual*_allLayers.json"))
        
        for nn_file in nn_files:
            # Extract visual layer from filename - THIS is the layer we care about
            match = re.search(r'contextual_neighbors_visual(\d+)_allLayers\.json$', str(nn_file))
            if not match:
                continue
            
            visual_layer = int(match.group(1))
            
            print(f"  Loading visual layer {visual_layer}...")
            with open(nn_file, 'r') as f:
                nn_results = json.load(f)
            
            results_list = nn_results.get('results', [])
            
            for image_result in tqdm(results_list, desc=f"    Processing visual layer {visual_layer}", leave=False):
                chunks = image_result.get('chunks', [])
                
                for chunk in chunks:
                    patches = chunk.get('patches', [])
                    
                    for patch in patches:
                        neighbors = patch.get('nearest_contextual_neighbors', [])
                        
                        for neighbor in neighbors:
                            token_str = neighbor.get('token_str', '')
                            caption = neighbor.get('caption', '')
                            
                            # Expand subword to full word using caption context
                            full_word = extract_full_word_from_token(caption, token_str)
                            word_clean = full_word.lower().strip()
                            
                            # Group by VISUAL LAYER (from filename), not contextual_layer
                            data[visual_layer]['total_words'] += 1
                            
                            # Look up concreteness rating
                            if word_clean in concreteness_ratings:
                                data[visual_layer]['sum_concreteness'] += concreteness_ratings[word_clean]
                                data[visual_layer]['matched_words'] += 1
                            data[visual_layer]['count'] += 1
            
            print(f"    ✓ Processed {len(results_list)} images, {data[visual_layer]['total_words']} words")
    else:
        print(f"  WARNING: Contextual NN directory not found: {contextual_dir}")
    
    # Load static nearest neighbors (layer 0) - only if we don't already have layer 0 from contextual
    if static_nn_dir and 0 not in data:
        static_dir = static_nn_dir / checkpoint_name
        if static_dir.exists():
            print(f"\nLoading static NN results (layer 0) from {static_dir}...")
            
            nn_file = static_dir / "nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json"
            
            if nn_file.exists():
                print(f"  Loading layer 0...")
                with open(nn_file, 'r') as f:
                    nn_results = json.load(f)
                
                validation_data = nn_results.get('splits', {}).get('validation', {})
                images = validation_data.get('images', [])
                
                for image_result in tqdm(images, desc="    Processing layer 0", leave=False):
                    chunks = image_result.get('chunks', [])
                    
                    for chunk in chunks:
                        patches = chunk.get('patches', [])
                        
                        for patch in patches:
                            neighbors = patch.get('nearest_neighbors', [])
                            
                            for neighbor in neighbors:
                                token_str = neighbor.get('token', '')
                                # Static NN doesn't have caption context, so just use token as-is
                                word_clean = token_str.lower().strip()
                                
                                data[0]['total_words'] += 1
                                
                                if word_clean in concreteness_ratings:
                                    data[0]['sum_concreteness'] += concreteness_ratings[word_clean]
                                    data[0]['matched_words'] += 1
                                data[0]['count'] += 1
                
                print(f"    ✓ Layer 0: {data[0]['count']} words processed")
            else:
                print(f"  WARNING: Layer 0 file not found: {nn_file}")
        else:
            print(f"  WARNING: Static NN directory not found: {static_dir}")
    
    return dict(data)


def plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=False):
    """Plot a single subplot with concreteness line."""
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
    
    if llm == 'average' and vision_encoder == 'average':
        title_text = 'Average'
    elif encoder_label:
        title_text = f'{llm_label} + {encoder_label}'
    else:
        title_text = llm_label
    
    # Sort layers
    layers = sorted(data.keys())
    
    # Calculate average concreteness per layer
    avg_concreteness = []
    match_rates = []
    
    for layer in layers:
        if data[layer]['matched_words'] > 0:
            avg = data[layer]['sum_concreteness'] / data[layer]['matched_words']
            avg_concreteness.append(avg)
        else:
            avg_concreteness.append(np.nan)  # Use NaN so matplotlib skips plotting this point
        
        if data[layer]['total_words'] > 0:
            match_rate = data[layer]['matched_words'] / data[layer]['total_words'] * 100
            match_rates.append(match_rate)
        else:
            match_rates.append(0)
    
    # Plot concreteness line
    color = '#E91E63'  # Pink
    line1 = ax.plot(layers, avg_concreteness, marker='o', label='Avg Concreteness', 
                    color=color, linewidth=2.5, markersize=8)
    
    # Add horizontal line for overall mean (3.04 based on the dataset)
    ax.axhline(y=3.04, color='gray', linestyle='--', alpha=0.5, label='Dataset mean')
    
    if not is_combined:
        ax.set_xlabel('Visual Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Concreteness (1-5)', fontsize=12, fontweight='bold')
    
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=8)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=11)
    ax.set_ylim(1, 5)
    ax.tick_params(axis='y', labelsize=11)
    
    if show_legend:
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    
    return line1[0], avg_concreteness, match_rates


def create_visualization(data, output_path, llm, vision_encoder):
    """Create line plot showing concreteness across layers."""
    if not data:
        print("No data found to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    line, avg_conc, match_rates = plot_single_subplot(ax, data, llm, vision_encoder, show_legend=True, is_combined=False)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
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
    
    print(f"\n{llm_label} + {encoder_label}:")
    print("="*100)
    print(f"{'Visual Layer':<15}{'Avg Concreteness':<20}{'Matched Words':<18}{'Total Words':<15}{'Match Rate %':<15}")
    print("-" * 100)
    for i, layer in enumerate(layers):
        avg = data[layer]['sum_concreteness'] / data[layer]['matched_words'] if data[layer]['matched_words'] > 0 else 0
        match_rate = data[layer]['matched_words'] / data[layer]['total_words'] * 100 if data[layer]['total_words'] > 0 else 0
        print(f"{layer:<15}{avg:<20.3f}{data[layer]['matched_words']:<18}{data[layer]['total_words']:<15}{match_rate:<15.1f}")
    print()


def compute_average_data(all_data_dict):
    """Compute average across all model combinations."""
    all_layers = set()
    for data in all_data_dict.values():
        all_layers.update(data.keys())
    all_layers = sorted(all_layers)
    
    if not all_layers:
        return {}
    
    avg_data = defaultdict(lambda: {'sum_concreteness': 0.0, 'count': 0, 'matched_words': 0, 'total_words': 0})
    count = 0
    
    for data in all_data_dict.values():
        if data:
            count += 1
            for layer in all_layers:
                if layer in data:
                    avg_data[layer]['sum_concreteness'] += data[layer]['sum_concreteness']
                    avg_data[layer]['matched_words'] += data[layer]['matched_words']
                    avg_data[layer]['total_words'] += data[layer]['total_words']
                    avg_data[layer]['count'] += data[layer]['count']
    
    return dict(avg_data)


def create_average_plot(avg_data, output_path):
    """Create a single plot showing the average across all combinations."""
    if not avg_data:
        print("No average data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plot_single_subplot(ax, avg_data, 'average', 'average', show_legend=True, is_combined=False)
    
    ax.set_title('Average Concreteness Across Visual Layers\n(averaged across all model combinations)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAverage plot saved to: {output_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Average plot also saved as PNG: {png_path}")
    
    plt.close()


def create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders):
    """Create a 3x3 subplot grid with all 9 combinations."""
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes_flat = axes.flatten()
    
    plot_idx = 0
    handles = None
    labels = None
    
    for llm_idx, llm in enumerate(all_llms):
        for encoder_idx, vision_encoder in enumerate(all_vision_encoders):
            ax = axes_flat[plot_idx]
            
            data = all_data_dict.get((llm, vision_encoder), {})
            
            if data:
                line, _, _ = plot_single_subplot(ax, data, llm, vision_encoder, show_legend=False, is_combined=True)
                
                if plot_idx == 0:
                    handles = [line]
                    labels = ['Avg Concreteness']
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plot_idx += 1
    
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=14, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Combined plot also saved as PNG: {png_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze concreteness of nearest neighbor words across visual layers'
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
        '--ratings-file',
        type=str,
        default=None,
        help='Path to concreteness ratings Excel file'
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
    script_dir = Path(__file__).resolve().parent
    
    # Set default paths
    if args.contextual_nn_dir is None:
        args.contextual_nn_dir = str(repo_root / 'analysis_results' / 'contextual_nearest_neighbors')
    if args.static_nn_dir is None:
        args.static_nn_dir = str(repo_root / 'analysis_results' / 'nearest_neighbors')
    if args.ratings_file is None:
        args.ratings_file = str(script_dir / 'Concreteness_ratings_Brysbaert_et_al.xlsx')
    
    # Load concreteness ratings
    print("Loading concreteness ratings...")
    concreteness_ratings = load_concreteness_ratings(args.ratings_file)
    
    # Define all model combinations
    all_llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    all_vision_encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    # Set up output directory
    output_dir = repo_root / 'analysis_results' / 'layer_evolution'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.combined:
        print(f"\n{'='*80}")
        print("Creating combined 3x3 plot for all 9 model combinations")
        print(f"{'='*80}")
        
        all_data_dict = {}
        
        for llm in all_llms:
            for vision_encoder in all_vision_encoders:
                key = (llm, vision_encoder)
                print(f"\nLoading data for {llm} + {vision_encoder}...")
                data = load_concreteness_per_layer(
                    args.contextual_nn_dir, 
                    args.static_nn_dir,
                    llm,
                    vision_encoder,
                    concreteness_ratings
                )
                if data:
                    all_data_dict[key] = data
                    layers = sorted(data.keys())
                    print(f"  ✓ Found {len(layers)} visual layers: {layers}")
                else:
                    print(f"  ⚠ No data found")
        
        # Create combined plot
        output_path = output_dir / 'concreteness_combined.pdf'
        print(f"\nCreating combined visualization...")
        create_combined_plot(all_data_dict, output_path, all_llms, all_vision_encoders)
        
        # Create average plot
        avg_data = compute_average_data(all_data_dict)
        if avg_data:
            avg_output_path = output_dir / 'concreteness_average.pdf'
            print(f"\nCreating average visualization...")
            create_average_plot(avg_data, avg_output_path)
        
        print(f"\n{'='*80}")
        print(f"Combined plot created successfully!")
        print(f"{'='*80}")
    else:
        # Process single combination
        llm = args.llm
        vision_encoder = args.vision_encoder
        
        if args.output is None:
            output_path = output_dir / f'concreteness_{llm}_{vision_encoder}.pdf'
        else:
            output_path = Path(args.output)
        
        print(f"\nLoading nearest neighbor results for {llm} + {vision_encoder}...")
        print(f"  Contextual NNs: {args.contextual_nn_dir}")
        print(f"  Static NNs: {args.static_nn_dir}")
        
        data = load_concreteness_per_layer(
            args.contextual_nn_dir, 
            args.static_nn_dir,
            llm,
            vision_encoder,
            concreteness_ratings
        )
        
        if not data:
            print(f"ERROR: No results found for {llm} + {vision_encoder}!")
            return
        
        layers = sorted(data.keys())
        print(f"\nFound data for {len(layers)} visual layers: {layers}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Creating concreteness visualization...")
        print("="*60)
        
        create_visualization(data, output_path, llm, vision_encoder)
        
        print("\n" + "="*60)
        print("Visualization created successfully!")
        print("="*60)


if __name__ == '__main__':
    main()
