#!/usr/bin/env python3
"""
Create a heatmap visualization of LLM judge interpretability results for nearest neighbors.
Rows: LLMs, Columns: Vision encoders, Cells: Interpretability percentage
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_results(results_dir):
    """Load all results JSON files and extract interpretability percentages.
    Only loads layer 0 results to match the heatmap title."""
    results_dir = Path(results_dir)
    
    # Dictionary to store: {llm: {vision_encoder: accuracy}}
    data = defaultdict(dict)
    
    # Find all results JSON files, but only for layer 0
    # Exclude ablations directory to avoid conflicts (ablations are variants, not main models)
    for results_file in results_dir.glob("**/results_*.json"):
        path_str = str(results_file)
        # Skip ablations directory
        if '/ablations/' in path_str:
            continue
        # Filter to only layer 0: check if path contains "layer0" (not "layer1", "layer2", etc.)
        if 'layer0' in path_str.lower() or '/layer0_' in path_str or '_layer0/' in path_str:
            with open(results_file, 'r') as f:
                spresults = json.load(f)
            
            llm = spresults.get('llm')
            vision_encoder = spresults.get('vision_encoder')
            accuracy = spresults.get('accuracy', 0.0)
            
            if llm and vision_encoder:
                data[llm][vision_encoder] = accuracy
    
    return data


def load_category_results(results_dir):
    """Load all results JSON files and extract category-specific percentages (concrete, abstract, global).
    Only loads layer 0 results to match the heatmap title."""
    results_dir = Path(results_dir)
    
    # Dictionary to store: {category: {llm: {vision_encoder: percentage}}}
    category_data = {
        'concrete': defaultdict(dict),
        'abstract': defaultdict(dict),
        'global': defaultdict(dict)
    }
    
    # Find all results JSON files, but only for layer 0
    # Exclude ablations directory to avoid conflicts (ablations are variants, not main models)
    for results_file in results_dir.glob("**/results_*.json"):
        path_str = str(results_file)
        # Skip ablations directory
        if '/ablations/' in path_str:
            continue
        # Filter to only layer 0: check if path contains "layer0" (not "layer1", "layer2", etc.)
        if 'layer0' not in path_str.lower() and '/layer0_' not in path_str and '_layer0/' not in path_str:
            continue
        
        with open(results_file, 'r') as f:
            spresults = json.load(f)
        
        llm = spresults.get('llm')
        vision_encoder = spresults.get('vision_encoder')
        responses = spresults.get('responses', {})
        
        if not llm or not vision_encoder:
            continue
        
        # Count totals for each category
        total_patches = 0
        concrete_count = 0
        abstract_count = 0
        global_count = 0
        
        # Iterate through all responses
        for image_path, patch_list in responses.items():
            if not isinstance(patch_list, list):
                continue
            for patch_response in patch_list:
                if not isinstance(patch_response, dict):
                    continue
                
                gpt_response = patch_response.get('gpt_response', {})
                if not isinstance(gpt_response, dict):
                    continue
                
                total_patches += 1
                
                # Check each category
                concrete_words = gpt_response.get('concrete_words', [])
                abstract_words = gpt_response.get('abstract_words', [])
                global_words = gpt_response.get('global_words', [])
                
                if concrete_words and len(concrete_words) > 0:
                    concrete_count += 1
                if abstract_words and len(abstract_words) > 0:
                    abstract_count += 1
                if global_words and len(global_words) > 0:
                    global_count += 1
        
        # Calculate percentages
        if total_patches > 0:
            category_data['concrete'][llm][vision_encoder] = (concrete_count * 100.0) / total_patches
            category_data['abstract'][llm][vision_encoder] = (abstract_count * 100.0) / total_patches
            category_data['global'][llm][vision_encoder] = (global_count * 100.0) / total_patches
        else:
            category_data['concrete'][llm][vision_encoder] = 0.0
            category_data['abstract'][llm][vision_encoder] = 0.0
            category_data['global'][llm][vision_encoder] = 0.0
    
    return category_data


def create_heatmap(data, output_path, title='Layer 0 Nearest Neighbors Interpretability (MLLM Judge)', 
                   cbar_label='Interpretability %', print_table=True):
    """Create and save a heatmap visualization."""
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
    
    # Filter to only include LLMs/encoders that exist in the data
    llms = [llm for llm in llm_order if llm in data]
    vision_encoders = [enc for enc in encoder_order if any(enc in data[llm] for llm in llms)]
    
    # Create matrix
    matrix = np.zeros((len(llms), len(vision_encoders)))
    
    for i, llm in enumerate(llms):
        for j, encoder in enumerate(vision_encoders):
            matrix[i, j] = data[llm].get(encoder, 0.0)
    
    # Create display labels with exact formatting
    llm_labels = [llm_display_names.get(llm, llm) for llm in llms]
    encoder_labels = [encoder_display_names.get(enc, enc) for enc in vision_encoders]
    
    # Create heatmap with better sizing
    # Adjust figure size based on number of rows/columns
    fig_height = max(6, len(llms) * 0.8)
    fig_width = max(10, len(vision_encoders) * 2.5)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Use a colormap that goes from low (red) to high (green) for interpretability
    # RdYlGn is red-yellow-green, but we want it reversed so green is high
    sns.heatmap(
        matrix,
        xticklabels=encoder_labels,
        yticklabels=llm_labels,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=25,  # Center the colormap around expected range
        vmin=0,
        vmax=100,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='gray',
        square=False,
        annot_kws={'size': 10}
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Vision Encoder', fontsize=12, fontweight='bold')
    plt.ylabel('LLM', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    
    # Also save as PNG with lower DPI for quick viewing
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap also saved as PNG: {png_path}")
    
    plt.close()
    
    # Print the matrix values with display names
    if print_table:
        print(f"\n{title}:")
        
        # Header with display names
        print("\n" + " " * 20 + "\t".join(f"{enc:20s}" for enc in encoder_labels))
        for i, llm in enumerate(llms):
            row = f"{llm_labels[i]:20s}\t"
            for encoder in vision_encoders:
                acc = data[llm].get(encoder, 0.0)
                row += f"{acc:6.1f}\t"
            print(row)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create heatmap from LLM judge results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='analysis_results/llm_judge_nearest_neighbors',
        help='Directory containing LLM judge results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_results/llm_judge_nearest_neighbors/heatmap_interpretability.pdf',
        help='Output path for heatmap (PDF or PNG)'
    )
    
    args = parser.parse_args()
    
    # Load overall results
    print(f"Loading results from: {args.results_dir}")
    data = load_results(args.results_dir)
    
    if not data:
        print("ERROR: No results found!")
        return
    
    # Print summary
    print("\nFound results for:")
    for llm in sorted(data.keys()):
        print(f"  {llm}: {len(data[llm])} vision encoders")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create overall interpretability heatmap
    print("\n" + "="*60)
    print("Creating overall interpretability heatmap...")
    print("="*60)
    create_heatmap(data, output_path, 
                   title='Layer 0 Nearest Neighbors Interpretability (MLLM Judge)',
                   cbar_label='Interpretability %')
    
    # Load category-specific results
    print("\n" + "="*60)
    print("Loading category-specific results...")
    print("="*60)
    category_data = load_category_results(args.results_dir)
    
    # Create category-specific heatmaps
    category_info = {
        'concrete': {
            'title': 'Layer 0 Nearest Neighbors - Concrete Interpretability (MLLM Judge)',
            'cbar_label': 'Concrete Interpretability %',
            'filename_suffix': 'concrete'
        },
        'abstract': {
            'title': 'Layer 0 Nearest Neighbors - Abstract Interpretability (MLLM Judge)',
            'cbar_label': 'Abstract Interpretability %',
            'filename_suffix': 'abstract'
        },
        'global': {
            'title': 'Layer 0 Nearest Neighbors - Global Interpretability (MLLM Judge)',
            'cbar_label': 'Global Interpretability %',
            'filename_suffix': 'global'
        }
    }
    
    for category, info in category_info.items():
        print(f"\n" + "="*60)
        print(f"Creating {category} interpretability heatmap...")
        print("="*60)
        category_output = output_path.parent / f"heatmap_interpretability_{info['filename_suffix']}.pdf"
        create_heatmap(category_data[category], category_output,
                      title=info['title'],
                      cbar_label=info['cbar_label'])
    
    print("\n" + "="*60)
    print("All heatmaps created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
