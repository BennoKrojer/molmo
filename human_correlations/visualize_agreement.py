#!/usr/bin/env python3
"""
Visualize human vs LLM judge agreement on interpretability.
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import from main script
from compute_correlations import (
    load_human_results,
    load_llm_judge_results,
    load_data_json,
    match_instances
)


def visualize_examples(matched_data, output_dir, num_examples=50, sample_type='all'):
    """
    Visualize examples with patch bounding boxes and judgements.
    
    Args:
        matched_data: List of matched instances
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
        sample_type: 'all', 'agree', 'disagree', 'human_yes_llm_no', 'human_no_llm_yes'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter examples based on sample_type
    filtered_data = []
    for item in matched_data:
        instance_id = item['instance_id']
        model_key = item['model_key']
        human_judgements = item['human_judgements']
        llm_judgement = item['llm_judgement']
        candidates = set(item['candidates'])
        
        # LLM interpretability
        llm_interpretable = llm_judgement['is_interpretable']
        
        # Human interpretability
        human_interpretable = False
        for human_judgement in human_judgements:
            selected_words = set(human_judgement['selected_words'].keys())
            if selected_words & candidates:
                human_interpretable = True
                break
        
        # Filter based on sample_type
        include = False
        if sample_type == 'all':
            include = True
        elif sample_type == 'agree':
            include = (llm_interpretable == human_interpretable)
        elif sample_type == 'disagree':
            include = (llm_interpretable != human_interpretable)
        elif sample_type == 'human_yes_llm_no':
            include = (human_interpretable and not llm_interpretable)
        elif sample_type == 'human_no_llm_yes':
            include = (not human_interpretable and llm_interpretable)
        
        if include:
            filtered_data.append({
                **item,
                'llm_interpretable': llm_interpretable,
                'human_interpretable': human_interpretable
            })
    
    print(f"\nFiltered {len(filtered_data)} examples of type '{sample_type}'")
    
    # Sample examples
    if len(filtered_data) > num_examples:
        import random
        random.seed(42)
        filtered_data = random.sample(filtered_data, num_examples)
    
    # Create visualizations
    for idx, item in enumerate(filtered_data):
        try:
            visualize_single_example(item, output_path, idx)
        except Exception as e:
            print(f"Error visualizing example {idx}: {e}")
    
    print(f"\nSaved {len(filtered_data)} visualizations to {output_path}")


def visualize_single_example(item, output_path, idx):
    """Visualize a single example."""
    instance_id = item['instance_id']
    model_key = item['model_key']
    llm_judgement = item['llm_judgement']
    human_judgements = item['human_judgements']
    candidates = item['candidates']
    llm_interpretable = item['llm_interpretable']
    human_interpretable = item['human_interpretable']
    
    # Get patch row and col from llm_judgement
    patch_row = llm_judgement['patch_row']
    patch_col = llm_judgement['patch_col']
    image_index = llm_judgement['image_index']
    
    # Find the LLM judge visualization image
    # Path: analysis_results/llm_judge_nearest_neighbors/llm_judge_{model_key}_layer0_gpt5_cropped/visualizations/
    # Filename: image_{image_index:04d}_patch_{patch_row}_{patch_col}_{pass/fail}.jpg
    llm_viz_dir = Path(f"analysis_results/llm_judge_nearest_neighbors/llm_judge_{model_key}_layer0_gpt5_cropped/visualizations")
    llm_status = "pass" if llm_interpretable else "fail"
    llm_viz_file = llm_viz_dir / f"image_{image_index:04d}_patch_{patch_row}_{patch_col}_{llm_status}.jpg"
    
    # Try to load the LLM judge visualization
    if llm_viz_file.exists():
        llm_img = Image.open(llm_viz_file)
        has_image = True
    else:
        has_image = False
        print(f"Warning: Could not find LLM viz image: {llm_viz_file}")
    
    # Create a figure with the information
    if has_image:
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1.2, 1]})
        ax_img.imshow(llm_img)
        ax_img.axis('off')
        ax_img.set_title(f"Image with Patch (red box)\nPatch: row={patch_row}, col={patch_col}", fontsize=10)
    else:
        fig, ax_text = plt.subplots(1, 1, figsize=(12, 8))
    
    ax_text.axis('off')
    
    # Title
    agreement = "✓ AGREE" if llm_interpretable == human_interpretable else "✗ DISAGREE"
    title = f"{agreement}\nInstance: {instance_id[:80]}...\nModel: {model_key}"
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=10, fontweight='bold',
            transform=ax.transAxes)
    
    # LLM judgement
    llm_text = f"LLM Judge: {'INTERPRETABLE' if llm_interpretable else 'NOT INTERPRETABLE'}\n"
    llm_text += f"Selected words: {', '.join(llm_judgement['all_selected_words']) if llm_judgement['all_selected_words'] else 'none'}\n"
    llm_text += f"  Concrete: {', '.join(llm_judgement['concrete_words']) if llm_judgement['concrete_words'] else 'none'}\n"
    llm_text += f"  Abstract: {', '.join(llm_judgement['abstract_words']) if llm_judgement['abstract_words'] else 'none'}\n"
    llm_text += f"  Global: {', '.join(llm_judgement['global_words']) if llm_judgement['global_words'] else 'none'}"
    
    ax.text(0.05, 0.75, llm_text, ha='left', va='top', fontsize=9,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Human judgements
    human_text = f"Human Judge: {'INTERPRETABLE' if human_interpretable else 'NOT INTERPRETABLE'}\n"
    for i, human_judgement in enumerate(human_judgements):
        selected_words = human_judgement['selected_words']
        user_id = human_judgement['user_id']
        if selected_words:
            human_text += f"  {user_id}: {', '.join([f'{w} ({r})' for w, r in selected_words.items()])}\n"
        else:
            human_text += f"  {user_id}: (no words selected)\n"
    
    ax.text(0.05, 0.45, human_text, ha='left', va='top', fontsize=9,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Candidates
    candidates_text = f"Candidates (5 nearest neighbor tokens):\n  {', '.join(candidates)}"
    ax.text(0.05, 0.20, candidates_text, ha='left', va='top', fontsize=9,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Save
    filename = f"{idx:03d}_{model_key}_{'agree' if llm_interpretable == human_interpretable else 'disagree'}.png"
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize human-LLM agreement')
    parser.add_argument('--human-data-dir', type=str,
                       default='human_correlations/interp_data',
                       help='Directory containing human judgement data')
    parser.add_argument('--llm-results-dir', type=str,
                       default='analysis_results/llm_judge_nearest_neighbors',
                       help='Directory containing LLM judge results')
    parser.add_argument('--output-dir', type=str,
                       default='human_correlations/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--layer', type=int, default=0,
                       help='Layer to analyze (default: 0)')
    parser.add_argument('--num-examples', type=int, default=50,
                       help='Number of examples to visualize')
    parser.add_argument('--sample-type', type=str, default='disagree',
                       choices=['all', 'agree', 'disagree', 'human_yes_llm_no', 'human_no_llm_yes'],
                       help='Type of examples to visualize')
    
    args = parser.parse_args()
    
    print("Loading human judgement data...")
    human_data = load_human_results(args.human_data_dir)
    print(f"Loaded {len(human_data)} unique instances with human judgements")
    
    print("Loading data.json for instance mapping...")
    data_json_path = Path(args.human_data_dir) / "data.json"
    data_mapping = load_data_json(data_json_path)
    print(f"Loaded mapping for {len(data_mapping)} instances")
    
    print("Loading LLM judge results...")
    model_keys = set()
    for instance_id in human_data.keys():
        if instance_id in data_mapping:
            model_keys.add(data_mapping[instance_id]['model_key'])
    
    llm_data_by_model = {}
    for model_key in model_keys:
        llm_data = load_llm_judge_results(args.llm_results_dir, model_key, layer=args.layer)
        if llm_data is not None:
            llm_data_by_model[model_key] = llm_data
            print(f"  Loaded LLM data for {model_key}: {len(llm_data)} instances")
    
    print("Matching instances...")
    matched_data = match_instances(human_data, llm_data_by_model, data_mapping)
    print(f"Matched {len(matched_data)} instances")
    
    print(f"\nVisualizing {args.sample_type} examples...")
    visualize_examples(matched_data, args.output_dir, args.num_examples, args.sample_type)


if __name__ == '__main__':
    main()

