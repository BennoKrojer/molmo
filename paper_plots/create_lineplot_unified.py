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

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "interpretability"


def load_nn_results(results_dir):
    """Load nearest neighbors results."""
    results_dir = Path(results_dir)
    data = defaultdict(lambda: defaultdict(dict))

    for results_file in results_dir.glob("**/results_*.json"):
        # Skip ablations subdirectory - those are separate experiments
        if '/ablations/' in str(results_file):
            continue
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
        # Skip ablations subdirectory - those are separate experiments
        if '/ablations/' in str(results_file):
            continue
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


def get_expected_layers(llm):
    """
    Get the expected layers for a given LLM.
    
    We use: [0, 1, 2, 4, 8, 16, 24, N-2, N-1]
    where N = 32 for OLMo/Llama, N = 28 for Qwen
    """
    if llm in ['olmo-7b', 'llama3-8b']:
        # N = 32, so N-2 = 30, N-1 = 31
        return [0, 1, 2, 4, 8, 16, 24, 30, 31]
    elif llm == 'qwen2-7b':
        # N = 28, so N-2 = 26, N-1 = 27
        return [0, 1, 2, 4, 8, 16, 24, 26, 27]
    else:
        # Default fallback
        return [0, 1, 2, 4, 8, 16, 24, 30, 31]


def filter_to_expected_layers(data):
    """
    Filter data to only include the expected layers for each model.
    
    Expected layers per LLM:
    - OLMo/Llama: [0, 1, 2, 4, 8, 16, 24, 30, 31]
    - Qwen: [0, 1, 2, 4, 8, 16, 24, 26, 27]
    """
    filtered_data = {}
    for (llm, encoder), layer_data in data.items():
        expected = set(get_expected_layers(llm))
        filtered_layer_data = {k: v for k, v in layer_data.items() if k in expected}
        filtered_data[(llm, encoder)] = filtered_layer_data
    
    return filtered_data


# Keep old function for backwards compatibility
def filter_last_layer(data):
    """Deprecated - use filter_to_expected_layers instead."""
    return filter_to_expected_layers(data)


def load_contextual_results(results_dir, nn_results_dir=None):
    """Load contextual NN results."""
    results_dir = Path(results_dir)
    data = defaultdict(lambda: defaultdict(dict))

    for results_file in results_dir.glob("**/results_*.json"):
        # Skip ablations subdirectory - those are separate experiments
        if '/ablations/' in str(results_file):
            continue
        with open(results_file, 'r') as f:
            spresults = json.load(f)
        
        path_str = str(results_file)
        
        # Extract layer from path
        layer_str = None
        match = re.search(r'_contextual(\d+)_', path_str)
        if match:
            layer_str = f"contextual{match.group(1)}"
        else:
            # Try alternative patterns - maybe layer 32 is at end of filename
            match = re.search(r'_contextual(\d+)\.json', path_str)
            if match:
                layer_str = f"contextual{match.group(1)}"
            # Also check if it's just contextual32 without underscore
            match = re.search(r'contextual(\d+)', path_str)
            if match and not layer_str:
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
    
    # X-axis ticks: union of all expected layers across all models
    # OLMo/Llama: [0, 1, 2, 4, 8, 16, 24, 30, 31]
    # Qwen: [0, 1, 2, 4, 8, 16, 24, 26, 27]
    # Union: [0, 1, 2, 4, 8, 16, 24, 26, 27, 30, 31]
    all_expected_layers = sorted(set(
        get_expected_layers('olmo-7b') + get_expected_layers('qwen2-7b')
    ))
    print(f"  X-axis ticks (union of expected layers): {all_expected_layers}")
    
    # Create figure with 3 subplots side by side
    # Make it much wider to give plenty of room for x-axis labels
    fig, axes = plt.subplots(1, 3, figsize=(36, 8))
    sns.set_style("whitegrid")
    
    # Titles and data for each subplot
    subplot_configs = [
        {
            'ax': axes[0],
            'data': nn_data,
            'title': 'Input Embedding Matrix',
            'xlabel': 'Layer'
        },
        {
            'ax': axes[1],
            'data': logitlens_data,
            'title': 'Output Embedding Matrix (Logitlens)',
            'xlabel': 'Layer'
        },
        {
            'ax': axes[2],
            'data': contextual_data,
            'title': 'LN-Lens',
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
        
        # Use expected layers for consistent x-axis across all subplots
        all_layers = all_expected_layers
        
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
        ax.set_ylabel('Interpretable Tokens %\n(via automated judge)', fontsize=14, fontweight='bold')
        ax.set_title(config['title'], fontsize=18, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Set x-axis to show only expected layers (where we have data)
        ax.set_xlim(min(all_expected_layers) - 0.5, max(all_expected_layers) + 0.5)
        # Show all expected layers as ticks
        ax.set_xticks(all_expected_layers)
        ax.set_xticklabels([str(t) for t in all_expected_layers])
        ax.tick_params(axis='both', labelsize=12)
    
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
    print(f"✓ Saved: {output_path.name}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path.name}")
    
    plt.close()


def create_baselines_lineplot(nn_data, logitlens_data, output_path):
    """Create a figure with 2 subplots stacked vertically for baselines (NN and LogitLens)."""

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

    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoder_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']

    llm_base_colors = {
        'olmo-7b': plt.cm.Blues,
        'llama3-8b': plt.cm.Greens,
        'qwen2-7b': plt.cm.Reds
    }
    encoder_shade_indices = [0.5, 0.7, 0.9]

    encoder_markers = {
        'vit-l-14-336': '*',
        'siglip': 'o',
        'dinov2-large-336': '^'
    }

    encoder_marker_facecolors = {
        'vit-l-14-336': None,
        'siglip': 'none',
        'dinov2-large-336': None
    }

    color_map = {}
    for llm in llm_order:
        base_cmap = llm_base_colors[llm]
        for enc_idx, encoder in enumerate(encoder_order):
            color_map[(llm, encoder)] = base_cmap(encoder_shade_indices[enc_idx])

    # X-axis ticks: union of all expected layers
    all_expected_layers = sorted(set(
        get_expected_layers('olmo-7b') + get_expected_layers('qwen2-7b')
    ))

    # Vertical layout: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    sns.set_style("whitegrid")

    subplot_configs = [
        {
            'ax': axes[0],
            'data': nn_data,
            'title': 'Input Embedding Matrix',
            'xlabel': 'Layer'
        },
        {
            'ax': axes[1],
            'data': logitlens_data,
            'title': 'Output Embedding Matrix (Logitlens)',
            'xlabel': 'Layer'
        }
    ]

    handles_dict = {}

    for config in subplot_configs:
        ax = config['ax']
        data = config['data']

        if not data:
            continue

        all_layers = all_expected_layers

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

                llm_label = llm_display_names.get(llm, llm)
                encoder_label = encoder_display_names.get(encoder, encoder)
                label = f"{llm_label} + {encoder_label}"

                marker = encoder_markers.get(encoder, 'o')
                marker_facecolor = encoder_marker_facecolors.get(encoder, None)

                if marker_facecolor is not None:
                    line, = ax.plot(layers, values, marker=marker,
                                   color=color_map[key], markerfacecolor=marker_facecolor,
                                   markeredgewidth=2, linewidth=2.5, markersize=10)
                else:
                    line, = ax.plot(layers, values, marker=marker,
                                   color=color_map[key], linewidth=2.5, markersize=10)

                if label not in handles_dict:
                    handles_dict[label] = line

        ax.set_xlabel(config['xlabel'], fontsize=16, fontweight='bold')
        ax.set_ylabel('Interpretable Tokens %\n(via automated judge)', fontsize=14, fontweight='bold')
        ax.set_title(config['title'], fontsize=18, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # Set x-axis to show only expected layers
        ax.set_xlim(min(all_expected_layers) - 0.5, max(all_expected_layers) + 0.5)
        ax.set_xticks(all_expected_layers)
        ax.set_xticklabels([str(t) for t in all_expected_layers])
        ax.tick_params(axis='both', labelsize=12)

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

    # Legend at the bottom for vertical layout
    fig.legend(ordered_handles, ordered_labels,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.02),
              ncol=3,
              fontsize=13,
              framealpha=0.9,
              columnspacing=2.0,
              handlelength=2.5,
              handletextpad=1.0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.25)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")

    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path.name}")

    plt.close()


def create_lnlens_lineplot(contextual_data, output_path):
    """Create a single figure for LN-Lens (our method)."""
    
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
    
    llm_order = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoder_order = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    llm_base_colors = {
        'olmo-7b': plt.cm.Blues,
        'llama3-8b': plt.cm.Greens,
        'qwen2-7b': plt.cm.Reds
    }
    encoder_shade_indices = [0.5, 0.7, 0.9]
    
    encoder_markers = {
        'vit-l-14-336': '*',
        'siglip': 'o',
        'dinov2-large-336': '^'
    }
    
    encoder_marker_facecolors = {
        'vit-l-14-336': None,
        'siglip': 'none',
        'dinov2-large-336': None
    }
    
    color_map = {}
    for llm in llm_order:
        base_cmap = llm_base_colors[llm]
        for enc_idx, encoder in enumerate(encoder_order):
            color_map[(llm, encoder)] = base_cmap(encoder_shade_indices[enc_idx])
    
    # X-axis ticks: union of all expected layers
    all_expected_layers = sorted(set(
        get_expected_layers('olmo-7b') + get_expected_layers('qwen2-7b')
    ))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.set_style("whitegrid")
    
    handles_dict = {}
    
    if contextual_data:
        for llm in llm_order:
            for encoder in encoder_order:
                key = (llm, encoder)
                if key not in contextual_data:
                    continue
                
                layer_data = contextual_data[key]
                layers = sorted(layer_data.keys())
                values = [layer_data[l] for l in layers]
                
                if len(layers) == 0:
                    continue
                
                llm_label = llm_display_names.get(llm, llm)
                encoder_label = encoder_display_names.get(encoder, encoder)
                label = f"{llm_label} + {encoder_label}"
                
                marker = encoder_markers.get(encoder, 'o')
                marker_facecolor = encoder_marker_facecolors.get(encoder, None)
                
                if marker_facecolor is not None:
                    line, = ax.plot(layers, values, marker=marker, 
                                   color=color_map[key], markerfacecolor=marker_facecolor,
                                   markeredgewidth=2, linewidth=2.5, markersize=10)
                else:
                    line, = ax.plot(layers, values, marker=marker,
                                   color=color_map[key], linewidth=2.5, markersize=10)
                
                if label not in handles_dict:
                    handles_dict[label] = line
    
    ax.set_xlabel('Layer', fontsize=16, fontweight='bold')
    ax.set_ylabel('Interpretable Tokens %\n(via automated judge)', fontsize=14, fontweight='bold')
    ax.set_title('LN-Lens', fontsize=18, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Set x-axis to show only expected layers
    ax.set_xlim(min(all_expected_layers) - 0.5, max(all_expected_layers) + 0.5)
    ax.set_xticks(all_expected_layers)
    ax.set_xticklabels([str(t) for t in all_expected_layers])
    ax.tick_params(axis='both', labelsize=12)
    
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
    
    ax.legend(ordered_handles, ordered_labels, 
              loc='lower right',
              ncol=1, 
              fontsize=12, 
              framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path.name}")
    
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
        default=None,
        help='Output path for unified line plot (default: paper_figures_output/unified_interpretability/fig1_unified_interpretability.pdf)'
    )
    
    args = parser.parse_args()
    
    # Load results from all three sources
    print("Loading nearest neighbors results...")
    nn_data = load_nn_results(args.nn_results_dir)
    print(f"  Found {len(nn_data)} model combinations")
    if nn_data:
        nn_all_layers = set()
        for layer_data in nn_data.values():
            nn_all_layers.update(layer_data.keys())
        print(f"  Layers found (before filter): {sorted(nn_all_layers)}")
    
    print("\nLoading logit lens results...")
    logitlens_data = load_logitlens_results(args.logitlens_results_dir)
    print(f"  Found {len(logitlens_data)} model combinations")
    if logitlens_data:
        ll_all_layers = set()
        for layer_data in logitlens_data.values():
            ll_all_layers.update(layer_data.keys())
        print(f"  Layers found (before filter): {sorted(ll_all_layers)}")
    
    print("\nLoading contextual NN results...")
    contextual_data = load_contextual_results(args.contextual_results_dir, args.nn_layer0_dir)
    print(f"  Found {len(contextual_data)} model combinations")
    if contextual_data:
        ctx_all_layers = set()
        for layer_data in contextual_data.values():
            ctx_all_layers.update(layer_data.keys())
        print(f"  Layers found (before filter): {sorted(ctx_all_layers)}")
    
    # Filter out last layer (32 for OLMo/Llama, 28 for Qwen) for consistency
    print("\nFiltering out last layer for consistency...")
    print("  (Layer 32 for OLMo/Llama, Layer 28 for Qwen)")
    nn_data = filter_last_layer(nn_data)
    logitlens_data = filter_last_layer(logitlens_data)
    contextual_data = filter_last_layer(contextual_data)
    
    # Print layers after filtering
    if nn_data:
        nn_all_layers = set()
        for layer_data in nn_data.values():
            nn_all_layers.update(layer_data.keys())
        print(f"  NN layers after filter: {sorted(nn_all_layers)}")
    if logitlens_data:
        ll_all_layers = set()
        for layer_data in logitlens_data.values():
            ll_all_layers.update(layer_data.keys())
        print(f"  LogitLens layers after filter: {sorted(ll_all_layers)}")
    if contextual_data:
        ctx_all_layers = set()
        for layer_data in contextual_data.values():
            ctx_all_layers.update(layer_data.keys())
        print(f"  Contextual NN layers after filter: {sorted(ctx_all_layers)}")
    
    if not nn_data and not logitlens_data and not contextual_data:
        print("ERROR: No results found!")
        return
    
    # Create output directory
    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "fig1_unified_interpretability.pdf"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create unified plot (all 3 subplots)
    print("\n" + "="*50)
    print("Creating unified line plot (all 3)...")
    create_unified_lineplot(nn_data, logitlens_data, contextual_data, output_path)
    
    # Create baselines plot (NN + LogitLens)
    print("\n" + "="*50)
    print("Creating baselines line plot (Input Embedding + Logitlens)...")
    baselines_path = output_path.parent / "fig_baselines.pdf"
    create_baselines_lineplot(nn_data, logitlens_data, baselines_path)
    
    # Create LN-Lens plot (our method)
    print("\n" + "="*50)
    print("Creating LN-Lens line plot (our method)...")
    lnlens_path = output_path.parent / "fig_lnlens.pdf"
    create_lnlens_lineplot(contextual_data, lnlens_path)
    
    print(f"\n✓ All plots saved to {output_path.parent}")


if __name__ == '__main__':
    main()

