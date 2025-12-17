#!/usr/bin/env python3
"""
Update paper figures with latest data from analysis_results.

Usage:
    python update_data.py                    # Extract data + regenerate plots
    python update_data.py --extract-only     # Just print the new data (for copy-paste)
    
This script reads from:
    - analysis_results/llm_judge_nearest_neighbors/
    - analysis_results/llm_judge_logitlens/
    - analysis_results/llm_judge_contextual_nn/
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# Paths (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "analysis_results"

NN_DIR = RESULTS_DIR / "llm_judge_nearest_neighbors"
LOGITLENS_DIR = RESULTS_DIR / "llm_judge_logitlens"
CONTEXTUAL_DIR = RESULTS_DIR / "llm_judge_contextual_nn"


def load_nn_results():
    """Load nearest neighbors results."""
    data = defaultdict(dict)
    for results_file in NN_DIR.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            results = json.load(f)
        path_str = str(results_file)
        
        match = re.search(r'_layer(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))
        
        match = re.search(r'llm_judge_([^_]+)_([^_]+)(?:_seed\d+)?_layer\d+', path_str)
        if not match:
            continue
        llm, enc = match.group(1), match.group(2)
        
        acc = results.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        data[f"{llm}+{enc}"][layer_num] = round(acc, 2)
    
    return {k: dict(sorted(v.items())) for k, v in sorted(data.items())}


def load_logitlens_results():
    """Load logit lens results."""
    data = defaultdict(dict)
    for results_file in LOGITLENS_DIR.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            res = json.load(f)
        path_str = str(results_file)
        
        layer_str = res.get('layer', '')
        if not layer_str:
            match = re.search(r'_layer(\d+)_', path_str)
            if match:
                layer_str = f"layer{match.group(1)}"
        
        model_str = res.get('model', '')
        if not model_str:
            match = re.search(r'llm_judge_([^_]+)_([^_]+?)(?:_(?:seed\d+_)?layer\d+|_layer\d+)', path_str)
            if match:
                model_str = f"{match.group(1)}_{match.group(2)}"
        
        parts = model_str.split('_')
        if len(parts) < 2:
            continue
        llm = parts[0]
        if 'vit-l-14-336' in model_str:
            enc = 'vit-l-14-336'
        elif 'dinov2' in model_str:
            enc = 'dinov2-large-336'
        elif 'siglip' in model_str:
            enc = 'siglip'
        else:
            enc = '_'.join(parts[1:])
        
        if not layer_str.startswith('layer'):
            continue
        layer_num = int(layer_str.replace('layer', ''))
        
        acc = res.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        data[f"{llm}+{enc}"][layer_num] = round(acc, 2)
    
    return {k: dict(sorted(v.items())) for k, v in sorted(data.items())}


def load_contextual_results():
    """Load contextual NN results (+ layer 0 from NN)."""
    data = defaultdict(dict)
    
    # Load contextual layers (1+)
    for results_file in CONTEXTUAL_DIR.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            res = json.load(f)
        path_str = str(results_file)
        
        match = re.search(r'_contextual(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))
        
        llm = res.get('llm', '')
        enc = res.get('vision_encoder', '')
        
        if not llm:
            match = re.search(r'llm_judge_([^_]+)_', path_str)
            if match:
                llm = match.group(1)
        if not enc:
            if 'vit-l-14-336' in path_str:
                enc = 'vit-l-14-336'
            elif 'dinov2' in path_str:
                enc = 'dinov2-large-336'
            elif 'siglip' in path_str:
                enc = 'siglip'
        
        if not llm or not enc:
            continue
        
        results_list = res.get('results', [])
        if results_list:
            total = len(results_list)
            interp_count = sum(1 for r in results_list if r.get('interpretable', False))
            acc = (interp_count / total * 100.0) if total > 0 else 0.0
        else:
            acc = res.get('accuracy', 0.0)
            if acc <= 1.0:
                acc *= 100.0
        
        data[f"{llm}+{enc}"][layer_num] = round(acc, 2)
    
    # Load layer 0 from NN results
    for results_file in NN_DIR.glob("**/results_*.json"):
        path_str = str(results_file)
        match = re.search(r'_layer(\d+)_', path_str)
        if not match or int(match.group(1)) != 0:
            continue
        
        match = re.search(r'llm_judge_([^/]+?)_layer\d+', path_str)
        if not match:
            continue
        
        parts = match.group(1).split('_')
        llm = parts[0]
        remaining = '_'.join(parts[1:])
        remaining = re.sub(r'_seed\d+$', '', remaining)
        
        if 'vit-l-14-336' in remaining:
            enc = 'vit-l-14-336'
        elif 'dinov2' in remaining:
            enc = 'dinov2-large-336'
        elif 'siglip' in remaining:
            enc = 'siglip'
        else:
            enc = remaining
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        acc = results.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        
        key = f"{llm}+{enc}"
        if 0 not in data.get(key, {}):
            data[key][0] = round(acc, 2)
    
    return {k: dict(sorted(v.items())) for k, v in sorted(data.items())}


def format_data_dict(data, var_name):
    """Format data dict as Python code."""
    lines = [f"{var_name} = {{"]
    for key in sorted(data.keys()):
        layer_data = data[key]
        layer_str = ", ".join(f"{k}: {v}" for k, v in sorted(layer_data.items()))
        lines.append(f'    "{key}": {{{layer_str}}},')
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Update paper figures data')
    parser.add_argument('--extract-only', action='store_true', 
                        help='Only print extracted data (for manual update)')
    args = parser.parse_args()
    
    print("Extracting data from analysis_results...")
    print(f"  NN dir: {NN_DIR}")
    print(f"  LogitLens dir: {LOGITLENS_DIR}")
    print(f"  Contextual dir: {CONTEXTUAL_DIR}")
    print()
    
    nn_data = load_nn_results()
    logitlens_data = load_logitlens_results()
    contextual_data = load_contextual_results()
    
    print(f"Found: {len(nn_data)} NN, {len(logitlens_data)} LogitLens, {len(contextual_data)} Contextual model combos")
    print()
    
    # Print data for copy-paste
    print("=" * 60)
    print("EXTRACTED DATA (copy to notebook if needed):")
    print("=" * 60)
    print()
    print(format_data_dict(nn_data, "NN_DATA"))
    print()
    print(format_data_dict(logitlens_data, "LOGITLENS_DATA"))
    print()
    print(format_data_dict(contextual_data, "CONTEXTUAL_DATA"))
    print()
    
    # Save to data.json
    output_data = {
        'nn': nn_data,
        'logitlens': logitlens_data,
        'contextual': contextual_data
    }
    data_json_path = Path(__file__).parent / "data.json"
    with open(data_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Saved data to {data_json_path}")
    print()
    
    if args.extract_only:
        return
    
    # Also regenerate plots using standalone script logic
    print("=" * 60)
    print("Regenerating plots...")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    OUTPUT_DIR = Path(__file__).parent / "paper_figures_output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Plotting config
    LLM_ORDER = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    ENC_ORDER = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    LLM_DISPLAY = {'llama3-8b': 'Llama3-8B', 'olmo-7b': 'OLMo-7B', 'qwen2-7b': 'Qwen2-7B'}
    ENC_DISPLAY = {'vit-l-14-336': 'CLIP ViT-L/14', 'siglip': 'SigLIP', 'dinov2-large-336': 'DINOv2'}
    LLM_COLORS = {'olmo-7b': plt.cm.Blues, 'llama3-8b': plt.cm.Greens, 'qwen2-7b': plt.cm.Reds}
    ENC_SHADES = [0.5, 0.7, 0.9]
    ENC_MARKERS = {'vit-l-14-336': '*', 'siglip': 'o', 'dinov2-large-336': '^'}
    ENC_FILL = {'vit-l-14-336': None, 'siglip': 'none', 'dinov2-large-336': None}
    
    colors = {(l, e): LLM_COLORS[l](ENC_SHADES[i]) for l in LLM_ORDER for i, e in enumerate(ENC_ORDER)}
    label = lambda l, e: f"{LLM_DISPLAY.get(l, l)} + {ENC_DISPLAY.get(e, e)}"
    
    # Create unified plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_style("whitegrid")
    
    configs = [
        (axes[0], nn_data, '(a) Static V-Lens (NN)'),
        (axes[1], logitlens_data, '(b) Logit Lens'),
        (axes[2], contextual_data, '(c) Contextual V-Lens'),
    ]
    
    handles = {}
    for ax, data, title in configs:
        all_layers = sorted(set(l for d in data.values() for l in d.keys()))
        for llm in LLM_ORDER:
            for enc in ENC_ORDER:
                key = f"{llm}+{enc}"
                if key not in data:
                    continue
                layers = sorted(data[key].keys())
                values = [data[key][l] for l in layers]
                lbl = label(llm, enc)
                marker, fill = ENC_MARKERS[enc], ENC_FILL[enc]
                if fill is not None:
                    line, = ax.plot(layers, values, marker=marker, color=colors[(llm, enc)],
                                   markerfacecolor=fill, markeredgewidth=2, linewidth=2.5, markersize=10)
                else:
                    line, = ax.plot(layers, values, marker=marker, color=colors[(llm, enc)],
                                   linewidth=2.5, markersize=10)
                if lbl not in handles:
                    handles[lbl] = line
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Interpretability %', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=11)
        if all_layers:
            ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)
    
    ordered = [(label(l, e), handles[label(l, e)]) for l in LLM_ORDER for e in ENC_ORDER if label(l, e) in handles]
    fig.legend([h for _, h in ordered], [l for l, _ in ordered],
              loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.25)
    
    for ext in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'fig1_unified_interpretability.{ext}', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plots to {OUTPUT_DIR}/")
    plt.close()


if __name__ == '__main__':
    main()

