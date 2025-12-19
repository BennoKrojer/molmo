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


# =============================================================================
# EXPECTED LAYERS: [0, 1, 2, 4, 8, 16, 24, N-2, N-1] where N varies per LLM
# =============================================================================

def get_expected_layers(llm):
    """Get expected layers for a given LLM."""
    if llm in ['olmo-7b', 'llama3-8b']:
        return [0, 1, 2, 4, 8, 16, 24, 30, 31]
    elif llm == 'qwen2-7b':
        return [0, 1, 2, 4, 8, 16, 24, 26, 27]
    else:
        return [0, 1, 2, 4, 8, 16, 24, 30, 31]


def filter_to_expected_layers(data):
    """Filter data to only include expected layers for each model."""
    filtered = {}
    for key, layer_data in data.items():
        llm = key.split('+')[0]
        expected = set(get_expected_layers(llm))
        filtered[key] = {k: v for k, v in layer_data.items() if k in expected}
    return filtered


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


# =============================================================================
# TOKEN SIMILARITY DATA (for same-token similarity plots)
# =============================================================================

VISION_SIM_DIR = RESULTS_DIR / "sameToken_acrossLayers_similarity"
TEXT_SIM_DIR = RESULTS_DIR / "sameToken_acrossLayers_text_similarity"


def extract_model_from_checkpoint(checkpoint_path):
    """Extract LLM and encoder from checkpoint path."""
    llms = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
    encoders = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
    
    for l in llms:
        if l in checkpoint_path:
            llm = l
            break
    else:
        return None, None
    
    for e in encoders:
        if e in checkpoint_path:
            encoder = e
            break
    else:
        return None, None
    
    return llm, encoder


def load_token_similarity_data():
    """
    Load token similarity data (vision and text).
    
    Returns: {
        'vision': {model_key: {layer: similarity}},
        'text': {model_key: {layer: similarity}}
    }
    """
    result = {'vision': {}, 'text': {}}
    
    # Load vision similarity
    if VISION_SIM_DIR.exists():
        for ckpt_dir in VISION_SIM_DIR.iterdir():
            if not ckpt_dir.is_dir():
                continue
            summary_file = ckpt_dir / "similarity_across_layers_summary.json"
            if not summary_file.exists():
                continue
            
            llm, encoder = extract_model_from_checkpoint(ckpt_dir.name)
            if not llm or not encoder:
                continue
            
            key = f"{llm}+{encoder}"
            with open(summary_file) as f:
                data = json.load(f)
            
            ga = data.get('global_averages', {})
            result['vision'][key] = {
                int(layer): round(ga[layer]['same_token']['mean_similarity'], 4)
                for layer in ga
            }
    
    # Load text similarity
    if TEXT_SIM_DIR.exists():
        for ckpt_dir in TEXT_SIM_DIR.iterdir():
            if not ckpt_dir.is_dir():
                continue
            summary_file = ckpt_dir / "text_similarity_across_layers_summary.json"
            if not summary_file.exists():
                continue
            
            llm, encoder = extract_model_from_checkpoint(ckpt_dir.name)
            if not llm or not encoder:
                continue
            
            key = f"{llm}+{encoder}"
            with open(summary_file) as f:
                data = json.load(f)
            
            ga = data.get('global_averages', {})
            result['text'][key] = {
                int(layer): round(ga[layer]['same_token']['mean_similarity'], 4)
                for layer in ga
            }
    
    # Sort by layer
    for modality in ['vision', 'text']:
        result[modality] = {
            k: dict(sorted(v.items())) 
            for k, v in sorted(result[modality].items())
        }
    
    return result


# =============================================================================
# LAYER ALIGNMENT DATA (for histogram plots)
# =============================================================================

CONTEXTUAL_NN_DIR = RESULTS_DIR / "contextual_nearest_neighbors"

# Layer configurations per LLM (Qwen has 28 layers, others have 32)
VISION_LAYERS_DEFAULT = [0, 1, 2, 4, 8, 16, 24, 30, 31]
VISION_LAYERS_QWEN = [0, 1, 2, 4, 8, 16, 24, 26, 27]
LLM_LAYERS_DEFAULT = [1, 2, 4, 8, 16, 24, 30, 31]
LLM_LAYERS_QWEN = [1, 2, 4, 8, 16, 24, 26, 27]

def get_vision_layers(llm):
    return VISION_LAYERS_QWEN if 'qwen' in llm else VISION_LAYERS_DEFAULT

def get_llm_layers(llm):
    return LLM_LAYERS_QWEN if 'qwen' in llm else LLM_LAYERS_DEFAULT

ALIGNMENT_MODELS = [
    ("llama3-8b", "dinov2-large-336"),
    ("llama3-8b", "siglip"),
    ("llama3-8b", "vit-l-14-336"),
    ("olmo-7b", "dinov2-large-336"),
    ("olmo-7b", "siglip"),
    ("olmo-7b", "vit-l-14-336"),
    ("qwen2-7b", "dinov2-large-336"),
    ("qwen2-7b", "siglip"),
    ("qwen2-7b", "vit-l-14-336"),
]


def find_contextual_model_dir(llm, encoder):
    """Find the model directory for contextual NN results."""
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        pattern = f"*{llm}_{encoder}_seed10_step12000-unsharded"
    else:
        pattern = f"*{llm}_{encoder}_step12000-unsharded"
    
    matches = list(CONTEXTUAL_NN_DIR.glob(pattern))
    matches = [m for m in matches if 'lite' not in str(m)]
    return matches[0] if matches else None


def load_layer_alignment_data(skip_if_slow=False):
    """
    Load layer alignment data: which LLM layers NNs come from for each vision layer.
    
    Returns: {model_key: {vision_layer: {llm_layer: count}}}
    
    Warning: This reads large JSON files and can take several minutes.
    """
    if skip_if_slow:
        print("  Skipping layer alignment (--skip-alignment flag)")
        return {}
    
    all_data = {}
    
    for llm, encoder in ALIGNMENT_MODELS:
        model_key = f"{llm}+{encoder}"
        print(f"  {model_key}...")
        
        model_dir = find_contextual_model_dir(llm, encoder)
        if not model_dir:
            print(f"    ERROR: Directory not found")
            continue
        
        # Use model-specific vision layers
        vision_layers = get_vision_layers(llm)
        counts = {vl: defaultdict(int) for vl in vision_layers}
        
        for vl in vision_layers:
            json_file = model_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
            if not json_file.exists():
                print(f"    Warning: visual{vl} not found")
                continue
            
            print(f"    Loading visual{vl}...", end=" ", flush=True)
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            total_nns = 0
            for result in data.get('results', []):
                for chunk in result.get('chunks', []):
                    for patch in chunk.get('patches', []):
                        for nn in patch.get('nearest_contextual_neighbors', []):
                            layer = nn.get('contextual_layer')
                            if layer is not None:
                                counts[vl][layer] += 1
                                total_nns += 1
            
            print(f"{total_nns:,} NNs")
        
        # Convert defaultdict to regular dict for JSON serialization
        all_data[model_key] = {vl: dict(lc) for vl, lc in counts.items()}
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description='Update paper figures data')
    parser.add_argument('--extract-only', action='store_true', 
                        help='Only print extracted data (for manual update)')
    parser.add_argument('--skip-alignment', action='store_true',
                        help='Skip layer alignment data (slow to extract)')
    parser.add_argument('--alignment-only', action='store_true',
                        help='Only extract layer alignment data')
    args = parser.parse_args()
    
    print("Extracting data from analysis_results...")
    print(f"  NN dir: {NN_DIR}")
    print(f"  LogitLens dir: {LOGITLENS_DIR}")
    print(f"  Contextual dir: {CONTEXTUAL_DIR}")
    print(f"  Contextual NN dir: {CONTEXTUAL_NN_DIR}")
    print()
    
    if args.alignment_only:
        # Only extract layer alignment
        print("Extracting layer alignment data (this takes several minutes)...")
        layer_alignment_data = load_layer_alignment_data(skip_if_slow=False)
        
        # Load existing data.json and update just layer_alignment
        data_json_path = Path(__file__).parent / "data.json"
        if data_json_path.exists():
            with open(data_json_path, 'r') as f:
                output_data = json.load(f)
        else:
            output_data = {}
        output_data['layer_alignment'] = layer_alignment_data
        
        with open(data_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Updated layer_alignment in {data_json_path}")
        return
    
    nn_data = load_nn_results()
    logitlens_data = load_logitlens_results()
    contextual_data = load_contextual_results()
    
    print(f"Found: {len(nn_data)} NN, {len(logitlens_data)} LogitLens, {len(contextual_data)} Contextual model combos")
    
    # Filter to expected layers only
    print("Filtering to expected layers: [0, 1, 2, 4, 8, 16, 24, N-2, N-1]...")
    nn_data = filter_to_expected_layers(nn_data)
    logitlens_data = filter_to_expected_layers(logitlens_data)
    contextual_data = filter_to_expected_layers(contextual_data)
    
    # Filter to main 9 model combinations only
    main_models = {f"{l}+{e}" for l in ['olmo-7b', 'llama3-8b', 'qwen2-7b'] 
                   for e in ['vit-l-14-336', 'siglip', 'dinov2-large-336']}
    nn_data = {k: v for k, v in nn_data.items() if k in main_models}
    logitlens_data = {k: v for k, v in logitlens_data.items() if k in main_models}
    contextual_data = {k: v for k, v in contextual_data.items() if k in main_models}
    
    # Load token similarity data
    print("Extracting token similarity data...")
    token_similarity_data = load_token_similarity_data()
    token_similarity_data['vision'] = {k: v for k, v in token_similarity_data['vision'].items() if k in main_models}
    token_similarity_data['text'] = {k: v for k, v in token_similarity_data['text'].items() if k in main_models}
    print(f"  Vision: {len(token_similarity_data['vision'])}, Text: {len(token_similarity_data['text'])} models")
    print()
    
    # Layer alignment data (optional, slow)
    layer_alignment_data = None
    if not args.skip_alignment:
        print("Extracting layer alignment data (this takes several minutes)...")
        layer_alignment_data = load_layer_alignment_data(skip_if_slow=False)
        print(f"Found: {len(layer_alignment_data)} models with layer alignment data")
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
        'contextual': contextual_data,
        'token_similarity': token_similarity_data
    }
    
    data_json_path = Path(__file__).parent / "data.json"
    
    # If skipping alignment, preserve existing layer_alignment from data.json
    if args.skip_alignment and data_json_path.exists():
        with open(data_json_path, 'r') as f:
            existing_data = json.load(f)
        if 'layer_alignment' in existing_data:
            output_data['layer_alignment'] = existing_data['layer_alignment']
            print("  (Preserved existing layer_alignment)")
    elif layer_alignment_data:
        output_data['layer_alignment'] = layer_alignment_data
    
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

