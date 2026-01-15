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
CONTEXTUAL_NN_RAW_DIR = RESULTS_DIR / "contextual_nearest_neighbors"


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
# ABLATIONS DATA
# =============================================================================

def load_ablations_nn_results():
    """Load NN results for ablation models."""
    data = defaultdict(dict)
    ablations_dir = NN_DIR / "ablations"
    if not ablations_dir.exists():
        return {}

    for results_file in ablations_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            results = json.load(f)
        path_str = str(results_file)

        # Extract layer number
        match = re.search(r'_layer(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))

        # Extract model name from path
        # Format: llm_judge_{model_name}_layer{N}_gpt5_cropped
        match = re.search(r'llm_judge_([^/]+?)_layer\d+', path_str)
        if not match:
            continue
        model_name = match.group(1)

        acc = results.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        data[model_name][layer_num] = round(acc, 2)

    return {k: dict(sorted(v.items())) for k, v in sorted(data.items())}


def load_ablations_logitlens_results():
    """Load LogitLens results for ablation models."""
    data = defaultdict(dict)
    ablations_dir = LOGITLENS_DIR / "ablations"
    if not ablations_dir.exists():
        return {}

    for results_file in ablations_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            res = json.load(f)
        path_str = str(results_file)

        # Extract layer from JSON or path
        layer_str = res.get('layer', '')
        if not layer_str:
            match = re.search(r'_layer(\d+)_', path_str)
            if match:
                layer_str = f"layer{match.group(1)}"

        if not layer_str.startswith('layer'):
            continue
        layer_num = int(layer_str.replace('layer', ''))

        # Extract model name
        match = re.search(r'llm_judge_([^/]+?)_layer\d+', path_str)
        if not match:
            continue
        model_name = match.group(1)

        acc = res.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        data[model_name][layer_num] = round(acc, 2)

    return {k: dict(sorted(v.items())) for k, v in sorted(data.items())}


def load_ablations_contextual_results():
    """Load Contextual NN results for ablation models (all layers)."""
    data = defaultdict(dict)
    ablations_dir = CONTEXTUAL_DIR / "ablations"
    if not ablations_dir.exists():
        return {}

    # Skip these model names - they have corrupted data (pointing to wrong input)
    # The olmo-7b_vit-l-14-336 in ablations folder actually reads from topbottom data!
    SKIP_CORRUPTED = {'olmo-7b_vit-l-14-336'}

    # Load contextual layers (1+)
    for results_file in ablations_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            res = json.load(f)
        path_str = str(results_file)

        # Extract layer
        match = re.search(r'_contextual(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))

        # Extract model name
        match = re.search(r'llm_judge_([^/]+?)_contextual\d+', path_str)
        if not match:
            continue
        model_name = match.group(1)

        # Skip corrupted data
        if model_name in SKIP_CORRUPTED:
            continue

        # Calculate accuracy
        results_list = res.get('results', [])
        if results_list:
            total = len(results_list)
            interp_count = sum(1 for r in results_list if r.get('interpretable', False))
            acc = (interp_count / total * 100.0) if total > 0 else 0.0
        else:
            acc = res.get('accuracy', 0.0)
            if acc <= 1.0:
                acc *= 100.0

        data[model_name][layer_num] = round(acc, 2)

    # Also load layer 0 from NN results
    ablations_nn_dir = NN_DIR / "ablations"
    if ablations_nn_dir.exists():
        for results_file in ablations_nn_dir.glob("**/results_*.json"):
            path_str = str(results_file)
            match = re.search(r'_layer(\d+)_', path_str)
            if not match or int(match.group(1)) != 0:
                continue

            match = re.search(r'llm_judge_([^/]+?)_layer0', path_str)
            if not match:
                continue
            model_name = match.group(1)

            with open(results_file, 'r') as f:
                results = json.load(f)
            acc = results.get('accuracy', 0.0)
            if acc <= 1.0:
                acc *= 100.0

            if 0 not in data.get(model_name, {}):
                data[model_name][0] = round(acc, 2)

    return {k: dict(sorted(v.items())) for k, v in sorted(data.items())}


# =============================================================================
# QWEN2-VL DATA (off-the-shelf model)
# =============================================================================

def load_qwen2vl_nn_results():
    """Load NN results for Qwen2-VL."""
    data = {}
    qwen2vl_dir = NN_DIR / "qwen2-vl"
    if not qwen2vl_dir.exists():
        return data

    for results_file in qwen2vl_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            results = json.load(f)
        path_str = str(results_file)

        match = re.search(r'_layer(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))

        acc = results.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        data[layer_num] = round(acc, 2)

    return dict(sorted(data.items()))


def load_qwen2vl_logitlens_results():
    """Load LogitLens results for Qwen2-VL."""
    data = {}
    qwen2vl_dir = LOGITLENS_DIR / "qwen2-vl"
    if not qwen2vl_dir.exists():
        return data

    for results_file in qwen2vl_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            res = json.load(f)
        path_str = str(results_file)

        layer_str = res.get('layer', '')
        if not layer_str:
            match = re.search(r'_layer(\d+)_', path_str)
            if match:
                layer_str = f"layer{match.group(1)}"

        # Handle both "layer0" format and plain "0" format
        if layer_str.isdigit():
            layer_num = int(layer_str)
        elif layer_str.startswith('layer'):
            layer_num = int(layer_str.replace('layer', ''))
        else:
            continue

        acc = res.get('accuracy', 0.0)
        if acc <= 1.0:
            acc *= 100.0
        data[layer_num] = round(acc, 2)

    return dict(sorted(data.items()))


def load_qwen2vl_contextual_results():
    """Load Contextual NN results for Qwen2-VL."""
    data = {}
    qwen2vl_dir = CONTEXTUAL_DIR / "qwen2-vl"
    if not qwen2vl_dir.exists():
        return data

    # Load contextual layers
    for results_file in qwen2vl_dir.glob("**/results_*.json"):
        with open(results_file, 'r') as f:
            res = json.load(f)
        path_str = str(results_file)

        match = re.search(r'_contextual(\d+)_', path_str)
        if not match:
            continue
        layer_num = int(match.group(1))

        results_list = res.get('results', [])
        if results_list:
            total = len(results_list)
            interp_count = sum(1 for r in results_list if r.get('interpretable', False))
            acc = (interp_count / total * 100.0) if total > 0 else 0.0
        else:
            acc = res.get('accuracy', 0.0)
            if acc <= 1.0:
                acc *= 100.0

        data[layer_num] = round(acc, 2)

    # Also load layer 0 from NN
    qwen2vl_nn_dir = NN_DIR / "qwen2-vl"
    if qwen2vl_nn_dir.exists():
        for results_file in qwen2vl_nn_dir.glob("**/results_*.json"):
            path_str = str(results_file)
            match = re.search(r'_layer(\d+)_', path_str)
            if not match or int(match.group(1)) != 0:
                continue

            with open(results_file, 'r') as f:
                results = json.load(f)
            acc = results.get('accuracy', 0.0)
            if acc <= 1.0:
                acc *= 100.0

            if 0 not in data:
                data[0] = round(acc, 2)

    return dict(sorted(data.items()))


def load_qwen2vl_layer_alignment_data(skip_if_slow=False):
    """
    Load layer alignment data for Qwen2-VL: which LLM layers NNs come from for each vision layer.

    Returns: {vision_layer: {llm_layer: count}}

    Warning: This reads large JSON files and can take several minutes.
    """
    if skip_if_slow:
        print("  Skipping Qwen2-VL layer alignment (--skip-alignment flag)")
        return {}

    qwen2vl_dir = CONTEXTUAL_NN_RAW_DIR / "qwen2_vl" / "Qwen_Qwen2-VL-7B-Instruct"
    if not qwen2vl_dir.exists():
        print(f"  ERROR: Qwen2-VL directory not found: {qwen2vl_dir}")
        return {}

    print(f"  Extracting Qwen2-VL layer alignment from {qwen2vl_dir}...")

    # Qwen2-VL has 28 layers
    vision_layers = [0, 1, 2, 4, 8, 16, 24, 26, 27]
    counts = {vl: defaultdict(int) for vl in vision_layers}

    for vl in vision_layers:
        json_file = qwen2vl_dir / f"contextual_neighbors_visual{vl}_allLayers.json"
        if not json_file.exists():
            print(f"    Warning: visual{vl} not found")
            continue

        print(f"    Loading visual{vl}...", end=" ", flush=True)
        with open(json_file, 'r') as f:
            data = json.load(f)

        total_nns = 0
        for result in data.get('results', []):
            # Qwen2-VL format: patches directly under result (no chunks)
            for patch in result.get('patches', []):
                for nn in patch.get('nearest_contextual_neighbors', []):
                    layer = nn.get('contextual_layer')
                    if layer is not None:
                        counts[vl][layer] += 1
                        total_nns += 1

        print(f"{total_nns:,} NNs")

    # Convert defaultdict to regular dict for JSON serialization
    return {vl: dict(lc) for vl, lc in counts.items()}


def load_qwen2vl_token_similarity_data():
    """
    Load token similarity data for Qwen2-VL (both vision and text).

    Returns: {'vision': {layer: mean_similarity}, 'text': {layer: mean_similarity}}
    """
    result = {'vision': {}, 'text': {}}

    # Load vision token similarities
    vision_dir = RESULTS_DIR / "sameToken_acrossLayers_similarity" / "qwen2_vl" / "Qwen_Qwen2-VL-7B-Instruct"
    vision_file = vision_dir / "similarity_across_layers_summary.json"

    if vision_file.exists():
        with open(vision_file, 'r') as f:
            data = json.load(f)
        global_averages = data.get('global_averages', {})
        for layer_str, layer_data in global_averages.items():
            layer_idx = int(layer_str)
            mean_sim = layer_data.get('same_token', {}).get('mean_similarity', 0.0)
            result['vision'][layer_idx] = round(mean_sim, 4)
    else:
        print(f"  Warning: Qwen2-VL vision token similarity not found: {vision_file}")

    # Load text token similarities
    text_dir = RESULTS_DIR / "sameToken_acrossLayers_text_similarity" / "qwen2_vl" / "Qwen_Qwen2-VL-7B-Instruct"
    text_file = text_dir / "text_similarity_across_layers_summary.json"

    if text_file.exists():
        with open(text_file, 'r') as f:
            data = json.load(f)
        global_averages = data.get('global_averages', {})
        for layer_str, layer_data in global_averages.items():
            layer_idx = int(layer_str)
            mean_sim = layer_data.get('same_token', {}).get('mean_similarity', 0.0)
            result['text'][layer_idx] = round(mean_sim, 4)
    else:
        print(f"  Warning: Qwen2-VL text token similarity not found: {text_file}")

    return result


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
# L2 NORM DATA (for token norm histograms)
# =============================================================================

VISION_L2NORM_DIR = RESULTS_DIR / "sameToken_acrossLayers_l2norm"
TEXT_L2NORM_DIR = RESULTS_DIR / "sameToken_acrossLayers_text_l2norm"


def load_l2norm_data():
    """
    Load L2 norm histogram data (vision and text).

    Returns: {
        'vision': {model_key: {layer: {counts, bin_edges, mean, std, ...}}},
        'text': {model_key: {layer: {counts, bin_edges, mean, std, ...}}}
    }
    """
    result = {'vision': {}, 'text': {}}

    # Load vision L2 norms
    if VISION_L2NORM_DIR.exists():
        for ckpt_dir in VISION_L2NORM_DIR.iterdir():
            if not ckpt_dir.is_dir():
                continue
            summary_file = ckpt_dir / "l2norm_across_layers_summary.json"
            if not summary_file.exists():
                continue

            llm, encoder = extract_model_from_checkpoint(ckpt_dir.name)
            if not llm or not encoder:
                continue

            key = f"{llm}+{encoder}"
            with open(summary_file) as f:
                data = json.load(f)

            histogram_data = data.get('histogram_data', {})
            result['vision'][key] = {
                int(layer): layer_data
                for layer, layer_data in histogram_data.items()
            }

    # Load text L2 norms
    if TEXT_L2NORM_DIR.exists():
        for ckpt_dir in TEXT_L2NORM_DIR.iterdir():
            if not ckpt_dir.is_dir():
                continue
            summary_file = ckpt_dir / "text_l2norm_across_layers_summary.json"
            if not summary_file.exists():
                continue

            llm, encoder = extract_model_from_checkpoint(ckpt_dir.name)
            if not llm or not encoder:
                continue

            key = f"{llm}+{encoder}"
            with open(summary_file) as f:
                data = json.load(f)

            histogram_data = data.get('histogram_data', {})
            result['text'][key] = {
                int(layer): layer_data
                for layer, layer_data in histogram_data.items()
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
LLM_LAYERS_DEFAULT = [0, 1, 2, 4, 8, 16, 24, 30, 31]  # layer 0 = Input Embedding Matrix
LLM_LAYERS_QWEN = [0, 1, 2, 4, 8, 16, 24, 26, 27]

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


def load_similarity_histogram_data(visual_layers=[0, 8, 16], num_bins=30):
    """
    Load similarity histogram data from contextual NN JSON files.
    Returns binned histogram data (counts and bin_edges) for each model and visual layer.
    """
    import numpy as np
    
    MODELS = [
        ("olmo-7b", "vit-l-14-336"),
        ("olmo-7b", "siglip"),
        ("olmo-7b", "dinov2-large-336"),
        ("llama3-8b", "vit-l-14-336"),
        ("llama3-8b", "siglip"),
        ("llama3-8b", "dinov2-large-336"),
        ("qwen2-7b", "vit-l-14-336"),
        ("qwen2-7b", "siglip"),
        ("qwen2-7b", "dinov2-large-336"),
    ]
    
    def find_model_dir(llm, encoder):
        """Find the model directory for contextual nearest neighbor results."""
        if llm == "qwen2-7b" and encoder == "vit-l-14-336":
            pattern = f"train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_seed10_step12000-unsharded"
        else:
            pattern = f"train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_step12000-unsharded"
        
        matches = list(CONTEXTUAL_NN_RAW_DIR.glob(pattern))
        matches = [m for m in matches if 'lite' not in str(m)]
        return matches[0] if matches else None
    
    all_data = {}
    
    for llm, encoder in MODELS:
        model_key = f"{llm}+{encoder}"
        model_dir = find_model_dir(llm, encoder)
        
        if not model_dir:
            print(f"  {model_key}: directory not found, skipping")
            continue
        
        all_data[model_key] = {}
        
        for visual_layer in visual_layers:
            json_file = model_dir / f"contextual_neighbors_visual{visual_layer}_allLayers.json"
            
            if not json_file.exists():
                print(f"  {model_key} visual{visual_layer}: not found")
                continue
            
            # Load similarities
            similarities = []
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for result in data.get('results', []):
                for chunk in result.get('chunks', []):
                    for patch in chunk.get('patches', []):
                        nns = patch.get('nearest_contextual_neighbors', [])
                        if nns:
                            similarities.append(nns[0].get('similarity', 0))
            
            if not similarities:
                continue
            
            # Compute histogram
            counts, bin_edges = np.histogram(similarities, bins=num_bins, range=(0, 1))
            
            all_data[model_key][visual_layer] = {
                'counts': counts.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mean': float(np.mean(similarities)),
                'median': float(np.median(similarities)),
                'n_samples': len(similarities)
            }
            
            print(f"  {model_key} visual{visual_layer}: {len(similarities):,} samples, mean={np.mean(similarities):.3f}")
    
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
        # Only extract layer alignment (main models + Qwen2-VL)
        print("Extracting layer alignment data (this takes several minutes)...")
        layer_alignment_data = load_layer_alignment_data(skip_if_slow=False)

        print("\nExtracting Qwen2-VL layer alignment data...")
        qwen2vl_layer_alignment = load_qwen2vl_layer_alignment_data(skip_if_slow=False)

        # Load existing data.json and update just layer_alignment
        data_json_path = Path(__file__).parent / "data.json"
        if data_json_path.exists():
            with open(data_json_path, 'r') as f:
                output_data = json.load(f)
        else:
            output_data = {}
        output_data['layer_alignment'] = layer_alignment_data
        if qwen2vl_layer_alignment:
            output_data['qwen2vl_layer_alignment'] = qwen2vl_layer_alignment

        with open(data_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Updated layer_alignment and qwen2vl_layer_alignment in {data_json_path}")
        return
    
    nn_data = load_nn_results()
    logitlens_data = load_logitlens_results()
    contextual_data = load_contextual_results()

    print(f"Found: {len(nn_data)} NN, {len(logitlens_data)} LogitLens, {len(contextual_data)} Contextual model combos")

    # Load ablations data
    print("\nExtracting ablations data...")
    ablations_nn = load_ablations_nn_results()
    ablations_logitlens = load_ablations_logitlens_results()
    ablations_contextual = load_ablations_contextual_results()
    print(f"  Found: {len(ablations_nn)} NN, {len(ablations_logitlens)} LogitLens, {len(ablations_contextual)} Contextual ablation models")

    # Load Qwen2-VL data
    print("\nExtracting Qwen2-VL data...")
    qwen2vl_nn = load_qwen2vl_nn_results()
    qwen2vl_logitlens = load_qwen2vl_logitlens_results()
    qwen2vl_contextual = load_qwen2vl_contextual_results()
    print(f"  NN: {len(qwen2vl_nn)} layers, LogitLens: {len(qwen2vl_logitlens)} layers, Contextual: {len(qwen2vl_contextual)} layers")

    # Load Qwen2-VL layer alignment data (optional, slow)
    qwen2vl_layer_alignment = None
    if not args.skip_alignment:
        print("\nExtracting Qwen2-VL layer alignment data...")
        qwen2vl_layer_alignment = load_qwen2vl_layer_alignment_data(skip_if_slow=False)
        if qwen2vl_layer_alignment:
            print(f"  Found: {len(qwen2vl_layer_alignment)} vision layers with layer alignment data")

    # Load Qwen2-VL token similarity data
    print("\nExtracting Qwen2-VL token similarity data...")
    qwen2vl_token_similarity = load_qwen2vl_token_similarity_data()
    if qwen2vl_token_similarity.get('vision'):
        print(f"  Found: {len(qwen2vl_token_similarity['vision'])} layers with token similarity data")

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

    # Load L2 norm data
    print("Extracting L2 norm data...")
    l2norm_data = load_l2norm_data()
    l2norm_data['vision'] = {k: v for k, v in l2norm_data['vision'].items() if k in main_models}
    l2norm_data['text'] = {k: v for k, v in l2norm_data['text'].items() if k in main_models}
    print(f"  Vision: {len(l2norm_data['vision'])}, Text: {len(l2norm_data['text'])} models")
    print()

    # Layer alignment data (optional, slow)
    layer_alignment_data = None
    if not args.skip_alignment:
        print("Extracting layer alignment data (this takes several minutes)...")
        layer_alignment_data = load_layer_alignment_data(skip_if_slow=False)
        print(f"Found: {len(layer_alignment_data)} models with layer alignment data")
        print()
    
    # Similarity histogram data
    print("Extracting similarity histogram data (layers 0, 8, 16)...")
    similarity_histogram_data = load_similarity_histogram_data(visual_layers=[0, 8, 16])
    print(f"Found: {len(similarity_histogram_data)} models with similarity histogram data")
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
        'token_similarity': token_similarity_data,
        'l2norm': l2norm_data,
        'similarity_histograms': similarity_histogram_data,
        'ablations': {
            'nn': ablations_nn,
            'logitlens': ablations_logitlens,
            'contextual': ablations_contextual
        },
        'qwen2vl': {
            'nn': qwen2vl_nn,
            'logitlens': qwen2vl_logitlens,
            'contextual': qwen2vl_contextual
        }
    }

    data_json_path = Path(__file__).parent / "data.json"

    # If skipping alignment, preserve existing layer_alignment from data.json
    if args.skip_alignment and data_json_path.exists():
        with open(data_json_path, 'r') as f:
            existing_data = json.load(f)
        if 'layer_alignment' in existing_data:
            output_data['layer_alignment'] = existing_data['layer_alignment']
            print("  (Preserved existing layer_alignment)")
        if 'qwen2vl_layer_alignment' in existing_data:
            output_data['qwen2vl_layer_alignment'] = existing_data['qwen2vl_layer_alignment']
            print("  (Preserved existing qwen2vl_layer_alignment)")
    else:
        if layer_alignment_data:
            output_data['layer_alignment'] = layer_alignment_data
        if qwen2vl_layer_alignment:
            output_data['qwen2vl_layer_alignment'] = qwen2vl_layer_alignment

    # Add Qwen2-VL token similarity if available
    if qwen2vl_token_similarity.get('vision'):
        output_data['qwen2vl_token_similarity'] = qwen2vl_token_similarity

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
    
    # Order: Input Embedding Matrix, Output Embedding Matrix (LogitLens), LN-Lens (ours last)
    configs = [
        (axes[0], nn_data, '(a) Input Embedding Matrix'),
        (axes[1], logitlens_data, '(b) Output Embedding Matrix (LogitLens)'),
        (axes[2], contextual_data, '(c) LN-Lens (Ours)'),
    ]

    handles = {}
    for idx, (ax, data, title) in enumerate(configs):
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
        # Shared y-axis: only show label on leftmost plot
        if idx == 0:
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

