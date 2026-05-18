#!/usr/bin/env python3
"""
Compute layer alignment data for the 3 larger off-the-shelf VLMs:
  - Molmo-72B
  - LLaVA-NeXT-34B
  - Qwen2.5-VL-32B

Reads raw contextual NN result files, counts which contextual_layer each
top-5 NN came from, and writes counts to data.json under
'{model-key}_layer_alignment'.

Run from repo root:
    source ../../env/bin/activate
    python paper_plots/compute_offtheshelf_layer_alignment.py
"""

import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
DATA_JSON = Path(__file__).parent / "data.json"

MODELS = [
    {
        'key': 'molmo-72b',
        'dir': REPO_ROOT / 'analysis_results' / 'contextual_nearest_neighbors' / 'molmo_72b' / 'allenai_Molmo-72B-0924',
        'display': 'Molmo-72B',
    },
    {
        'key': 'llava-next-34b',
        'dir': REPO_ROOT / 'analysis_results' / 'contextual_nearest_neighbors' / 'llava_next' / 'llava-hf_llava-v1.6-34b-hf',
        'display': 'LLaVA-NeXT-34B',
    },
    {
        'key': 'qwen2.5-vl-32b',
        'dir': REPO_ROOT / 'analysis_results' / 'contextual_nearest_neighbors' / 'qwen2_5_vl' / 'Qwen_Qwen2.5-VL-32B-Instruct',
        'display': 'Qwen2.5-VL-32B',
    },
]


def compute_layer_alignment(model_dir):
    """
    Read all contextual_neighbors_visual*_allLayers.json files and count
    which contextual_layer each NN came from.

    Returns: {str(visual_layer): {str(contextual_layer): count}}
    """
    counts = {}
    files = sorted(model_dir.glob('contextual_neighbors_visual*_allLayers.json'))
    if not files:
        raise FileNotFoundError(f"No NN files found in {model_dir}")

    for f in files:
        data = json.load(open(f))
        visual_layer = str(data['visual_layer'])
        layer_counts = defaultdict(int)

        for result in data['results']:
            for patch in result['patches']:
                for nn in patch['nearest_contextual_neighbors']:
                    cl = str(nn['contextual_layer'])
                    layer_counts[cl] += 1

        counts[visual_layer] = dict(layer_counts)
        total = sum(layer_counts.values())
        print(f"  visual_layer={visual_layer}: {total} NNs across layers {sorted(layer_counts.keys(), key=int)}")

    return counts


def main():
    data = json.load(open(DATA_JSON))

    for model in MODELS:
        key = model['key']
        data_key = f"{key}_layer_alignment"
        print(f"\n{'='*50}")
        print(f"Processing {model['display']}  →  data.json['{data_key}']")

        counts = compute_layer_alignment(model['dir'])
        data[data_key] = counts
        print(f"  ✓ {len(counts)} visual layers processed")

    with open(DATA_JSON, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ data.json updated")


if __name__ == '__main__':
    main()
