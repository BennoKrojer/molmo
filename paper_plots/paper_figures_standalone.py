#!/usr/bin/env python3
"""
V-Lens Paper Figures - Standalone Script
=========================================
Just run: python paper_figures_standalone.py

Requirements: pip install matplotlib seaborn numpy
"""

import matplotlib
matplotlib.use('Agg')  # For headless servers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path('paper_figures_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# RAW DATA (LLM Judge evaluated interpretability %)
# =============================================================================

NN_DATA = {
    "llama3-8b+dinov2-large-336": {0: 20.33, 1: 17.0, 2: 19.0, 3: 18.0, 4: 20.0, 8: 15.0, 12: 19.0, 16: 20.0, 20: 24.0, 24: 19.0, 28: 22.0, 32: 12.0},
    "llama3-8b+siglip": {0: 23.33, 1: 30.0, 2: 24.0, 3: 29.0, 4: 28.0, 8: 31.0, 12: 31.0, 16: 29.0, 20: 34.0, 24: 27.0, 28: 27.0, 32: 29.0},
    "llama3-8b+vit-l-14-336": {0: 35.33, 1: 30.0, 2: 37.0, 3: 32.0, 4: 29.0, 8: 37.0, 12: 44.0, 16: 51.0, 20: 52.0, 24: 47.0, 28: 43.0, 32: 21.0},
    "olmo-7b+dinov2-large-336": {0: 42.0, 1: 45.0, 2: 40.0, 3: 41.0, 4: 47.0, 8: 56.0, 12: 61.0, 16: 67.0, 20: 70.0, 24: 67.0, 28: 67.0, 32: 33.0},
    "olmo-7b+siglip": {0: 41.67, 1: 39.0, 2: 38.0, 3: 28.0, 4: 39.0, 8: 45.0, 12: 55.0, 16: 49.0, 20: 55.0, 24: 56.0, 28: 53.0, 32: 22.0},
    "olmo-7b+vit-l-14-336": {0: 55.0, 1: 52.0, 2: 56.0, 3: 60.0, 4: 59.0, 8: 60.0, 12: 62.0, 16: 62.0, 20: 63.0, 24: 59.0, 28: 62.0, 32: 35.0},
    "qwen2-7b+dinov2-large-336": {0: 7.0, 1: 10.0, 2: 9.0, 3: 7.0, 4: 9.0, 8: 11.0, 12: 11.0, 16: 11.0, 20: 12.0, 24: 14.0, 28: 9.0},
    "qwen2-7b+siglip": {0: 5.33, 1: 4.0, 2: 5.0, 3: 5.0, 4: 4.0, 8: 3.0, 12: 5.0, 16: 4.0, 20: 5.0, 24: 5.0, 28: 9.0},
    "qwen2-7b+vit-l-14-336": {0: 17.67, 1: 15.0, 2: 9.0, 3: 13.0, 4: 15.0, 8: 18.0, 12: 18.0, 16: 9.0, 20: 8.0, 24: 16.0, 28: 10.0},
}

LOGITLENS_DATA = {
    "llama3-8b+dinov2-large-336": {0: 9.0, 1: 7.0, 2: 9.0, 3: 11.0, 4: 5.0, 8: 10.0, 12: 10.0, 16: 11.0, 20: 9.0, 24: 7.0, 28: 7.0, 29: 9.0, 30: 13.0, 31: 7.0, 32: 7.0},
    "llama3-8b+siglip": {0: 9.0, 1: 9.0, 2: 10.0, 3: 8.0, 4: 12.0, 8: 10.0, 12: 9.0, 16: 9.0, 20: 13.0, 24: 14.0, 28: 8.0, 29: 10.0, 30: 9.0, 31: 9.0, 32: 7.0},
    "llama3-8b+vit-l-14-336": {0: 13.0, 1: 10.0, 2: 10.0, 3: 11.0, 4: 14.0, 8: 10.0, 12: 7.0, 16: 12.0, 20: 26.0, 24: 50.0, 28: 52.0, 29: 62.0, 30: 64.0, 31: 76.0, 32: 81.0},
    "olmo-7b+dinov2-large-336": {0: 11.0, 1: 13.0, 2: 13.0, 3: 13.0, 4: 17.0, 8: 15.0, 12: 23.0, 16: 39.0, 20: 61.0, 24: 78.0, 28: 76.0, 29: 78.0, 30: 69.0, 31: 56.0, 32: 32.0},
    "olmo-7b+siglip": {0: 14.0, 1: 20.0, 2: 15.0, 3: 21.0, 4: 16.0, 8: 20.0, 12: 22.0, 16: 26.0, 20: 52.0, 24: 69.0, 28: 83.0, 29: 86.0, 30: 82.0, 31: 63.0, 32: 43.0},
    "olmo-7b+vit-l-14-336": {0: 11.0, 1: 8.0, 2: 18.0, 3: 19.0, 4: 19.0, 8: 22.0, 12: 25.0, 16: 23.0, 20: 49.0, 24: 75.0, 28: 78.0, 29: 82.0, 30: 74.0, 31: 59.0, 32: 31.0},
    "qwen2-7b+dinov2-large-336": {0: 15.0, 1: 9.0, 2: 10.0, 3: 12.0, 4: 9.0, 8: 7.0, 12: 8.0, 16: 13.0, 20: 14.0, 24: 25.0, 25: 34.0, 26: 42.0, 27: 56.0, 28: 45.0},
    "qwen2-7b+siglip": {0: 8.0, 1: 7.0, 2: 9.0, 3: 7.0, 4: 9.0, 8: 8.0, 12: 6.0, 16: 8.0, 20: 7.0, 24: 11.0, 25: 6.0, 26: 11.0, 27: 6.0, 28: 12.0},
    "qwen2-7b+vit-l-14-336": {0: 6.0, 1: 4.0, 2: 6.0, 3: 2.0, 4: 3.0, 8: 7.0, 12: 12.0, 16: 8.0, 20: 9.0, 24: 43.0, 25: 51.0, 26: 59.0, 27: 78.0, 28: 71.0},
}

CONTEXTUAL_DATA = {
    "llama3-8b+dinov2-large-336": {0: 20.33, 1: 82.0, 2: 81.0, 4: 82.0, 8: 82.0, 16: 82.0, 24: 83.33},
    "llama3-8b+siglip": {0: 23.33, 1: 63.0, 2: 60.0, 4: 65.0, 8: 62.0, 16: 58.0, 24: 63.89},
    "llama3-8b+vit-l-14-336": {0: 35.33, 1: 82.0, 2: 84.0, 4: 79.0, 8: 82.0, 16: 82.0, 24: 70.0},
    "olmo-7b+dinov2-large-336": {0: 42.0, 1: 81.0, 2: 79.0, 4: 80.0, 8: 80.0, 16: 81.0, 24: 82.43},
    "olmo-7b+siglip": {0: 41.67, 1: 63.0, 2: 64.0, 4: 65.0, 8: 64.0, 16: 65.0, 24: 60.87},
    "olmo-7b+vit-l-14-336": {0: 54.67, 1: 71.0, 2: 71.0, 4: 70.0, 8: 71.0, 16: 70.0, 24: 69.7},
    "qwen2-7b+dinov2-large-336": {0: 7.0, 1: 79.0, 2: 81.0, 4: 83.0, 8: 81.0, 16: 80.0, 24: 82.81},
    "qwen2-7b+siglip": {0: 5.33, 1: 65.0, 2: 64.0, 4: 62.0, 8: 68.0, 16: 69.0, 24: 66.67},
    "qwen2-7b+vit-l-14-336": {0: 17.67, 1: 76.0, 2: 80.0, 4: 78.0, 8: 76.0, 16: 79.0, 24: 78.0, 26: 71.43},
}

# =============================================================================
# PLOTTING CONFIG
# =============================================================================

LLM_DISPLAY = {'llama3-8b': 'Llama3-8B', 'olmo-7b': 'OLMo-7B', 'qwen2-7b': 'Qwen2-7B'}
ENC_DISPLAY = {'vit-l-14-336': 'CLIP ViT-L/14', 'siglip': 'SigLIP', 'dinov2-large-336': 'DINOv2'}
LLM_ORDER = ['olmo-7b', 'llama3-8b', 'qwen2-7b']
ENC_ORDER = ['vit-l-14-336', 'siglip', 'dinov2-large-336']
LLM_COLORS = {'olmo-7b': plt.cm.Blues, 'llama3-8b': plt.cm.Greens, 'qwen2-7b': plt.cm.Reds}
ENC_SHADES = [0.5, 0.7, 0.9]
ENC_MARKERS = {'vit-l-14-336': '*', 'siglip': 'o', 'dinov2-large-336': '^'}
ENC_FILL = {'vit-l-14-336': None, 'siglip': 'none', 'dinov2-large-336': None}

def get_colors():
    return {(l, e): LLM_COLORS[l](ENC_SHADES[i]) for l in LLM_ORDER for i, e in enumerate(ENC_ORDER)}

def label(llm, enc):
    return f"{LLM_DISPLAY.get(llm, llm)} + {ENC_DISPLAY.get(enc, enc)}"

# =============================================================================
# MAIN FIGURE: 3-panel unified plot
# =============================================================================

def create_unified_plot():
    colors = get_colors()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_style("whitegrid")
    
    configs = [
        (axes[0], NN_DATA, '(a) Static V-Lens (NN)'),
        (axes[1], LOGITLENS_DATA, '(b) Logit Lens'),
        (axes[2], CONTEXTUAL_DATA, '(c) Contextual V-Lens'),
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
    
    # Legend
    ordered = [(label(l, e), handles[label(l, e)]) for l in LLM_ORDER for e in ENC_ORDER if label(l, e) in handles]
    fig.legend([h for _, h in ordered], [l for l, _ in ordered],
              loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.25)
    
    for ext in ['pdf', 'png']:
        plt.savefig(OUTPUT_DIR / f'fig1_unified_interpretability.{ext}', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/fig1_unified_interpretability.pdf")
    plt.close()

# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    print("V-Lens Paper Figures")
    print("=" * 40)
    create_unified_plot()
    print(f"\nFigures saved to: {OUTPUT_DIR.absolute()}")

