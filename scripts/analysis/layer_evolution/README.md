# Layer Evolution Analysis

This directory contains scripts for analyzing how the interpretation of vision tokens evolves across layers in vision-language models.

## Quick Start

To generate all visualizations at once:

```bash
bash scripts/analysis/layer_evolution/generate_all_visualizations.sh
```

This will create single-model and comparison visualizations for all available model combinations.

**All results are saved to:** `analysis_results/layer_evolution/`

The scripts automatically use absolute paths relative to the repository root, so they can be run from any directory.

## Scripts

### 1. `visualize_interpretation_types.py`

Visualizes the evolution of interpretation types (concrete, abstract, global) across contextual layers for a single model combination.

**Usage:**
```bash
python scripts/analysis/layer_evolution/visualize_interpretation_types.py \
    --llm olmo-7b \
    --vision-encoder vit-l-14-336 \
    --results-dir analysis_results/llm_judge_contextual_nn \
    --nn-results-dir analysis_results/llm_judge_nearest_neighbors \
    --output analysis_results/layer_evolution/interpretation_types_olmo-7b_vit-l-14-336.pdf
```

**Arguments:**
- `--llm`: LLM model name (choices: `olmo-7b`, `llama3-8b`, `qwen2-7b`)
- `--vision-encoder`: Vision encoder name (choices: `vit-l-14-336`, `siglip`, `dinov2-large-336`)
- `--results-dir`: Directory containing contextual NN LLM judge results (default: `analysis_results/llm_judge_contextual_nn`)
- `--nn-results-dir`: Directory containing nearest neighbors LLM judge results for layer 0 (default: `analysis_results/llm_judge_nearest_neighbors`)
- `--output`: Output path for visualization (auto-generated if not provided)

**Output:**
- A stacked bar plot showing the percentage of concrete/abstract/global interpretable tokens per layer
- Both PDF (high-res) and PNG (preview) versions
- A data table printed to console

### 2. `compare_interpretation_types.py`

Creates side-by-side visualizations comparing interpretation type evolution across multiple model combinations.

**Usage:**
```bash
python scripts/analysis/layer_evolution/compare_interpretation_types.py \
    --models "olmo-7b:vit-l-14-336" "qwen2-7b:vit-l-14-336" \
    --results-dir analysis_results/llm_judge_contextual_nn \
    --nn-results-dir analysis_results/llm_judge_nearest_neighbors \
    --output analysis_results/layer_evolution/compare_olmo_vs_qwen2.pdf
```

**Arguments:**
- `--models`: One or more model combinations in format `"llm:encoder"` (required)
- `--results-dir`: Directory containing contextual NN LLM judge results
- `--nn-results-dir`: Directory containing nearest neighbors LLM judge results for layer 0
- `--output`: Output path for visualization (auto-generated if not provided)

**Output:**
- A figure with side-by-side subplots for each model
- Both PDF and PNG versions
- Data tables for each model printed to console

## Interpretation Types

### What's Being Measured

For each **vision token/patch**, we extract its top-5 **nearest neighbor words** from the LLM's embedding space. The LLM judge then evaluates these words and categorizes them into three types:

1. **Concrete**: Nearest neighbor words literally describe something visible inside the image region (e.g., objects, colors, text, shapes)

2. **Abstract**: Nearest neighbor words describe broader concepts, emotions, or activities related to the region but not literally visible (e.g., "luxury", "beautiful", "transportation")

3. **Global**: Nearest neighbor words describe something present elsewhere in the image (outside the highlighted region)

### Classification Logic

Each vision token is classified according to a hierarchy based on its nearest neighbors:
- If any concrete words are found → classified as **Concrete**
- Else if any abstract words are found → classified as **Abstract**  
- Else if any global words are found → classified as **Global**
- Else → not interpretable (excluded from percentages)

**Important**: The y-axis shows the distribution among **interpretable tokens only**. Non-interpretable tokens are excluded from the denominator, so the three categories always sum to 100%.

## Examples

### Single model visualization
```bash
# Visualize Olmo-7B + CLIP ViT
python scripts/analysis/layer_evolution/visualize_interpretation_types.py \
    --llm olmo-7b --vision-encoder vit-l-14-336

# Visualize Qwen2-7B + CLIP ViT
python scripts/analysis/layer_evolution/visualize_interpretation_types.py \
    --llm qwen2-7b --vision-encoder vit-l-14-336
```

### Compare multiple models
```bash
# Compare Olmo vs Qwen2 (both with CLIP ViT)
python scripts/analysis/layer_evolution/compare_interpretation_types.py \
    --models "olmo-7b:vit-l-14-336" "qwen2-7b:vit-l-14-336"

# Compare different vision encoders for Olmo
python scripts/analysis/layer_evolution/compare_interpretation_types.py \
    --models "olmo-7b:vit-l-14-336" "olmo-7b:siglip" "olmo-7b:dinov2-large-336"
```

## Key Findings

Based on the visualizations, we observe:

1. **Olmo-7B + CLIP ViT**: Shows high concrete interpretability (~70%) even at layer 0, which remains relatively stable across layers with a slight increase in middle layers (up to ~81% at layer 16)

2. **Qwen2-7B + CLIP ViT**: Starts with much lower interpretability at layer 0 (~52% concrete, 27% abstract) but quickly aligns by layer 1-2, reaching ~78-83% concrete

3. **Evolution pattern**: Concrete interpretability tends to increase in early-to-middle layers, while global interpretability decreases (tokens become more localized)

4. **Abstract concepts**: Abstract interpretability is relatively stable across layers (10-26%), showing that LLMs maintain some level of conceptual abstraction throughout processing

## Data Sources

These scripts consume JSON results produced by:
- `llm_judge/run_single_model_with_viz_contextual.py` (for contextual layers 1+)
- `llm_judge/run_single_model_with_viz.py` (for layer 0, nearest neighbors)

The LLM judge evaluates each vision token's top-5 nearest neighbors and categorizes them as concrete, abstract, or global based on their relationship to the image region.

## Directory Structure

```
scripts/analysis/layer_evolution/
├── visualize_interpretation_types.py    # Single model visualization
├── compare_interpretation_types.py      # Multi-model comparison
├── generate_all_visualizations.sh       # Batch generation script
└── README.md                             # This file

analysis_results/layer_evolution/
├── interpretation_types_{llm}_{encoder}.pdf    # Individual model plots (PDF)
├── interpretation_types_{llm}_{encoder}.png    # Individual model plots (PNG)
├── compare_interpretation_types_*.pdf          # Comparison plots (PDF)
└── compare_interpretation_types_*.png          # Comparison plots (PNG)
```

**Input data sources:**
- `analysis_results/llm_judge_contextual_nn/` - Contextual layer results (layers 1+)
- `analysis_results/llm_judge_nearest_neighbors/` - Layer 0 results (static nearest neighbors)

