# Ablations and Qwen2-VL Plotting

## Overview

This directory now includes support for plotting ablation studies and Qwen2-VL (off-the-shelf model) results.

## Workflow

### 1. Extract Data

Run `update_data.py` to extract all data (main models + ablations + Qwen2-VL):

```bash
python update_data.py --skip-alignment
```

This will:
- Extract main 9 model results (3 LLMs × 3 vision encoders)
- Extract ablation results from `analysis_results/llm_judge_*/ablations/`
- Extract Qwen2-VL results from `analysis_results/llm_judge_*/qwen2-vl/`
- Save everything to `data.json`

### 2. Generate Plots

#### Ablations

Generate all ablation plots:

```bash
python create_ablations_plots.py
```

Options:
- `--mega-only` - Only create mega plot (all ablations vs baseline)
- `--grouped-only` - Only create grouped plots (one per ablation type)

Output: `paper_figures_output/ablations/`

**Ablation Models:**
1. **Baseline:** olmo-7b + vit-l-14-336 (default settings)
2. **Caption Style:** first-sentence (train on first sentence only)
3. **ViT Layers:** earlier-vit-layers-6, earlier-vit-layers-10
4. **Connector:** linear (vs MLP baseline)
5. **Seeds:** seed10, seed11 (different random seeds)
6. **LLM Frozen:** unfreeze, unfreeze-llm (fine-tune LLM)
7. **Task:** topbottom (frozen), topbottom_unfreeze-llm (different task)

#### Qwen2-VL

Generate Qwen2-VL plots:

```bash
python create_qwen2vl_plots.py
```

Options:
- `--unified-only` - Only create unified 3-panel plot
- `--individual-only` - Only create individual plots per method

Output: `paper_figures_output/qwen2vl/`

### 3. Add to Notebook

The notebook `paper_figures.ipynb` should have two new sections:

**Section: Ablations**
- Load ablation data from `data.json`
- Create mega plot (baseline + all ablations)
- Create grouped plots (one per ablation type)
- Show interpretability trends

**Section: Qwen2-VL**
- Load Qwen2-VL data from `data.json`
- Create unified 3-panel plot (NN, LogitLens, Contextual)
- Show individual method plots
- Compare to main models if needed

## Data Structure

### data.json

```json
{
  "nn": {...},              // Main 9 models NN results
  "logitlens": {...},       // Main 9 models LogitLens results
  "contextual": {...},      // Main 9 models Contextual results
  "ablations": {
    "nn": {...},            // Ablation NN results by model name
    "logitlens": {...},     // Ablation LogitLens results
    "contextual": {...}     // Ablation Contextual results
  },
  "qwen2vl": {
    "nn": {...},            // Qwen2-VL NN results by layer
    "logitlens": {...},     // Qwen2-VL LogitLens results
    "contextual": {...}     // Qwen2-VL Contextual results
  }
}
```

### Ablation Model Naming

Model names in `ablations` dict:
- `olmo-7b_vit-l-14-336` - baseline
- `first-sentence_olmo-7b_vit-l-14-336` - caption ablation
- `olmo-7b_vit-l-14-336_earlier-vit-layers-6` - ViT layer ablation
- `olmo-7b_vit-l-14-336_linear` - connector ablation
- `olmo-7b_vit-l-14-336_seed10` - seed ablation
- `olmo-7b_vit-l-14-336_unfreeze` - LLM frozen ablation
- `train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336` - task ablation

## Expected Layers

All ablations use OLMo-7B (32 layers):
- Layers: **0, 1, 2, 4, 8, 16, 24, 30, 31** (9 layers)

Qwen2-VL (28 layers):
- Layers: **0, 1, 2, 4, 8, 16, 24, 26, 27** (9 layers)

## Notes

- Ablations are generated from `run_all_missing.sh`
- Qwen2-VL results are generated in Phases 7-11 of `run_all_missing.sh`
- If results aren't complete, plots will only show available data
- TopBottom ablations may have incomplete data if still running
