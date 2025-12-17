# V-Lens Paper Figures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BennoKrojer/molmo/blob/interp/paper_plots/paper_figures.ipynb)

All data and plotting code for the paper figures. Self-contained, no external dependencies on large files.

## Quick Start

**Option 1: Click the Colab badge above** (easiest)

**Option 2: Run locally**
```bash
pip install matplotlib seaborn numpy
python paper_figures_standalone.py
```

## Files

- `paper_figures.ipynb` - Full notebook with all plots and data tables
- `paper_figures_standalone.py` - Minimal script (~100 lines)
- `paper_figures_output/` - Generated PDFs/PNGs

## Updating Data

When results change, run:
```bash
cd paper_plots
python update_data.py
```

This will:
1. Extract fresh data from `analysis_results/llm_judge_*/`
2. Print the new data (copy to notebook if needed)
3. Regenerate the plots

For just extracting data without regenerating:
```bash
python update_data.py --extract-only
```

## Data

All interpretability data (LLM Judge evaluated) is embedded directly in the notebook/script. No need for external files.

