# V-Lens Paper Figures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BennoKrojer/molmo/blob/interp/paper_plots/paper_figures.ipynb)

All data and plotting code for the paper figures. Self-contained, no external dependencies on large files.

## Quick Start

**Option 1: Click the Colab badge above** (easiest)

**Option 2: Run locally**
```bash
pip install matplotlib seaborn numpy
python create_lineplot_unified.py
```

## Files

- `paper_figures.ipynb` - Full notebook with all plots and data tables
- `create_lineplot_unified.py` - Main figure generation script (reads from raw analysis_results)
- `data.json` - **Single source of truth** for all interpretability numbers
- `paper_figures_output/` - Generated PDFs/PNGs

**WARNING:** Never create scripts with hardcoded data values. Always read from `data.json` or raw `analysis_results/`.

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

**Central data file: `data.json`**

This is the single source of truth for all interpretability results. Structure:
```json
{
  "nn": {                           // Input Embedding Matrix method
    "llama3-8b+dinov2-large-336": {
      "0": 20.33,                   // Layer 0: 20.33% interpretable
      "1": 16.83,
      ...
    },
    ...                             // All 9 model combinations
  },
  "logitlens": { ... },             // LogitLens method
  "contextual": { ... }             // LatentLens method
}
```

**Layers stored:** `[0, 1, 2, 4, 8, 16, 24, 26/30, 27/31]` (26/27 for Qwen2, 30/31 for OLMo/LLaMA)

**To update data.json:**
```bash
python update_data.py
```

This extracts fresh results from `analysis_results/llm_judge_*/` and regenerates all plots.

