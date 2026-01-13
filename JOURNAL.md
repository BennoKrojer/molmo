# V-Lens Development Journal

A concise log of major changes, results, and git operations.

---

## 2026-01

### 2026-01-13 (Sunburst chart - UPPERCASE target words)

**Improved visibility of target words in phrases:**
- Changed from `<b>word</b>` HTML bold to UPPERCASE (e.g., "...the FOREGROUND")
- Bold was too subtle at small font sizes in radial text orientation
- Multiple unique phrases now merged into one segment with `<br>` newlines
- Shows up to 3 different phrases per word

**Git:** Pushed to final (d4961fa), paper repo updated (8e3e16f)

### 2026-01-13 (Sunburst chart fixes - Plotly)

**Switched to Plotly for proper radial text:**
- `insidetextorientation='radial'` handles text orientation automatically
- No more manual rotation calculations
- Script: `scripts/analysis/layer_evolution/visualize_sunburst_interpretation_types.py`

**All requirements now met:**
1. ✓ Wedges proportional to actual word counts
2. ✓ Multiple example phrases per word (up to 3)
3. ✓ Preceding context format: "...context WORD" (UPPERCASE target)
4. ✓ Radial text orientation (Plotly auto-handles, never upside down)

**Git:** Pushed to final (a7b67be, ba5110a), paper repo updated (1ecfb2e)

### 2026-01-13 (Interpretation types visualization - ablations bug fix + paper update)

**Critical bug fixed in interpretation types scripts:**
- `visualize_interpretation_types.py` and `compare_interpretation_types.py`
- Glob pattern `**/results_*.json` was picking up ablation files
- For olmo-7b+vit-l-14-336: 90 files loaded instead of 9 (1 main + 9 ablations × 9 layers)
- Added `/ablations/` exclusion per CLAUDE.md rule
- Verified: visual_attributes and pos_tags scripts use direct globs (no `**`), not affected

**Regenerated plots:**
- `analysis_results/layer_evolution/interpretation_types_combined.pdf` (3x3 grid)
- `analysis_results/layer_evolution/interpretation_types_average.pdf`
- `analysis_results/layer_evolution/interpretation_types_qwen2vl.pdf` (new)

**Paper appendix updated (`sec:evolution_details`):**
- Added 3x3 interpretation types figure with caption
- Added Qwen2-VL interpretation types figure
- Added explanatory text for concrete/abstract/global categories
- Key finding: concrete dominates (70-75%), stable across layers

**Numerical data summary (averages across 9 models):**
- Concrete: 72-75% (stable across layers)
- Abstract: 12-15%
- Global: 11-13%
- SigLIP shows higher global (20-30%) vs CLIP/DINOv2

**CLAUDE.md update:**
- Added rule to explicitly confirm reading CLAUDE.md at start of each chat

**Sunburst chart created (Plotly):**
- 3-ring nested pie chart showing: Type → Top Words → Visual Genome Phrases
- Inner: Concrete 65%, Abstract 19%, Global 16%
- Middle: Top 5 words per type (colors dominate Concrete, positional terms for Abstract)
- Outer: Example phrase contexts from Visual Genome corpus
- Uses Plotly's `insidetextorientation='radial'` for automatic radial text
- Data saved to `analysis_results/layer_evolution/sunburst_data.pkl`
- Figure: `paper_plots/paper_figures_output/interpretability/sunburst_interpretation_types.pdf`

**Git:** Pushed to final (d089cce, earlier: a50889d, 806fdac, 48fa5ec, d6c7c6c, 2342c45, d76ee10)

### 2026-01-12 (Phrase annotation examples - LaTeX approach)

**Refactored appendix phrase examples:**
- Replaced matplotlib-generated PDFs with pure LaTeX approach
- Images: Just preprocessed vision encoder input with red bbox
- Text: Native LaTeX with `\colorbox{yellow!50}{\textbf{word}}` highlighting
- Script: `scripts/analysis/generate_phrase_examples_latex.py`

**Vision encoder preprocessing (high-res for paper):**
- CLIP: resize + pad (preserves aspect ratio), black padding
- SigLIP/DINOv2: squash to square
- 3x resolution (1008×1008 or 1152×1152) for print quality
- 300 DPI PDF output

**Layout:**
- Full image + 5×5 crop side by side, left-aligned
- Main image: 3.2cm height, crop: 2.0cm height
- Label: "Random phrase, same token:" (merged from Overleaf)

**Fixed:** Typo in background.tex (`\Sigme` → `\Sigma`)

**Git:** Multiple commits to paper submodule and main repo (final branch)

### 2026-01-10 (CLAUDE.md consolidation + demo regeneration)

**CLAUDE.md consolidation:** Reduced from 413 → 171 lines (59% reduction)
- Added "At the Start of Every Chat" section listing required files to read
- Consolidated 18+ sections into 7 clear sections
- Promoted "THREE RULES" prominently: 1) Edit never rewrite, 2) Validate data before plotting, 3) Commit immediately
- Moved detailed patterns to README.md where appropriate

**Demo viewer regeneration:**
- Ran `create_unified_viewer.py` with 10 images
- All 9 model combinations + 10 ablations included
- Synced to `website/vlm_interp_demo/` and pushed

**Git:** Pushed CLAUDE.md changes (b874200), website updated (da652b5)

### 2026-01-10 (Fix: LN-Lens figure data - ablations exclusion)

**Root cause identified:** The glob pattern `**/results_*.json` in `create_lineplot_unified.py` was picking up files from the `ablations/` subdirectory in `llm_judge_contextual_nn/`. This caused incorrect data to be loaded for the main LN-Lens figure (e.g., layer 16 showed 42% instead of 70% for OLMo+ViT).

**Fix:** Added filter to exclude `/ablations/` paths in all three loading functions:
- `load_nn_results()`
- `load_logitlens_results()`
- `load_contextual_results()`

**Verification:** Layer 16 for olmo-7b + vit-l-14-336 now correctly shows 70% (from main results file) instead of 42% (from ablation variant).

**Git:** Pushed to `final` branch (b9291f7) and paper submodule (01f0fa4)

### 2026-01-10 (Terminology Standardization for ICML)

**Comprehensive terminology update across codebase:**
- "Static NN" / "Static V-Lens" → "Input Embedding Matrix" or "Input Emb."
- "Contextual NN" / "Contextual V-Lens" → "LN-Lens"

**Files updated:**

*Paper (already pushed earlier):*
- sections/4_experiments.tex: Table headers, figure captions, inline text
- sections/1_intro.tex: Method descriptions
- sections/appendix.tex: Section titles
- sections/6_causal.tex: Comments

*Python scripts:*
- paper_plots/create_ablations_plots.py: Method labels
- paper_plots/create_qwen2vl_plots.py: Method labels
- paper_plots/create_ln_lens_comparison.py: Docstrings
- paper_plots/paper_figures_standalone.py: Header/prints
- scripts/analysis/generate_ablation_viewers.py: HTML labels

*Documentation:*
- README.md: Method descriptions, section titles
- CLAUDE.md: Method descriptions, directory comments
- paper_plots/paper_figures.ipynb: Labels and titles

*Regenerated:*
- All ablation plots with updated labels (mega + grouped)

**Git:** Pushed to `final` branch (commits 938c16a, eadb32d) and paper submodule (commit 074c6fd)

### 2026-01-10 (Paper: Update main results figure layout)

**Updated main results figure (fig1_unified_interpretability):**
- Shared y-axis label (only on leftmost plot)
- New titles: "(a) Input Embedding Matrix", "(b) Output Embedding Matrix (LogitLens)", "(c) LN-Lens (Ours)"
- Order: baselines first, our method last
- Shared legend at bottom

**Git:** Pushed to paper submodule and main repo

### 2026-01-10 (Paper: Combine Qwen2-VL figures into subfigures)

**Combined two Qwen2-VL figures into one with subfigures:**
- Layer alignment + token drift now side by side (0.48\linewidth each)
- Saves vertical space in the ablation subsection
- Updated text reference to use combined `fig:qwen2vl_analysis`

**Git:** Pushed to paper submodule and updated main repo

### 2026-01-10 (Qwen2-VL token similarity plot: fixed to match 3x3 exactly)

**Fixed Qwen2-VL token similarity plot to match 3x3 style:**
- Added text token similarity script: `scripts/analysis/qwen2_vl/sameToken_acrossLayers_text_similarity.py`
- Re-ran vision similarity with ALL 27 layers (was only 8)
- Re-ran text similarity with ALL 27 layers
- Updated plotting code to match 3x3 exactly:
  - Same colors: Vision=#2E86AB, Text=#A23B72
  - Same markers: Vision='o', Text='s'
  - Same line style: linewidth=2.5, markersize=8, alpha=0.8
  - Same y-axis label format with newline
  - Same fontsize=15 (no bold) on axis labels

**Interesting finding:** Qwen2-VL text tokens start with very LOW similarity to layer 0 (~0.15 at layer 1)
compared to frozen LLM models where text tokens stay close to original (~1.0 for Llama3, ~0.5-0.6 for OLMo).
This confirms Qwen2-VL's LLM is finetuned, not frozen.

**Added CLAUDE.md rule:** Explicit adherence confirmation before reporting changes.

**Git:** Pushed to `final` branch and paper submodule

### 2026-01-10 (Paper: Resolve Overleaf merge conflict)

**Resolved merge conflict in paper submodule:**
- Conflict in `sections/4_experiments.tex` between HEAD and Overleaf branch
- Kept HEAD version with correct single-plot caption for Qwen2-VL figure
- Overleaf had old (a), (b), (c) format with TODO that was already implemented

**Git:**
- Pushed merge commit to paper repo
- Updated paper submodule reference in main repo

### 2026-01-09 (Qwen2-VL: Complete Analysis with Layer Alignment and Token Similarity)

**Added complete Qwen2-VL analysis (matching sec:which_layer for main models):**

**New analysis script:**
- `scripts/analysis/qwen2_vl/sameToken_acrossLayers_similarity.py` - Token drift analysis

**Updated plotting scripts:**
- `paper_plots/update_data.py` - Added `load_qwen2vl_layer_alignment_data()` and `load_qwen2vl_token_similarity_data()`
- `paper_plots/create_qwen2vl_plots.py` - Changed to single combined lineplot (not 3 subplots)
- `paper_plots/create_layer_alignment_heatmaps.py` - Added `create_qwen2vl_heatmap()`
- `paper_plots/create_token_similarity_plots.py` - Added `create_qwen2vl_plot()`

**New figures:**
- `paper_figures_output/qwen2vl/qwen2vl_unified.pdf` - Single lineplot with 3 methods
- `paper_figures_output/qwen2vl/qwen2vl_layer_alignment_heatmap.pdf` - Layer alignment
- `paper_figures_output/qwen2vl/qwen2vl_token_similarity.pdf` - Token drift plot

**Paper updates:**
- `paper/sections/4_experiments.tex` - Added layer alignment and token similarity figures/discussion
- `paper/figures/fig_qwen2vl_layer_alignment.pdf` - New figure
- `paper/figures/fig_qwen2vl_token_similarity.pdf` - New figure

**Key findings for Qwen2-VL:**
- Layer alignment: Input vision tokens align most to mid-layer (layer 4) contextual embeddings
- Token drift: Similarity drops from 0.96 (layer 1) to 0.10 (layer 27) - more drift than frozen LLMs
- Interpretability: LN-Lens (60-73%) >> LogitLens (max 53%) >> Static NN (9-26%)

**Git:**
- Pushed to `final` branch and paper submodule

### 2026-01-09 (Paper: Qwen2-VL Section + Human Validation Updates)

**Added Qwen2-VL subsection to paper (section 4.5):**
- New figure `figures/fig_qwen2vl.pdf`: 3-panel plot showing Static NN, LogitLens, Contextual interpretability
- Results consistent with main paper methodology (100 patches, same LLM judge)
- Key findings: LN-Lens works well (60-73%) even on finetuned LLMs, Static NN very low (9-26%)
- Added cross-reference from section 4.3 (layer evolution discussion)
- Added `wang2024qwen2vl` citation to bibliography

**Updated human validation numbers:**
- Changed appendix numbers from old (n=323, κ=0.65) to final verified (n=360, κ=0.66, accuracy=84.7%)
- Updated per-model breakdown table with correct numbers
- Clarified methodology: majority vote criterion (≥50% of annotators)

**Files changed:**
- `paper/sections/4_experiments.tex`: Added Qwen2-VL subsection
- `paper/sections/appendix.tex`: Updated human validation numbers and table
- `paper/literature.bib`: Added Qwen2-VL citation
- `paper/figures/fig_qwen2vl.pdf`: New figure

### 2026-01-08 (Paper: LaTeX Tabular for L2 Norm Figures - FINAL FIX)

**Replaced overlay approach with proper LaTeX tabular:**
- Previous overlay approach produced misaligned borders (pixel estimates were wrong)
- New approach: 18 individual PDFs (9 L2 norm + 9 max token) arranged in LaTeX tabular
- `\begin{tabular}{|c|c|c|}` with `\hline` provides clean, native gridlines
- Individual plots in `figures/l2norm_individual/` and `figures/max_token_individual/`
- `generate_individual_max_token_plots.py` - Creates 9 individual max token PDFs

**Why this is better:**
- LaTeX handles borders natively (no pixel guessing)
- Individual files can be replaced/updated independently
- Clean separation of content (plots) and presentation (tabular)

**Lesson learned:** For grid figures, use LaTeX tabular with individual files instead of trying to overlay borders on combined images.

### 2026-01-08 (Paper: L2 Norm Analysis Appendix)

**Added new appendix section `app:outliers` to paper:**
- `sections/appendix.tex`: New section "L2 Norm Analysis of Vision Tokens" with two subsections
- `figures/l2norm_vision_text_distributions.pdf`: 3x6 grid showing Vision|Text L2 norm histograms
- `figures/max_token_embedding_distributions.pdf`: 3x3 grid showing embedding dimension distributions
- `sections/3_method.tex`: Updated figure caption to reference `\Cref{app:outliers}`

**Key findings documented in paper:**
- Vision tokens have 1-2 orders of magnitude larger L2 norms than text tokens
- OLMo-7B has ~100x smaller embedding scale than LLaMA3/Qwen2
- High L2 norms from uniform scaling, not sparse outliers
- DINOv2 consistently produces largest L2 norms

### 2026-01-08 (Notebook Update: L2 Norm Plots)

**Added two new plot sections to `paper_plots/paper_figures.ipynb`:**
1. **L2 Norm Distribution (3x6 grid)**: Vision vs Text histograms with log scale, p99 and max markers
2. **Max Token Embedding Values (3x3 grid)**: Individual embedding dimension distributions

Both plots load pre-computed data from `analysis_results/` for fast execution (no GPU needed).
Plots use 100% identical logic to standalone scripts for reproducibility.

### 2026-01-08 (CLAUDE.md Guidelines Update)

**Added new guidelines to prevent future mistakes:**
- TLDR updated: Re-read CLAUDE.md and JOURNAL.md during long conversations before significant changes
- New section "INCREMENTAL CHANGES": Find exact code, make minimal changes, git push often
- Emphasis on viewing visual outputs before modifying them

**Lesson learned**: When asked for small plot changes, don't rewrite - find exact original code and modify minimally.

### 2026-01-08 (Max L2 Norm Token Embedding Dimension Analysis)

**New analysis**: Investigating whether high L2 norms are driven by few large embedding dimensions or uniformly larger values across all dimensions.

**New scripts created**:
- `scripts/analysis/extract_max_token_embeddings.py` - Extracts full embedding vector of max L2 norm vision token
- `run_extract_max_embeddings.sh` - Parallel runner for all 9 models on 8 GPUs

**Key findings**:
- **OLMo-7B has ~100-150x smaller embedding scale** than Llama3/Qwen2 (std ~10-20 vs ~1700-11000)
- **All distributions are Gaussian-like** - high L2 norms come from uniformly larger values across ALL 3584-4096 dimensions, NOT sparse outliers
- **DINOv2 consistently produces the largest scale** across all LLMs
- **Llama3-8B max tokens occur at layer 0** (input), while OLMo/Qwen max tokens occur at **layer 24** (late)

| Model | Vision Encoder | Max L2 | Layer | Std Dev |
|-------|---------------|--------|-------|---------|
| OLMo-7B | CLIP | 743 | 24 | 11.60 |
| OLMo-7B | DINOv2 | 1,011 | 24 | 15.80 |
| OLMo-7B | SigLIP | 1,289 | 24 | 20.14 |
| Llama3-8B | CLIP | 109,908 | 0 | 1,716.72 |
| Llama3-8B | DINOv2 | 721,474 | 0 | 11,267.36 |
| Llama3-8B | SigLIP | 196,945 | 0 | 3,076.76 |
| Qwen2-7B | CLIP | 163,280 | 24 | 2,726.58 |
| Qwen2-7B | DINOv2 | 475,051 | 24 | 7,933.76 |
| Qwen2-7B | SigLIP | 231,242 | 24 | 3,862.31 |

**Output**: `paper_plots/paper_figures_output/l2norm_plots/max_token_embedding_values_3x3.png`

### 2026-01-08 (L2 Norm Analysis of Vision vs Text Tokens)

**New analysis**: Measuring L2 norm of vision and text tokens across LLM layers to understand embedding magnitude differences.

**New scripts created**:
- `scripts/analysis/sameToken_acrossLayers_l2norm.py` - Vision token L2 norm across layers
- `scripts/analysis/sameToken_acrossLayers_text_l2norm.py` - Text token L2 norm across layers
- `run_parallel_sameToken_l2norm.sh` - Run both on all 9 model combinations
- `paper_plots/create_l2norm_plots.py` - Generate histogram visualizations

**Key findings**:
- OLMo-7B: Vision (~30-55) and text (~65) L2 norms are comparable
- Llama3-8B: Vision (~620) >> text (~43) - order of magnitude larger
- Qwen2-7B: Vision (~4000+) >>> text (~100-300) - massive gap

**Output**: `paper_plots/paper_figures_output/l2norm_plots/`
- 3x3 grid plot (PNG + PDF) with log scale
- 9 individual model plots

**Visualization**: Yellow→orange→red color scheme for layers (0,4,8,16,24,31), solid bars for vision, dashed for text.

### 2026-01-06 (Patchscopes Integration into Demo Viewer)

**Full integration of Patchscopes as 4th interpretability method in the unified viewer.**

**Changes to `patchscopes_descriptive.py`**:
- Added `--num-patches` argument (default: 10 random patches per image, matching lite viewer)
- Added `--lite-suffix` argument for output directory naming (`_lite10`)
- Added `--skip-html` flag for batch runs
- Changed default layers to `0,2,4,8,16` (optimal for patchscopes)
- Output format now matches logitlens/NN structure: `chunks → patches`
- Each patch has `description` field (single string) instead of top-5

**New file `run_parallel_patchscopes.sh`**:
- Same pattern as `run_parallel_contextual_nn.sh`
- Launches up to 8 independent GPU jobs for 9 model combinations
- Uses staggered launches with model loading detection
- Outputs to `analysis_results/patchscopes/`

**Changes to `create_unified_viewer.py`**:
- Added patchscopes scanning in `scan_analysis_results()`
- Added patchscopes loading in `load_all_analysis_data()`
- Updated CSS: 4-column grid, purple color for patchscopes
- Updated HTML column order: **LN-Lens, Logit Lens, Embedding Matrix, Patchscopes**
- Added JavaScript rendering for patchscopes descriptions

**Files modified**:
- `scripts/analysis/patchscopes/patchscopes_descriptive.py`
- `scripts/analysis/create_unified_viewer.py`
- `run_parallel_patchscopes.sh` (new)

**Next steps**: Run patchscopes on all 9 models, regenerate demo viewer

### 2026-01-06 (SYSTEMIC FIX: create_unified_viewer.py now includes ablations automatically)
- **USER REPORT**: "ablations are missing from the demo - please make sure things work"
- **ROOT CAUSE ANALYSIS**:
  - Fragmented workflow: needed to run 3 scripts (`create_unified_viewer.py` → `generate_ablation_viewers.py` → `add_models_to_viewer.py`)
  - Easy to forget steps → incomplete index.html produced
  - `generate_demo.sh` existed but wasn't obvious / user ran Python directly
- **SYSTEMIC FIX**: Modified `create_unified_viewer.py` to handle ablations automatically:
  - Added `--no-ablations` flag (ablations included BY DEFAULT)
  - Scans for existing ablation viewers in `output-dir/ablations/`
  - Adds ablation section to index.html if viewers found
  - Now ONE command produces COMPLETE viewer
- **FILES CHANGED**:
  - `scripts/analysis/create_unified_viewer.py` - Added ablation handling (+130 lines)
  - `generate_demo.sh` - Simplified to single command (was 3 steps)
  - `README.md` - Updated documentation
  - `CLAUDE.md` - Added "VIEWER GENERATION - SINGLE COMMAND" section
- **NEW WORKFLOW** (either works):
  ```bash
  ./generate_demo.sh --num-images 10
  # OR
  python scripts/analysis/create_unified_viewer.py --output-dir ... --num-images 10
  ```
- **Git**: `ad47308` - Make create_unified_viewer.py include ablations automatically
- **Previous quick fix**: `4a90b91` (website), `b38f48f`, `93bbd6b` (molmo) - manually restored ablations

### 2026-01-06 (Patchscopes Bug Fix: Persistent Hook)

**Critical bug fixed in `patchscopes_descriptive.py`**: Hook was only applied on first forward pass during autoregressive generation, causing all subsequent tokens to see original "X" instead of patched visual hidden state.

**Bug location**: The hook was removed after the first token generation (line 124), but autoregressive generation passes the FULL sequence each time, so subsequent passes re-processed the original X without patching.

**Fix**:
1. Removed `call_count == 0` check from `hook_fn` - now patches on EVERY forward pass
2. Keep hook active throughout ALL generation, only remove at the very end

**Results after fix**:
- Text embeddings work correctly: "dog" → "type of animal", "Paris" → "capital of France"
- Visual tokens now generate **206 unique descriptions** (vs ALL "letter of alphabet" before)
- Spatial coherence observed: church-related descriptions cluster where church tower is in image
- Layer 0 shows most semantic diversity; later layers converge toward "letter X" interpretation

**Files modified**: `scripts/analysis/patchscopes/patchscopes_descriptive.py`
**Also fixed**: Missing comma in `pyproject.toml` causing pip install failure

---

### 2026-01-06 (Paper Review: Concrete Additions & Placeholder Fixes)

Claude overnight audit of paper. All values grounded in repo data.

---

#### 1. PLACEHOLDER FIXES (copy-paste ready)

**Section 4, Line 14 (PixMo-Cap description) - EMPIRICALLY VERIFIED:**
```latex
% OLD: around 700K captions with a very high level of detail and quality, with the average caption containing N sentences.
% NEW (from running code on actual dataset):
717K image-caption pairs (716K unique images) with detailed captions averaging 167 words and 9 sentences each.
```
Empirical stats (computed via `load_dataset('allenai/pixmo-cap')`):
- Total rows: 717,042
- Unique images: 716,551
- Avg words: 167.4 (std: 50.3), median: 160
- Avg sentences: 8.8 (std: 3.8), median: 8

**Section 3, Line 26 (Visual Genome corpus) - EMPIRICALLY VERIFIED:**
```latex
% OLD: Visual Genome with N unique captions
% NEW (from analyzing vg_phrases.txt):
Visual Genome with 2.99M phrase-region annotations (2,991,848 unique after deduplication)
```
Empirical stats (computed on vg_phrases.txt):
- Total lines: 2,992,117
- Non-empty: 2,992,111
- Unique (raw): 2,991,848 (263 exact duplicates)
- Unique (normalized, no punct): 2,876,652

**Section 4, Line 31-32 (Judge Design paragraph):**
```latex
\paragraph{Judge Design.}
We use GPT-4o as our automatic judge. Given an image with a red bounding box highlighting the vision token's receptive field, and the top-5 nearest neighbors, we prompt the judge to classify each word as \textit{concrete} (directly visible), \textit{abstract} (conceptually related), or \textit{global} (present elsewhere in image). A token is marked interpretable if at least one of the top-5 neighbors falls into any category. The full prompt is in \Cref{app:evaluation}.
```

---

#### 2. HUMAN VALIDATION RESULTS (Section 3, Line 63-64)

```latex
% Replace: Human validation: \textcolor{red}{todo}
% With:
We validate the LLM judge against human annotations on 360 vision tokens (60 images $\times$ 6 tokens each, annotated by 4 authors).
For Static NN at layer 0, we find strong agreement: Spearman $\rho = 0.65$, Cohen's $\kappa = 0.65$, and accuracy = 84.2\% (n=323 matched instances).
```

Per-model breakdown (for appendix):
| Model | n | Spearman ρ | Cohen's κ | Accuracy |
|-------|---|-----------|-----------|----------|
| OLMo + DINOv2 | 38 | 0.90 | 0.89 | 94.7% |
| OLMo + ViT-L/14 | 34 | 0.83 | 0.82 | 91.2% |
| OLMo + SigLIP | 35 | 0.71 | 0.67 | 82.9% |
| Qwen2 + SigLIP | 33 | 0.80 | 0.78 | 97.0% |
| LLaMA3 + ViT-L/14 | 45 | 0.65 | 0.64 | 82.2% |
| LLaMA3 + DINOv2 | 37 | 0.21 | 0.21 | 73.0% |
| Qwen2 + ViT-L/14 | 38 | 0.70 | 0.66 | 89.5% |

---

#### 3. ABLATION TABLE VALUES (Table 1)

All values from `data.json`. Baseline = OLMo-7B + ViT-L/14.

| Model Variant | NN Overlap (all/high-sim) | Static NN L0 | LN-Lens L0 | Caption Score |
|---------------|---------------------------|--------------|------------|---------------|
| **Baseline** | --- | 55.0% | 71.0% | 6.60 |
| Different seed (seed10) | 13.6 / 45.4 | 52.7% | 68.3% | 6.68 |
| Different seed (seed11) | --- | 59.7% | 68.5% | --- |
| Linear connector | 11.8 / 44.9 | 54.7% | 70.0% | 6.24 |
| First-sentence captions | 10.5 / 42.7 | 48.0% | 62.3% | 6.81 |
| Unfrozen LLM | 11.8 / 44.6 | 67.7% | 65.3% | 7.12 |
| ViT layer 6 | --- | 59.7% | 72.7% | --- |
| ViT layer 10 | --- | 50.3% | 72.2% | --- |
| TopBottom (frozen) | 0.0 / --- | 10.3% | 40.0% | (75.7% task acc) |
| TopBottom (unfrozen) | 0.1 / --- | 22.0% | 40.0% | (86.7% task acc) |

---

#### 4. QWEN2-VL COMPARISON NUMBERS

Off-the-shelf Qwen2-VL-7B-Instruct (pretrained, not our training):

| Layer | Static NN | LogitLens | LN-Lens |
|-------|-----------|-----------|---------|
| 0 | 26% | 12% | 63% |
| 1 | 24% | 13% | 63% |
| 16 | 16% | 5% | 62% |
| 24 | 15% | 15% | 64% |
| 26 | 9% | 41% | 73% |
| 27 | 14% | 53% | 60% |

Key insight: Even off-the-shelf models show high LN-Lens interpretability (63%+ at L0).

---

#### 5. V-LENS CORPUS DETAILS (for Appendix)

```latex
\subsection{Contextual Embedding Corpus}
\label{app:vlens-design}

For contextual \vlens, we extract embeddings from Visual Genome~\citep{krishna2017visual} phrase-region annotations.
We process 2,992,111 unique phrases through each LLM, storing up to 20 contextual embeddings per vocabulary token at layers [1, 2, 4, 8, 16, 24, N-2, N-1].
This results in approximately 2.5M embeddings across 26,862 unique tokens per layer.
To reduce storage, embeddings are stored in float8 format (25\% of fp32 size).
We provide embeddings for OLMo-7B, LLaMA3-8B, Qwen2-7B, and Qwen2-VL-7B-Instruct.
```

---

#### 6. LLM JUDGE PROMPT (for Appendix)

The full prompt used (from `llm_judge/prompts.py`, IMAGE_PROMPT_WITH_CROP version):

```
You are a visual interpretation expert specializing in connecting textual concepts
to specific image regions. Your task is to analyze a list of candidate words and
determine how strongly each one relates to the content of the highlighted region.

### **Inputs**
1. **Full Image**: An image containing a red bounding box highlighting the region of interest.
2. **Cropped Region**: A close-up view of the exact region highlighted by the red bounding box.
3. **Candidate Words**: A list of words to evaluate. Here are the candidate words:
    - {candidate_words}

### **Evaluation Guidelines**
There are three types of relationships to consider:

* **Concrete**: A word is **concretely related** if it names something literally visible
  in the cropped region: objects, colors, textures, text, shapes.

* **Abstract**: A word is **abstractly related** if it describes concepts, emotions,
  or activities related to but not literally visible: cultural concepts, emotions, functions.

* **Global**: A word is **globally related** if it describes something present elsewhere
  in the full image but not in the highlighted region.

### **Output Format**
Return a single JSON object with fields:
{
    "reasoning": "initial reasoning...",
    "interpretable": true/false,
    "concrete_words": [...],
    "abstract_words": [...],
    "global_words": [...]
}
```

---

#### 7. MAIN RESULTS SUMMARY (for figure captions)

**Layer 0 Interpretability (% tokens interpretable):**

| Model | CLIP ViT | SigLIP | DINOv2 |
|-------|----------|--------|--------|
| **Static NN** |
| OLMo-7B | 55% | 42% | 42% |
| LLaMA3-8B | 35% | 23% | 20% |
| Qwen2-7B | 18% | 5% | 7% |
| **LN-Lens (ours)** |
| OLMo-7B | 71% | 63% | 84% |
| LLaMA3-8B | 82% | 63% | 85% |
| Qwen2-7B | 79% | 66% | 80% |

Key finding: LN-Lens shows 2-10x higher interpretability than Static NN at input layer!

---

#### 8. REMAINING RED/TODO ITEMS (need investigation)

1. **Line 144**: "1-2 sentences on simple baselines eg how often does llm judge trigger on random vectors/NNs?"
   - Need to run random baseline experiment

2. **Line 261**: "TODO: investigate properly" (SigLIP/DINOv2 generic NN behavior)
   - Check why SigLIP has more global/abstract concepts

3. **Line 268**: "TODO: for 50 examples, check what the color NN turn into when they are not color anymore"
   - Manual investigation needed

4. **Line 276**: "check if more evolution is observed in Qwen2-VL (unfrozen)"
   - Qwen2-VL data available; can compare evolution patterns

5. **Appendix LLM judge**: "TODO" - Now filled in above

---

#### 9. CLARIFICATION QUESTIONS

1. **PixMo-Cap sentence count**: The ~200 words/caption corresponds to roughly 10-15 sentences (assuming 15-20 words/sentence). Want exact count from data?

2. **Bimodal distribution correlation (Line 76-77)**: Paper says "We manually inspect data and find a correlation of X". What correlation metric should I compute?

3. **Caption scoring**: The ablation table shows "Caption Score" from LLM judge (1-10 scale). Are these correct values or do you have updated ones?

---

### 2026-01-05 (Patchscopes fixes - text works, vision still fails)
- **Found critical bug in test scripts**: Was checking logits at `patch_position`, should be `-1`
  - Identity prompt `"cat->cat; ?->"` predicts what comes AFTER the final `->"`
- **After fix - TEXT works (80-87% Top-1 at layers 0-16)**
- **Fixed vision script** (`patchscopes_identity.py`):
  - Updated prompt format from `"...?"` to `"...?->"` (paper format)
  - Updated patch_position and logits indexing
- **Vision tokens STILL don't work** - predict identity prompt tokens ("hello", "cat"), not visual content
- **Conclusion**: Patchscopes works for text but not for visual soft prompts
- **Git**: `fcb17b9` - Fix test script, `7b2e7ad` - Fix vision script prompt format

### 2026-01-05 (Patchscopes baseline implementation)
- **Implemented Patchscopes** (Ghandeharioun et al., ICML 2024) as a baseline for comparison
- **Method**: Identity prompt `"cat->cat; 1135->1135; hello->hello; ?"` with l→l patching
  - Patches visual token hidden states into identity prompt at "?" position
  - Lets model's own layers (l through L) process the patched representation
  - Paper claims up to 98% improvement over LogitLens for next-token prediction
- **Implementation** via PyTorch forward pre-hooks on transformer blocks:
  - Hook modifies `args[0]` (hidden states tensor) at specified position
  - Verified: OLMo blocks receive `args = (hidden_states,)` with shape [B, seq, hidden_dim]
- **Comprehensive sanity checks** (all passed):
  - Identity prompt works: predicts "world", "ha", "cherry" correctly
  - Hooks actually called and modify output (logit diff = 2.57)
  - Layer indexing correct: `hidden_states[l]` == input to `blocks[l]`
  - Batch patching works, no memory leaks
- **Key finding**: Patchscopes may NOT be ideal for interpretability!
  - When patching "dog" representation, model predicts format tokens (`->`, `;`) not "dog"
  - Identity prompt context **overwhelms** semantic content of patched representation
  - LogitLens showed MORE interpretable results (`home`, `building`, `brick` vs whitespace)
  - Patchscopes paper evaluated next-token prediction accuracy, not interpretability
- **Files created**:
  - `scripts/analysis/patchscopes/patchscopes_identity.py` - Main analysis script
  - `scripts/analysis/patchscopes/sanity_checks.py` - Core sanity tests
  - `scripts/analysis/patchscopes/additional_checks.py` - Edge case tests
  - `scripts/analysis/patchscopes/test_patchscopes.py` - Validation tests
  - `scripts/analysis/patchscopes/visualize_comparison.py` - HTML comparison generator
  - `run_all_combinations_patchscopes.sh` - Run script for all models
- **Git**: `118783b` - Add Patchscopes baseline implementation with comprehensive tests

### 2026-01-05 (Website submodule for demo hosting)
- **Added `bennokrojer.github.io`** as git submodule at `./website`
- **Synced** updated `unified_viewer_lite/` (with Layer 0 fix) to `website/vlm_interp_demo/`
- **Demo live at**: https://bennokrojer.github.io/vlm_interp_demo/
- **Git (website)**: `20f42af` - Update VLM interp demo with Layer 0 LN-Lens fix
- **Git (molmo)**: `6e7f370` - Add website as submodule for hosting VLM interp demo

### 2026-01-05 (FIX: Missing Layer 0 LN-Lens data in main model viewers)
- **USER REPORT**: "main 3x3 models are missing the layer 0 LN-Lens results"
  - Qwen2-VL and ablations showed Layer 0, but main 9 models did not
- **ROOT CAUSE** in `create_unified_viewer.py` (`load_all_analysis_data()`):
  - Code only loaded ONE visual_layer file (visual_layer=0)
  - But stored results under **contextual layer** keys (1, 2, 4, 8...) instead of **visual layer** key (0)
  - When layer dropdown showed "Layer 0", no LN-Lens data existed at that key!
  - In contrast, `generate_ablation_viewers.py` correctly stored under visual layer keys
- **FIX**: Load ALL visual layer files, each stored under its visual layer key
  - Now loads: `{0: data, 1: data, 2: data, 4: data, 8: data, 16: data, 24: data, 30: data, 31: data}`
  - Matches how ablation viewer handles contextual data
- **MODULARITY**: Added shared functions to `viewer_lib.py`:
  - `find_analysis_files()`: Discover JSON files for NN/LogitLens/Contextual
  - `load_analysis_data_for_type()`: Load data handling Format A/B differences
  - `extract_patches_from_data()`, `get_grid_dimensions()`: Shared patch handling
  - `process_*_patch()`: Process patches into unified format
- **REGENERATED**: `unified_viewer_lite/` for all 9 main models with fix
- **Git**: `0379443` - Fix missing layer 0 LN-Lens data in main model viewers
- **Git**: `589cf2a` - Add shared data loading functions to viewer_lib.py

### 2026-01-05 (Comprehensive ablation table for paper)
- **Verified all paper numbers** against analysis_results/:
  - Captioning score: `eval_captioning_gpt-judge.py` → LLM judge mean (GPT-4o, 1-10 scale)
  - NN Overlap: `scripts/analysis/ablations_comparison.py` → Top-5 NN overlap with baseline
  - Static NN L0: `llm_judge_nearest_neighbors/ablations/` → accuracy field
  - LN-Lens L0: `llm_judge_contextual_nn/ablations/` → computed from results list
  - Task Accuracy: `captions/*/generated_captions.json` → position matching
- **Seed mapping confirmed**: seed10 matches paper (52.7% L0 interp), not seed11 (59.7%)
- **TopBottom accuracy** slight discrepancy: computed 75.67%/86.67% vs paper 76.33%/87.33% (2/300 images)
- **New comprehensive table** in paper with 6 columns:
  - NN Overlap, Static NN L0, LN-Lens L0 (new!), Caption Score, Task Acc (new!)
  - Added ViT layer 6/10 ablations
  - Width calculation: ~383pt < 468pt (ICML full page width)
- **Git paper**: `23f9cbe` - Update ablation table: add LN-Lens L0, ViT layers
- **Git molmo**: `b59d46d` - Update paper submodule

### 2026-01-05 (TopBottom ablation LLM judge evaluations)
- **ROOT CAUSE**: TopBottom models never ran due to naming collision in `run_all_missing.sh`
  - `sed 's/train_mlp-only_pixmo_topbottom_//'` stripped prefix completely
  - Both TopBottom and baseline became `olmo-7b_vit-l-14-336` → TopBottom found existing baseline results and skipped
- **FIX**: Changed to `sed 's/train_mlp-only_pixmo_topbottom_/topbottom_/'` to preserve prefix
- **Ran LLM judge** for both TopBottom models (GPT-5, 100 images/layer, 9 layers each):
  - TopBottom (frozen LLM): 37-44% interpretability (layer 16 highest at 44%)
  - TopBottom (unfrozen LLM): 37-45% interpretability (layers 1,4 highest at 45%)
- **Updated data.json** with TopBottom results (now 10 contextual ablation models)
- **Regenerated ablation plots** including task ablation group
- **Git**: `bf27da3` - Fix naming collision: TopBottom and baseline shared same output name

### 2026-01-04 (FIX: Corrupted ablation baseline data)
- **CRITICAL BUG FOUND**: `olmo-7b_vit-l-14-336` in ablations folder was reading from topbottom data!
  - Ablation "baseline" showed ~38% interpretability instead of correct ~70%
  - Evidence: NNs were generic words ("cap.", "distance") instead of descriptive ("building", "roads")
  - Root cause: Data generation bug - wrong input_json path in results
- **Fix applied**:
  - `update_data.py`: Skip corrupted `olmo-7b_vit-l-14-336` from ablations extraction
  - `create_ablations_plots.py`: Use main model data (`olmo-7b+vit-l-14-336`) as baseline
- **Git**: `57bd1e0` - Fix ablation baseline data corruption bug

### 2026-01-04 (Qwen2-VL and Ablation Plotting)
- **Generated plots** for Qwen2-VL (off-the-shelf model):
  - Unified plot: `paper_figures_output/qwen2vl/qwen2vl_unified.pdf`
  - Individual plots for NN, LogitLens, Contextual V-Lens
  - All 9 layers (0, 1, 2, 4, 8, 16, 24, 26, 27) extracted successfully
- **Generated ablation plots**:
  - Mega plots comparing all ablations vs baseline (NN, Contextual)
  - Grouped plots by ablation type (caption_style, vit_layers, connector, seeds, llm_frozen, task)
  - LogitLens skipped (no ablation data available)
- **Bug fixes**:
  - `update_data.py`: Fixed Qwen2-VL LogitLens extraction - handled `"layer": "0"` format (not just `"layer0"`)
  - `create_qwen2vl_plots.py`: Convert JSON string keys to int for proper sorting
  - `create_ablations_plots.py`: Same string-to-int key conversion fix
- **Git**: Committed `4ac5022` - Add Qwen2-VL and ablation plotting scripts, fix data extraction bugs

### 2026-01-04 (FIX: Force square image display in viewer - cells were rectangular)
- **USER REPORT**: "this should all be square! both the full image and the cells"
  - I fixed the grid calculation but **MISSED** the visual display requirement
  - Images displayed with original aspect ratio (e.g., 640×480 → shown as 512×384)
  - This made grid cells rectangular instead of square
- **WHY I MISSED IT**: Focused on grid data (15×15 vs 16×16 tokens), not visual presentation
  - User's instruction WAS clear: "both the full image and the cells"
  - I interpreted "square" as grid data, not display
- **THE FIX - `create_unified_viewer.py` CSS**:
  - Added `height: 512px` to `.base-image` (was only width-constrained)
  - Added `object-fit: cover` to crop/fill the square container
  - Added `object-position: center` to center the image in the crop
  - Now ALL images display as 512×512 squares regardless of original aspect ratio
  - Grid cells are now square overlays on square images
- **APPLIES TO**: Both main viewer and ablation viewer (share same template)
- **VERIFIED**: Regenerated Qwen2-VL viewer - images now display as 512×512 squares
- **Git**: Committed and pushed (commit 6e3ada8)

### 2026-01-04 (FIX: Ablation viewer grid bug - wrong patches_per_chunk calculation)
- **USER REPORT**: "Image 006 and 0003 have last column AND row missing in viewer"
  - Viewer shows 14x14 cells but data has 15x15 = 225 patches
  - Viewer HTML says "16×16 patches (576 total)" which is mathematically wrong (16×16 = 256!)
- **ROOT CAUSE**: Line 527 in `generate_ablation_viewers.py`:
  ```python
  patches_per_chunk = max(patches_per_chunk, len(all_patches))
  ```
  - Default patches_per_chunk = 576 (24×24, wrong!)
  - Actual patches for Image 0003 = 225 (15×15)
  - Code took max(576, 225) = 576 instead of using actual 225
  - Then grid_size calculated as sqrt(576) = 24, but then overridden to 16 somehow
- **THE FIX**: Use actual patch count for THIS image, not max with default:
  ```python
  patches_per_chunk = len(all_patches)  # Use actual, not max!
  ```
  - Also improved logic to break after finding patches (don't accumulate across analysis types)
  - Changed default from 576 to 256 (16×16 for main models)
- **VERIFIED**: Regenerated Qwen2-VL viewer - now displays correctly!
  - Image 0000: 16×16 patches (256 total) ✓
  - Image 0003: 15×15 patches (225 total) ✓
  - Both variable grids now display all cells correctly with proper image overlay
- **Git**: Committed and pushed (commit 9ac69c7)

### 2026-01-04 (CRITICAL FIX: Qwen2-VL variable grid bug - disable processor do_resize)
- **USER REPORT**: "Qwen2-VL shows non-square images... visually shown as normal without resizing"
  - Images are 15x15 grid instead of consistent 16x16 = variable width cells
  - This was supposedly fixed days ago but data STILL has variable grids!
- **INVESTIGATION**: Checked data generated Jan 3 with `--force-square`:
  - Some images: 256 tokens (16x16) ✓
  - Some images: 225 tokens (15x15) ✗
  - **Conclusion**: The previous `--force-square` fix didn't actually work!
- **ROOT CAUSE FOUND**: Qwen2-VL processor's `do_resize=True` overrides our manual preprocessing
  - We resize to 448x448 square → pass to processor → processor STILL resizes it!
  - Processor adaptively resizes based on internal logic → some images get 420x420 (15x15 grid)
  - Setting `min_pixels=max_pixels` is NOT enough - processor still has `do_resize=True`
- **THE FIX**: Disable processor's internal resizing with `processor.image_processor.do_resize = False`
  - Applied to ALL 3 Qwen2-VL scripts:
    - `scripts/analysis/qwen2_vl/nearest_neighbors.py`
    - `scripts/analysis/qwen2_vl/logitlens.py`
    - `scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py`
  - Now processor uses our manually preprocessed 448x448 images as-is
  - Should guarantee consistent 16x16 grid for ALL images
- **NEXT**: Need to regenerate ALL Qwen2-VL data (NN, LogitLens, Contextual) + regenerate viewer
- **Git**: Committed and pushed (commit 1c060fc)

### 2026-01-04 (FIX: Ablation checkpoint path resolution + Qwen2-VL handling)
- **FOUND ROOT CAUSE**: Ablation checkpoints have different directory structure than main models
  - viewer_models.json: `checkpoint: "train_..._seed10_step12000-unsharded"` (includes suffix)
  - Actual directory: `molmo_data/checkpoints/ablations/train_..._seed10/` (no suffix!)
  - Result: preprocessor looked in wrong path, failed for ALL ablations
- **FOUND ISSUE**: Qwen2-VL is off-the-shelf HuggingFace model with no local checkpoint
  - Previous code crashed when trying to create preprocessor
  - Should gracefully fall back to basic resize
- **THE FIX - `scripts/analysis/viewer_lib.py`**:
  - Detect `_step12000-unsharded` suffix in checkpoint name
  - Strip suffix and check if `molmo_data/checkpoints/ablations/{base_name}/` exists
  - Use ablation path if found, otherwise use full checkpoint name
  - This resolves path correctly for all ablations while maintaining backward compatibility
- **THE FIX - `scripts/analysis/generate_ablation_viewers.py`**:
  - Catch both RuntimeError and FileNotFoundError when creating preprocessor
  - Log warning and fall back to basic resize (for Qwen2-VL)
  - All other ablations now successfully create preprocessors
- **VERIFIED**: All 10 ablations regenerated successfully
  - 9/10 created preprocessors with `resize='default'` (correct black padding for ViT)
  - 1/10 (Qwen2-VL) uses basic resize (expected, no local checkpoint)
  - Logs confirm: "MultiModalPreprocessor initialized: normalize='openai', resize='default'"
- **Git**: Committed and pushed (commit a3570ff)

### 2026-01-04 (FIX: Preprocessor path bug - fail loudly on errors)
- **FOUND BUG**: Preprocessor path was doubling `/step12000-unsharded` for ablations
  - Ablation checkpoints already end with `_step12000-unsharded`
  - But `create_preprocessor()` always appended `/step12000-unsharded`
  - Result: `molmo_data/checkpoints/.../step12000-unsharded/step12000-unsharded/config.yaml` ❌
- **FOUND BUG**: Silent failures - preprocessor errors only logged as WARNING
  - Should FAIL LOUDLY since preprocessing is critical for correct display
  - ViT models MUST use black padding, not resize
- **THE FIX - `scripts/analysis/viewer_lib.py`**:
  - Check if checkpoint_name already ends with `step12000-unsharded`
  - Only append if missing
  - Check if config.yaml exists BEFORE trying to load
  - Raise RuntimeError with clear error message instead of silent warning
- **Status**: Fix implemented and tested
- **Git**: Committed and pushed (commit 8f8b4da)

### 2026-01-04 (REFACTOR: Extract viewer_lib for modularity)
- **REFACTORED VIEWER GENERATION** to fix lack of modularity
  - **Problem**: Two separate scripts (`create_unified_viewer.py`, `generate_ablation_viewers.py`) had duplicated logic
  - **Impact**: Any fix to one had to be manually replicated to the other → error-prone, caused bugs
  - **Example**: LN-Lens badge fix applied to main viewer but not ablations → bug slipped through
- **THE REFACTOR - Created `scripts/analysis/viewer_lib.py`**:
  - Extracted shared functions: `pil_image_to_base64`, `create_preprocessor`, `escape_for_html`, `patch_idx_to_row_col`
  - Both scripts now import from single source of truth
  - Removed ~90 lines of duplicate code from `create_unified_viewer.py`
- **Updated `create_unified_viewer.py`**: Import from viewer_lib, removed duplicate definitions
- **Updated `generate_ablation_viewers.py`**: Import from viewer_lib instead of create_unified_viewer
- **Tested**: Both scripts import successfully
- **Git**: Will commit refactoring, then regenerate ablations

### 2026-01-04 (CRITICAL FIX: Image Preprocessing Bug in Ablation Viewers)
- **FOUND CRITICAL BUG**: All ablation viewers showing resized images instead of model-specific preprocessing
  - **Root cause**: `generate_ablation_viewers.py` hardcoded `pil_image.resize()` instead of using preprocessor
  - **Impact**: All ViT-based ablations showed WRONG preprocessing (resize instead of black padding)
  - **Why it happened**: Lack of modularity - two separate scripts with duplicated logic
- **THE FIX - `scripts/analysis/generate_ablation_viewers.py`**:
  - Added import: `create_preprocessor` from `create_unified_viewer`
  - Added `preprocessor` parameter to `create_image_viewer()`
  - Replaced hardcoded resize with: `pil_image_to_base64(pil_image, preprocessor)`
  - Now creates preprocessor per model and applies correct preprocessing (ViT=padding, SigLIP/DINOv2=resize)
- **Git**: Committed preprocessing fix (commit 17d9a6f)

### 2026-01-04 (CRITICAL FIX: LN-Lens Contextual Layer Badge Bug + Strict Validation)
- **FIXED CRITICAL BUG**: LN-Lens showing sparse/missing data and no layer badges
  - **Root cause**: Line 910 in `create_unified_viewer.py` was filtering contextual neighbors by layer
  - **The bug**: `nearest_contextual = [n for n in all_neighbors if n.get("contextual_layer") == layer]`
  - **Why it broke**: Layer dropdown selects VISUAL layer (0, 1, 2...), not contextual layer
  - **Expected behavior**: Show ALL top-5 contextual neighbors with badges (L1, L2, L8) showing which contextual layer each came from
  - **Example**: Visual layer 0, patch 5 → shows "sky (L8)", "blue (L2)", "cloud (L8)", etc.
- **THE FIX - `scripts/analysis/create_unified_viewer.py` lines 897-939**:
  - Removed the filter on line 910 that destroyed the badge functionality
  - Now shows all top-5 neighbors with `contextual_layer` field preserved for badge display
  - Each neighbor's badge shows which contextual layer it came from
- **KEY ARCHITECTURE INSIGHT**: LN-Lens has TWO separate layer concepts:
  1. **Visual layer** (dropdown): Which visual representation to analyze (layer 0, 1, 2...)
  2. **Contextual layer** (badges): Which LLM layer the contextual neighbor came from (L1, L2, L8...)
- **FIXED QWEN2-VL MISSING CONTEXTUAL DATA**:
  - **Root cause**: Wrong path in `viewer_models.json` - said `ablations/...` but should be `qwen2_vl/...`
  - Fixed contextual path: `qwen2_vl/Qwen_Qwen2-VL-7B-Instruct`
  - Regenerated Qwen2-VL viewer - now has all 3 analysis types (NN, LogitLens, LN-Lens)
  - File size went from 3.3MB → 7.0MB (contextual data added)
- **ADDED STRICT VALIDATION** to prevent partial data issues:
  - **Changed `add_models_to_viewer.py`**: Now requires ALL analysis types have data (no partial allowed!)
  - Old validation: passed if ANY type had data → allowed broken configs through
  - New validation: FAILS LOUDLY if any type is missing → forces fix before generation
  - Exit code 1 if validation fails, prints exactly what's missing
  - Added `has_all_data` field and `missing_types` list
- **Regenerated**: `analysis_results/unified_viewer_lite/` with corrected code
- **Generated**: All 10 ablation viewers with complete data
- **Updated SCHEMA_STANDARDIZATION.md**: Added viewer architecture section explaining visual vs contextual layers
- **Updated CLAUDE.md**: Added "VALIDATION BEFORE EXECUTION" section + Qwen2-VL special case warning
- **Git**: Will commit all fixes to `origin/final`

### 2026-01-03 (Viewer Investigation: LN-Lens Data Sparse But Working) ⚠️ INCORRECT ANALYSIS
- **NOTE**: This entry documents a MISUNDERSTANDING - the sparse data was actually a BUG, not expected behavior
- **What happened**: LN-Lens appeared to show sparse data per layer in viewer
- **Incorrect conclusion**: Thought this was expected because top-5 neighbors concentrate in 1-2 layers
- **Actual problem**: Line 910 was filtering contextual neighbors, breaking the badge display (fixed 2026-01-04)
- **Deleted all old viewer directories** (5 variants: _lite, _lite_final, _lite_new, _lite_old, _lite_qwen2vl_test)
- **Regenerated**: unified_viewer_lite (but with the bug still present)

### 2026-01-03 (Unified Viewer Layer Filtering Fix)
- **FIXED VIEWER BUG**: "No data for layer X" errors in unified_viewer_lite
  - **Root cause**: No layer filtering - viewer loaded ALL 15 layer files instead of standard 9-layer subset
  - **Before**: NN showed 15 layers (0,1,2,3,4,8,12,16,20,24,28,29,30,31,32)
  - **After**: NN shows 9 standard layers (0,1,2,4,8,16,24,30,31) ✓
- **FIXED - `scripts/analysis/create_unified_viewer.py`**:
  - Added `STANDARD_LAYERS_OLMO_LLAMA = [0,1,2,4,8,16,24,30,31]` and `STANDARD_LAYERS_QWEN = [0,1,2,4,8,16,24,26,27]`
  - Modified `scan_analysis_results()` to filter NN/LogitLens layers to standard subset
  - Modified `load_all_analysis_data()` to filter contextual layers to standard subset
  - Both functions now accept `llm` parameter for model-specific filtering
- **Regenerated**: `analysis_results/unified_viewer_lite/` with 10 images - all 9 models now work correctly
- **Updated CLAUDE.md**: Added prominent "DO NOT CREATE RANDOM FILES" section (lines 8-25)
  - Emphasizes JOURNAL.md is the ONLY place for documentation
  - Lists examples of bad patterns (WORK_SUMMARY_*.md, FIX_SUMMARY.md, NOTES.md)
  - Workflow: Code changes → Git commit → Update JOURNAL.md
- **Git**: Committed viewer fix + CLAUDE.md update, pushed to `origin/final`

### 2026-01-03 (Unified LLM Judge + Branch Rename)
- **UNIFIED LLM JUDGE**: Created `llm_judge/run_llm_judge.py`
  - Consolidates 3 scripts into 1: `run_single_model_with_viz.py`, `run_single_model_with_viz_logitlens.py`, `run_single_model_with_viz_contextual.py`
  - Uses `--analysis-type {nn, logitlens, contextual}` argument
  - Standardized `--api-key-file` for all analysis types
  - Handles full word extraction for contextual (vs raw tokens for NN/LogitLens)
  - Uses `ijson` streaming for contextual large JSON files
- **Updated `run_all_missing.sh`** to use the unified script
- **BRANCH RENAME**: `cleanup-legacy` → `final`
- **Git**: Pushed to `origin/final`, deleted `origin/cleanup-legacy`

### 2026-01-03 (Qwen2-VL Grid Bug + Missing Symlink + Repo Audit)
- **REPOSITORY CLEANUP AUDIT**: Created `CLEANUP_CANDIDATES.md`
  - Identified ~40+ test/debug files safe to remove
  - Found ~5 redundant script versions (older versions of NN/contextual scripts)
  - Catalogued 25 run_*.sh scripts (many superseded by run_all_missing.sh)
  - Listed one-time HF upload scripts, fix scripts, exploratory files
  - Recommended phased cleanup approach
- **FOUND MISSING SYMLINK**: `molmo_data` symlink was missing from repo!
  - `analysis_results` symlink existed → `/mnt/research/scratch/bkroje/analysis_results`
  - `molmo_data` symlink was missing → should point to `/mnt/research/scratch/bkroje/molmo_data`
  - This caused `run_all_missing.sh` to fail validation for ablation checkpoints
  - **FIX**: Created symlink: `ln -s /mnt/research/scratch/bkroje/molmo_data molmo_data`
  - **FIX**: Updated README.md with explicit symlink creation instructions
- **Qwen2-VL regeneration running** via `./run_all_missing.sh --qwen2vl-only --force-qwen2vl`

### 2026-01-03 (Qwen2-VL Grid Bug - ROOT CAUSE FOUND + FIXED)
- **ROOT CAUSE IDENTIFIED**: Qwen2-VL missing grid cells (e.g., 252 instead of 256 patches)
  - **Symptom**: Last row of Qwen2-VL viewer had missing cells (row 15, cols 12-15 empty)
  - **Root cause 1**: Analysis scripts didn't force square images!
    - Qwen2-VL's `min_pixels/max_pixels` constrains total pixels but **preserves aspect ratio**
    - A 640×480 image → 26×36 patches → 13×18 = 234 tokens (NOT 16×16=256!)
  - **Root cause 2**: LLM Judge hardcoded 24×24 grid (512/24 pixel crop)
    - Wrong for Qwen2-VL's 16×16 grid (should be 512/16)
    - Cropped regions were ~50% smaller than intended!
  - **Root cause 3**: LLM Judge expected `chunks` format but Qwen2-VL uses `patches` directly
- **FIXED - Analysis scripts**: Added `--force-square` flag (default=True):
  - `scripts/analysis/qwen2_vl/nearest_neighbors.py`
  - `scripts/analysis/qwen2_vl/logitlens.py`
  - `scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py`
- **FIXED - LLM Judge scripts**: Dynamic grid size + handle both formats:
  - `llm_judge/run_single_model_with_viz.py`
  - `llm_judge/run_single_model_with_viz_contextual.py`
  - `llm_judge/run_single_model_with_viz_logitlens.py`
- **Updated `run_all_missing.sh`**: Consolidated Qwen2-VL regeneration
  - Added `--qwen2vl-only` flag to skip ablations
  - Added `--force-qwen2vl` flag to delete and regenerate all Qwen2-VL data
  - Added Phase 10.5: Contextual NN for Qwen2-VL
  - Added Phase 11: LLM Judge Contextual NN for Qwen2-VL (parallel)
  - All phases use `--force-square` for consistent 16×16 grids
- **RUNNING**: Full Qwen2-VL regeneration via `./run_all_missing.sh --qwen2vl-only --force-qwen2vl`
- **Created `generate_demo.sh`**: Single command for complete demo (main + ablations)
- **Git pushes**: 7e82eb0, 5529b18, b4d942b, f43f172

### 2026-01-01 (Schema Standardization - continued)
- **FIXED** Qwen2-VL NN output key: `top_neighbors` → `nearest_neighbors` (now consistent with Molmo)
- **FIXED** `create_unified_viewer.py`: 
  - Added `import time` at module level (was causing UnboundLocalError)
  - Fixed indentation of timing log (must be inside `if has_results` block)
- **RESTORED** `analysis_results` symlink (was accidentally removed by git reset)
- **AUDITED** all consumer scripts (50+) for key dependencies:
  - `paper_plots/*.py` - uses `nearest_neighbors` ✓
  - `llm_judge/*.py` - uses `nearest_neighbors`, `nearest_contextual_neighbors` ✓
  - `generate_ablation_viewers.py` - has fallback for both keys ✓
- **SIMPLIFIED** `SCHEMA_STANDARDIZATION.md` to clean reference docs (no pending items)
- **Demo generation** running in background for all 9 models
- **Git pushes**: 72dd843, 9cc412f

### 2026-01-01 (Schema Standardization)
- **CREATED** `SCHEMA_STANDARDIZATION.md` - comprehensive analysis of all inconsistencies:
  - Documented 5 categories of inconsistencies: folders, files, JSON structure, patch keys, neighbor fields
  - Defined target unified schema for all 3 lenses
  - Created migration plan with 4 phases
  - Inventoried all scripts (active vs legacy)
- **FIXED** `create_unified_viewer.py` for allLayers format:
  - LN-Lens now uses `contextual_nearest_neighbors/` folder (removed `_vg` dependency)
  - Added support for new allLayers file format (one file per visual layer, contains all contextual layers)
  - Filters neighbors by `contextual_layer` during processing
  - Adds `contextual_layer` field to output for layer badge display
- **KEY INSIGHT**: Root cause of viewer issues is lack of data contract:
  - Embedding Matrix (Molmo): `nearest_neighbors` key
  - Embedding Matrix (Qwen2-VL): `top_neighbors` key (different!)
  - LogitLens: `top_predictions` key
  - LN-Lens: `nearest_contextual_neighbors` key
- **Git push**: d026e35 "Add schema standardization docs + fix LN-Lens allLayers format loading"

---

## 2024-12

### 2024-12-31 (Phase 1 Standardization)
- **RENAMED DISPLAY NAMES** in viewer for paper consistency:
  - "Nearest Neighbors (NN)" → "Embedding Matrix"
  - "Contextual NN" → "LN-Lens"
  - LogitLens unchanged
  - Updated `create_unified_viewer.py` (column headers, stats labels, comments)
  - Fixed indentation error in main loop (line 1902)

### 2024-12-31 (continued)
- **FIX**: Ablation viewer now works correctly:
  - Images resized to 512x512 for consistent square grid display
  - Changed `contextual` → `contextual_vg` key to match JS template expectations
  - Fixed parameter order: `ctx_layers` now passed to correct position (`ctx_vg_layers`, not `ctx_cc_layers`)
  - Added layer badge (`L{N}`) display for contextual neighbors (was missing from template)
- **ROOT CAUSE IDENTIFIED**: No data contract between analysis scripts (output JSON) and viewer scripts (consume JSON)
  - Different scripts use different key names: `nearest_neighbors` vs `neighbors`, `top_tokens` vs `predictions`
  - Wrapper keys: `contextual` vs `contextual_vg` vs `contextual_cc`
  - Will create `viewer_schema.py` as single source of truth
- **Git push**: "Fix ablation viewer: square images, layer badges, correct data keys"

### 2024-12-31
- **CRITICAL BUG FIX**: Qwen2-VL preprocessing inconsistency discovered and fixed:
  - Contextual NN used `--fixed-resolution 448` → 16x16 grid (256 tokens)
  - NN/LogitLens had no fixed resolution → 28x28 variable grids (777+ tokens)
  - **Same image, different grids = broken viewer!**
  - Fixed `scripts/analysis/qwen2_vl/nearest_neighbors.py` - added `--fixed-resolution` param
  - Fixed `scripts/analysis/qwen2_vl/logitlens.py` - added `--fixed-resolution` param
  - Updated `run_all_missing.sh` to use `--fixed-resolution 448` for Qwen2-VL
  - Deleted broken Qwen2-VL NN/LogitLens results (will regenerate)
- **BUG FIX**: Syntax error in `llm_judge/run_single_model_with_viz_contextual.py` (unexpected indent at line 391)
- **Updated** `generate_ablation_viewers.py` to find max grid across all analysis types
- **Git push**: "Standardize Qwen2-VL preprocessing across all scripts"

### 2024-12-30
- **BUG FIX**: `generate_ablation_viewers.py` was silently failing to load data due to format mismatches:
  - **Format A (main models/ablations)**: 
    - NN: `splits/validation/images` → `chunks/patches/nearest_neighbors`
    - LogitLens: `results` → `chunks/patches/top_predictions`
    - Contextual: `results` → `chunks/patches/nearest_contextual_neighbors`
  - **Format B (Qwen2-VL)**:
    - NN: `results` → `patches/top_neighbors` (different key name!)
    - LogitLens: `results` → `patches/top_predictions`
    - Contextual: `results` → `patches/nearest_contextual_neighbors`
  - Script now auto-detects format and handles both correctly
  - Added comprehensive validation: `--validate-only` flag, strict mode, data checks at load time
  - Added output validation: file size check, content verification
  - Now fails LOUDLY instead of generating empty viewers silently
- **BUG FOUND**: `run_all_missing.sh` only ran layer 0 for NN/LogitLens (`LAYERS="0"`). Ablation data is incomplete:
  - LogitLens: only layer 0 exists for all ablations
  - NN: inconsistent legacy data (some have random layers like 12, 20, 28 from old experiments)
  - Expected: layers 0,1,2,4,8,16,24,30,31 (9 layers) for OLMo ablations
- Added rule to .cursorrules/CLAUDE.md: "ALWAYS flag inconsistencies and investigate"
- **Git push**: "Add ablation viewer system and Qwen2-VL analysis scripts" (191 files, 52k+ insertions)
- Created bulletproof viewer management system:
  - `scripts/analysis/viewer_models.json` - config file for all models
  - `scripts/analysis/add_models_to_viewer.py` - validates data, updates index
  - `scripts/analysis/generate_ablation_viewers.py` - generates ablation viewers
- Generated 10 ablation viewers with 10 images each:
  - Qwen2-VL (off-the-shelf): NN=9, LogitLens=9, Contextual=9 layers
  - Seed 10, Seed 11: NN=3, LogitLens=1, Contextual=9 layers
  - Linear Connector: NN=4, LogitLens=1, Contextual=9 layers
  - Unfreeze LLM: NN=15, LogitLens=1, Contextual=9 layers
  - First-Sentence: NN=14, LogitLens=1, Contextual=9 layers
  - Earlier ViT (6, 10): NN=1, LogitLens=1, Contextual=9 layers
  - TopBottom variants: NN=1, LogitLens=1, Contextual=9 layers
- Updated README.md, .cursorrules, CLAUDE.md with new scripts and git workflow
- Created JOURNAL.md for change tracking

### 2024-12-29
- Created `run_all_missing.sh` master script for ablation analysis pipeline
- Added Qwen2-VL support for static NN and LogitLens
- LLM Judge evaluation running for contextual NN ablations

### 2024-12-23
- Initial unified viewer with 9 main models working
- Qwen2-VL contextual NN integration

---

## Format Guide

Each entry should be:
```
[YYYY-MM-DD] Brief description of what was done
```

Categories:
- **Scripts**: New or modified analysis scripts
- **Results**: New analysis results generated
- **Viewer**: Changes to the interactive viewer
- **Git**: Push/pull operations (include commit message)
- **Cleanup**: Deleted files, reorganization


### 2026-01-10 (Consolidate CLAUDE.md)

**Reduced from 413 → 171 lines (59% smaller):**
- Added "At Start of Every Chat" section (read README, JOURNAL, paper, SCHEMA)
- Top 3 rules prominently displayed
- Consolidated 18 sections → 7 sections
- Removed redundancy while keeping all critical rules

**Git:** Pushed to `final` branch (304f2e1)
