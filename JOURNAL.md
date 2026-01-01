# V-Lens Development Journal

A concise log of major changes, results, and git operations.

---

## 2026-01

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

