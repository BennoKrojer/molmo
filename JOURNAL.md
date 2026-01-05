# V-Lens Development Journal

A concise log of major changes, results, and git operations.

---

## 2026-01

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

