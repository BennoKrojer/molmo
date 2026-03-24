# ICML 2026 Rebuttal — LatentLens (Submission 26565)

## Scores: 3, 4, 3, 4 (tiFA, khVQ, u6Pj, BNgn)

## Reviewer Concerns & Action Plan

### HIGH PRIORITY

**1. Corpus size sensitivity / ablation** (tiFA, khVQ, u6Pj)
- Status: IN PROGRESS
- Experiment: Run LatentLens with 5%, 10%, 25%, 50%, 100% of contextual corpus
- Models: OLMo+CLIP, LLaMA+SigLIP, Qwen2+DINOv2 (3 diverse combos)
- Plot: X=corpus size, Y=avg interpretable tokens across layers
- Scripts: `scripts/analysis/corpus_ablation/`
- Expected output: `analysis_results/corpus_ablation/corpus_ablation_plot.pdf`

**2. Faithfulness / causal guarantees** (u6Pj, BNgn)
- Status: TODO (writing-only)
- Arguments to make:
  - Human correlation study (Cohen's Kappa 0.68) validates judge
  - Cross-layer alignment (Fig 4) diagonal structure = evidence of faithfulness
  - LatentLens operates in the model's own representation space (more faithful by construction)
  - Plausibility vs faithfulness distinction: LogitLens has same issue

**3. Stronger baselines: Tuned Lens, SAEs** (u6Pj)
- Status: TODO
- Options:
  - (a) Argue Tuned Lens requires per-layer training (not training-free), different comparison class
  - (b) Actually implement and run Tuned Lens (2-3 days)
  - SAEs: argue different goal (decomposition vs interpretation), cite concurrent work

### MEDIUM PRIORITY

**4. VLM judge bias: full words vs fragments** (BNgn, tiFA)
- Status: IN PROGRESS
- Approach: Report pass@1 alongside pass@5 (no post-processing — full-word output is a direct advantage of LatentLens)
- Experiment: Re-run LLM judge with `--top-k 1` for 3 models × 3 methods (same models as #1)
- Models: OLMo+CLIP, LLaMA+SigLIP, Qwen2+DINOv2
- Methods: EmbeddingLens, LogitLens, LatentLens (9 jobs total, each 9 layers × 100 patches)
- Script: `scripts/analysis/corpus_ablation/run_topk1_evaluation.sh`
- Output: `analysis_results/llm_judge_topk1/{method}/{model}/evaluation_results.json`
- Uses release repo evaluate script with new `--top-k` arg (`latentlens_release/reproduce/scripts/evaluate/evaluate_interpretability.py`)
- Estimated API cost: ~$10-15

**5. Computational cost numbers** (tiFA, khVQ)
- Status: TODO (just report numbers)
- Corpus construction time, storage (GB), NN search time per image

**6. More off-the-shelf VLMs** (BNgn)
- Status: IN PROGRESS
- Models: **Molmo-7B-D** (`allenai/Molmo-7B-D-0924`) + **LLaVA-1.5-7B** (`llava-hf/llava-1.5-7b-hf`)
- Same pipeline as Qwen2-VL: all 3 methods (EmbeddingLens, LogitLens, LatentLens) + LLM judge + layer alignment heatmap
- Scripts: `scripts/analysis/molmo_7b/` and `scripts/analysis/llava_1_5/`
- Architecture details:
  - Molmo-7B-D: Qwen2 LLM backbone (28 layers, 3584 hidden, 152064 vocab), custom vision encoder
  - LLaVA-1.5-7B: LLaMA/Vicuna backbone (32 layers, 4096 hidden, 32064 vocab), CLIP ViT-L/14-336 (24×24=576 patches)
- Contextual embeddings: extract from each VLM's finetuned LLM (not vanilla), same as Qwen2-VL approach
- Deliverables: core LLM judge scores + layer alignment heatmaps

#### Implementation Plan (item #6)

**Phase 1: Infrastructure & Preprocessing** (~1 hour)
- [ ] 1a. Create `scripts/analysis/molmo_7b/preprocessing.py` — single source of truth for Molmo image preprocessing
  - Determine image size, grid size, vision token detection (how Molmo marks image tokens)
  - Must handle Molmo's custom processor (MolmoProcessor with trust_remote_code)
- [ ] 1b. Create `scripts/analysis/llava_1_5/preprocessing.py` — same for LLaVA-1.5
  - CLIP ViT-L/14-336: 336×336 input → 24×24=576 patches (no CLS token passed to LLM)
  - Force-square center-crop for consistent grids
- [ ] 1c. Unit tests for preprocessing: verify token counts, grid sizes, image dimensions

**Phase 2: Contextual Embedding Extraction** (~2-4 hours compute)
- [ ] 2a. Create `scripts/analysis/molmo_7b/create_contextual_embeddings.py`
  - Extract from Molmo's Qwen2 backbone (finetuned weights, not vanilla Qwen2)
  - Layers: 1,2,4,8,16,24,26,27 (28-layer model, same as Qwen2)
  - Output: `molmo_data/contextual_llm_embeddings_vg/allenai_Molmo-7B-D-0924/`
- [ ] 2b. Create `scripts/analysis/llava_1_5/create_contextual_embeddings.py`
  - Extract from LLaVA's Vicuna backbone (finetuned weights)
  - Layers: 1,2,4,8,16,24,30,31 (32-layer model, same as LLaMA)
  - Output: `molmo_data/contextual_llm_embeddings_vg/llava-hf_llava-1.5-7b-hf/`
- [ ] 2c. Launch extraction jobs (long-running, ~2-4 hours each)
- [ ] 2d. Build search caches after extraction completes

**Phase 3: Three Interpretability Methods** (~2 hours)
- [ ] 3a. Molmo-7B: `nearest_neighbors.py`, `logitlens.py`, `contextual_nearest_neighbors_allLayers_singleGPU.py`
  - Adapt from Qwen2-VL scripts; key challenge is vision token identification in Molmo's sequence
- [ ] 3b. LLaVA-1.5: same 3 scripts
  - LLaVA uses `<image>` token (id=32000) as placeholder; vision features replace it in forward pass
- [ ] 3c. Sanity checks: verify vision token count matches expected grid, spot-check NN results visually

**Phase 4: Demo Viewer** (~1 hour)
- [ ] 4a. Generate HTML demo for both models (adapt `create_unified_viewer.py` or standalone)
  - Verify token-to-image-position alignment visually
  - This is the best way to catch preprocessing bugs early

**Phase 5: LLM Judge Evaluation** (~3-4 hours compute, ~$15-20 API cost)
- [ ] 5a. Run LLM judge for both models × 3 methods × 9 layers × 100 patches
  - Molmo layers: 0,1,2,4,8,16,24,26,27
  - LLaVA layers: 0,1,2,4,8,16,24,30,31
- [ ] 5b. Collect results, compare with Qwen2-VL numbers
- [ ] 5c. Generate layer alignment heatmaps

**Phase 6: Integration** (~30 min)
- [ ] 6a. Add results to `paper_plots/data.json`
- [ ] 6b. Update paper text (off-the-shelf subsection)

**7. Layer alignment explanation for early layers** (BNgn)
- Status: TODO (writing-only)
- Key point: early visual layers match MID text layers (that's the point of Fig 4)

**8. Training dynamics over time** (khVQ)
- Status: TODO (nice-to-have, probably skip)

## Experiment Log

### 2026-03-24: Off-the-shelf VLMs — Molmo-7B + LLaVA-1.5 (item #6)
- **Goal:** Replicate Qwen2-VL off-the-shelf analysis for two more VLMs
- **Architecture confirmed:**
  - Molmo-7B-D (`allenai/Molmo-7B-D-0924`): Qwen2 backbone, 28 layers, 3584 hidden, 152064 vocab, custom vision encoder (OpenAI CLIP ViT-L/14)
  - LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`): LLaMA/Vicuna backbone, 32 layers, 4096 hidden, 32064 vocab, CLIP ViT-L/14-336 (24×24=576 patches)
- **Key design decision:** Use each VLM's finetuned LLM weights for contextual embedding extraction (not vanilla LLM), following Qwen2-VL precedent
- **Scripts location:** `scripts/analysis/molmo_7b/` and `scripts/analysis/llava_1_5/`, adapted from `scripts/analysis/qwen2_vl/`
- **Contextual embeddings output:**
  - `molmo_data/contextual_llm_embeddings_vg/allenai_Molmo-7B-D-0924/`
  - `molmo_data/contextual_llm_embeddings_vg/llava-hf_llava-1.5-7b-hf/`

**Implementation progress:**
- [x] Phase 1: Preprocessing scripts + unit tests (30 tests pass)
- [x] Phase 2: Contextual embedding extraction (launched, ~10% as of 18:00 UTC)
- [x] Phase 3: All 3 interpretability method scripts created
- [ ] Phase 4: Demo viewer
- [ ] Phase 5: LLM Judge evaluation
- [ ] Phase 6: Integration

**Key findings:**
- Molmo-7B-D uses multi-crop (base crop = 12×12 = 144 tokens + high-res crops). Analysis uses base crop only.
- LLaVA-1.5 HF processor already expands `<image>` to 576 tokens in input_ids (no sequence expansion during forward)
- Molmo's `model.model.forward()` requires float16 image tensors (dtype mismatch with float32 otherwise)
- Molmo: `wte.embedding` (Parameter, not Embedding.weight), `ff_out` (LM head), `ln_f` (RMSLayerNorm)
- LLaVA: `model.model.language_model.embed_tokens`, `model.lm_head`, `model.model.language_model.norm`
- Hidden states: Molmo = 29 (emb + 28 layers), LLaVA = 33 (emb + 32 layers)

**Files created:**
- `scripts/analysis/contextual_embeddings_common.py` — shared extraction infrastructure
- `scripts/analysis/molmo_7b/` — preprocessing.py, test_preprocessing.py, create_contextual_embeddings.py, nearest_neighbors.py, logitlens.py, contextual_nearest_neighbors_allLayers_singleGPU.py, run_extract_contextual.sh, run_all_analysis.sh
- `scripts/analysis/llava_1_5/` — same structure as molmo_7b/

**Monitor extraction:**
- `tail -5 analysis_results/molmo_7b_extraction.log`
- `tail -5 analysis_results/llava_1_5_extraction.log`

### 2026-03-24: Pass@1 (top-k=1) Evaluation Setup
- Added `--top-k` argument to `latentlens_release/reproduce/scripts/evaluate/evaluate_interpretability.py`
  - Threads through `extract_words_for_patch()` and `extract_words_from_contextual()`
  - Default remains 5 (backward compatible)
- Fixed `load_analysis_results()` to skip directories with `.json` suffix
- Created `scripts/analysis/corpus_ablation/create_indexed_images.py` to symlink PixMoCap validation images as `{idx:05d}.jpg`
  - Output: `analysis_results/pixmo_cap_validation_indexed/` (100 symlinks)
- Created `scripts/analysis/corpus_ablation/run_topk1_evaluation.sh` orchestration script
  - 3 models × 3 methods = 9 jobs, each evaluating 9 layers × 100 patches with top-1 only
  - Uses same seed=42 and 100 patches as main paper evaluation
- Dry-run verified: all 3 methods correctly return single word with `top_k=1`
- Comparison script: `scripts/analysis/corpus_ablation/compare_topk.py` (loads pass@5 from `data.json`, pass@1 from `llm_judge_topk1/`)
- Argument: full-word interpretability is a *feature* of LatentLens, not a bias. Pass@1 comparison shows this.
- **Launched evaluation** at ~14:30 UTC, 2026-03-24. 9 jobs running via nohup, log at `analysis_results/llm_judge_topk1/run.log`
- Pass@5 baselines for these 3 models (from data.json):
  - OLMo+CLIP:     NN=57.1%, LogitLens=34.3%, LatentLens=72.3%
  - LLaMA+SigLIP:  NN=24.0%, LogitLens=7.1%,  LatentLens=62.3%
  - Qwen2+DINOv2:  NN=8.8%,  LogitLens=16.2%, LatentLens=79.9%
- Expected: all methods drop with top-1, but LatentLens drops LESS because its top-1 is a full word
- **Monitor progress:** `tail -20 analysis_results/llm_judge_topk1/run.log`
- **Check partial results:** `python scripts/analysis/corpus_ablation/compare_topk.py`
- **Estimated completion:** ~3-4 hours from launch (~15-30 min per job × 9 jobs, sequential)

### 2026-03-24: Corpus Size Ablation Setup
- Created subsampling script: `scripts/analysis/corpus_ablation/subsample_corpus.py`
- Created orchestration script: `scripts/analysis/corpus_ablation/run_corpus_ablation.sh`
- Created plotting script: `scripts/analysis/corpus_ablation/plot_corpus_ablation.py`
- Full corpus: ~307K embeddings per layer (OLMo), 8 layers per model
- Subsample fractions: 5% (~15K), 10% (~31K), 25% (~77K), 50% (~154K)
- Smaller corpora are strict subsets of larger ones (same seed, prefix of permutation)
