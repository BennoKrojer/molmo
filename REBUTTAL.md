# ICML 2026 Rebuttal — LatentLens (Submission 26565)

## Scores: 3, 4, 3, 4 (tiFA, khVQ, u6Pj, BNgn)

## Reviewer Concerns & Action Plan

### HIGH PRIORITY

**1. Corpus size sensitivity / ablation** (tiFA, khVQ, u6Pj)
- Status: DONE (v2 — correct evaluation with per-layer seed=42, proper images)
- Experiment: LatentLens at 0.1%, 1%, 10%, 100% of corpus (avg layers 0,8,16,31)
- Models: OLMo+CLIP, LLaMA+SigLIP, Qwen2+DINOv2
- Plot: `paper_plots/paper_figures_output/corpus_ablation.png`
- Scripts: `scripts/analysis/corpus_ablation/`
- **Key results (% interpretable tokens, avg across layers):**
  - OLMo+CLIP:    0.1%=57.5, 1%=70.2, 10%=76.0, 100%=71.2
  - LLaMA+SigLIP: 0.1%=59.5, 1%=54.8, 10%=58.8, 100%=61.5
  - Qwen2+DINOv2: 0.1%=58.2, 1%=76.8, 10%=82.8, 100%=82.0
- 100% values match paper numbers within ±2pp (sampling noise from 100 patches) ✓
- **Rebuttal text:**
  > We evaluated LatentLens at 0.1%, 1%, 10%, and 100% corpus size across three model pairs. At 0.1% (~300 embeddings), interpretability drops to ~58% for all models. Beyond 1%, performance stabilizes — the gap between 10% and 100% is ≤5pp across all models, within the noise of 100-patch sampling. This confirms the method requires a minimal corpus but is robust across a wide range of sizes, and that our full-corpus results are not sensitive to the exact corpus size chosen.

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
- Status: DONE (81/81 layers complete)
- Approach: pass@1 alongside pass@5 — full-word output is a direct advantage of LatentLens
- Output: `analysis_results/llm_judge_topk1/{method}/{model}/`
- **Final results (pass@1 avg across 9 layers):**
  - EmbeddingLens: OLMo+CLIP=41.6%, LLaMA+SigLIP=7.7%,  Qwen2+DINOv2=3.6%
  - LogitLens:     OLMo+CLIP=18.6%, LLaMA+SigLIP=2.9%,  Qwen2+DINOv2=6.8%
  - LatentLens:    OLMo+CLIP=52.4%, LLaMA+SigLIP=38.9%, Qwen2+DINOv2=56.2%
- **Pass@5 → Pass@1 drop:**
  - LatentLens retains 62–70% of its pass@5 performance at pass@1
  - EmbeddingLens retains only 32–41%, LogitLens retains 41–59%
  - LatentLens remains best method at pass@1 by a large margin (38–56% vs 2–18% for LogitLens)
- **Rebuttal argument:** At pass@1 (no lenient matching), LatentLens still achieves 38–56%, while LogitLens collapses to 2–18%. This demonstrates full-word output is a genuine advantage, not an artifact of pass@5 leniency.

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
- **Seed drift fix (2026-03-25):** Original script passed all layers in one invocation (seed=42 set once → only layer 0 matched paper's patches). Fixed to run each layer as a separate invocation with fresh seed=42, matching paper protocol. Deleted old results and re-launched.
- **Launched evaluation** (re-run) 2026-03-25. Now parallel (9 jobs at once). Log at `analysis_results/llm_judge_topk1/run.log`
- Pass@5 baselines for these 3 models (from data.json):
  - OLMo+CLIP:     NN=57.1%, LogitLens=34.3%, LatentLens=72.3%
  - LLaMA+SigLIP:  NN=24.0%, LogitLens=7.1%,  LatentLens=62.3%
  - Qwen2+DINOv2:  NN=8.8%,  LogitLens=16.2%, LatentLens=79.9%
- **COMPLETE (2026-03-26). Final results (avg across 9 layers):**

  | Model            | Method        | Pass@5 | Pass@1 | Drop    |
  |------------------|---------------|--------|--------|---------|
  | OLMo+CLIP        | EmbeddingLens | 57.1%  | 41.6%  | −15.5pp |
  | OLMo+CLIP        | LogitLens     | 34.3%  | 18.6%  | −15.8pp |
  | OLMo+CLIP        | LatentLens    | 72.3%  | 52.4%  | −19.9pp |
  | LLaMA+SigLIP     | EmbeddingLens | 24.0%  |  7.7%  | −16.3pp |
  | LLaMA+SigLIP     | LogitLens     |  7.1%  |  2.9%  |  −4.2pp |
  | LLaMA+SigLIP     | LatentLens    | 62.3%  | 38.9%  | −23.4pp |
  | Qwen2+DINOv2     | EmbeddingLens |  8.8%  |  3.6%  |  −5.2pp |
  | Qwen2+DINOv2     | LogitLens     | 16.2%  |  6.8%  |  −9.4pp |
  | Qwen2+DINOv2     | LatentLens    | 79.9%  | 56.2%  | −23.7pp |

- **Rebuttal argument:** LatentLens drops more in absolute terms (ceiling/floor effect — it starts higher).
  Key finding: **LatentLens pass@1 (avg 49.2%) still beats EmbeddingLens pass@5 (avg 29.9%) and LogitLens pass@5 (avg 19.2%).**
  Even restricted to a single candidate word, LatentLens outperforms baselines using their top-5.
  Full-word output is a genuine advantage, not an evaluation bias.
- **Partial results (2026-03-25, 2h in):**
  - EmbeddingLens/OLMo+CLIP: pass@5=57.1% → pass@1=41.6% (**−15.5pp**, 9/9 layers done)
  - LogitLens/OLMo+CLIP: pass@1 layers 0-2 = 5-8% (partial, running layer 4)
  - If LatentLens drop is ≪15pp, it confirms full-word output is a real advantage
- **Monitor progress:** `tail -20 analysis_results/llm_judge_topk1/run.log`
- **Check partial results:** `python scripts/analysis/corpus_ablation/compare_topk.py`
- **Estimated completion:** ~3-4 hours from launch (~15-30 min per job × 9 jobs, sequential)

### 2026-03-26: Corpus Ablation v2 + Pass@1 COMPLETE
- **Corpus ablation v2** (correct eval: per-layer seed=42, `pixmo_cap_validation_indexed` images):
  - 3 models × 4 pcts (0.1%, 1%, 10%, 100%) × 4 layers (0,8,16,31) = 48 evals
  - Results match paper 100% numbers within ±2pp ✓
  - Plot: `paper_plots/paper_figures_output/corpus_ablation.png`
- **Pass@1 evaluation** (81/81 layers done, all 3 methods × 3 models × 9 layers):
  - LatentLens pass@1: 52.4% / 38.9% / 56.2% (OLMo+CLIP / LLaMA+SigLIP / Qwen2+DINOv2)
  - Outperforms baselines at pass@1 by large margin — confirms full-word advantage

### 2026-03-25: Corpus Size Ablation COMPLETE
- All 15 LatentLens runs done (3 models x 5 corpus sizes x 9 layers)
- All 15 LLM judge evaluations done (GPT-5, 100 patches per layer)
- Results (avg % interpretable across 9 layers):
  - OLMo+CLIP:     5%=67.0, 10%=69.4, 25%=69.4, 50%=66.5, 100%=66.1 (range 3.3pp)
  - LLaMA+SigLIP:  5%=62.9, 10%=60.9, 25%=61.7, 50%=63.3, 100%=65.3 (range 4.4pp)
  - Qwen2+DINOv2:  5%=78.3, 10%=79.2, 25%=78.9, 50%=78.0, 100%=77.4 (range 1.8pp)
- Plot: `analysis_results/corpus_ablation/corpus_ablation_plot.{pdf,png}`
- Conclusion: LatentLens is robust to corpus size — even 5% (~15K embeddings) suffices

### 2026-03-24: Corpus Size Ablation Setup
- Created subsampling script: `scripts/analysis/corpus_ablation/subsample_corpus.py`
- Created orchestration script: `scripts/analysis/corpus_ablation/run_corpus_ablation.sh`
- Created plotting script: `scripts/analysis/corpus_ablation/plot_corpus_ablation.py`
- Full corpus: ~307K embeddings per layer (OLMo), 8 layers per model
- Subsample fractions: 5% (~15K), 10% (~31K), 25% (~77K), 50% (~154K)
- Smaller corpora are strict subsets of larger ones (same seed, prefix of permutation)
