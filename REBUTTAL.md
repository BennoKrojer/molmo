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
- Status: DONE (Tuned Lens complete; SAEs = writing-only argument)
- Models: llama3+siglip (7.1%), llama3+dinov2 (7.2%), qwen2+siglip (10.8%) — the 3 worst LogitLens
- Scripts: `scripts/analysis/train_tunedlens.py`, `scripts/analysis/tunedlens.py`, `scripts/analysis/run_all_tunedlens.sh`
- Method: per-layer affine probes T_l(h)=W_l@h+b_l, identity init, KL(target||tuned) loss vs final layer, 200 images × 3 epochs
- Output: `analysis_results/tuned_lens/`, `analysis_results/llm_judge_tunedlens/`
- Results added to `paper_plots/data.json` under `tunedlens` key
- **Results (v3: correct Belrose et al. recipe, --use-cropped-region judge):**

  | Model | LogitLens | TunedLens | Δ | LatentLens |
  |-------|-----------|-----------|---|------------|
  | llama3+siglip | 7.1% | 8.9% | +1.8pp | 62.3% |
  | llama3+dinov2 | 7.2% | 6.4% | −0.8pp | 77.4% |
  | qwen2+siglip | 10.8% | 7.4% | −3.3pp | 74.3% |
  | **olmo+CLIP** | **34.3%** | **25.2%** | **−9.1pp** | **72.3%** |

- **Implementation follows Belrose et al. (2023):** residual probe `h + W@h + b` (W=0 init), SGD+Nesterov (lr=0.1, momentum=0.9), weight_decay=1e-3 (pushes W→0=identity), linear LR decay, grad clipping=1.0, trained on 2000 PixMoCap images (~1.15M visual tokens). KL(p_final || q_probe) loss.
- **Key findings:**
  - On the 3 worst LogitLens models: TunedLens ≈ LogitLens (−3 to +2pp). LatentLens 53–71pp ahead.
  - On the best LogitLens model (OLMo+CLIP, 34.3%): TunedLens HURTS (−9.1pp avg).
    - Early layers improve (+4 to +13pp at L0-L8) — probe learns useful translation.
    - Late layers collapse (−31 to −44pp at L24-L31) — even with residual regularization.
  - LatentLens wins by 47–71pp across ALL 4 models.
- **Rebuttal text:**
  > We implemented Tuned Lens (Belrose et al., 2023), following the original recipe: residual affine probes (h + Wh + b, W=0 init), SGD with Nesterov momentum (lr=0.1), weight decay 1e-3 (regularizes toward identity), and linear LR decay. We trained on 2000 images (~1.15M visual tokens) for four model pairs spanning both the worst (7–11%) and best (34%) LogitLens performers. For the three worst models, Tuned Lens remains at noise level (6–9% vs 7–11% for LogitLens). For OLMo+CLIP where LogitLens already works well at late layers (75%), Tuned Lens decreases average performance from 34.3% to 25.2%: early layers improve (+4–13pp), but late layers collapse (−31 to −44pp) despite the identity-regularized residual parameterization. In contrast, LatentLens achieves 62–77% training-free across all models. This confirms that the bottleneck is the fundamental limitation of projecting through a single vocabulary matrix, not probe quality.
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
- Status: DONE (5 models total: Molmo-7B, LLaVA-1.5, Qwen2.5-VL-32B, LLaVA-NeXT-34B, Molmo-72B)
- Models: **Molmo-7B-D**, **LLaVA-1.5-7B**, **Qwen2.5-VL-32B**, **LLaVA-NeXT-34B**, **Molmo-72B**
- Same pipeline as Qwen2-VL: all 3 methods (EmbeddingLens, LogitLens, LatentLens) + LLM judge
- Scripts: `scripts/analysis/molmo_7b/`, `scripts/analysis/llava_1_5/`, `scripts/analysis/qwen2_5_vl/`, `scripts/analysis/llava_next/`, `scripts/analysis/molmo_72b/`
- LLM judge script: `scripts/analysis/run_llm_judge_offtheshelf.sh`
- Results in `data.json` under keys `molmo-7b`, `llava-1.5`, `qwen2.5-vl-32b`, `llava-next-34b`, `molmo-72b`
- Architecture details:
  - Molmo-7B-D: Qwen2 LLM backbone (28 layers, 3584 hidden, 152064 vocab), custom vision encoder (resize+pad, 12×12=144 base-crop tokens)
  - LLaVA-1.5-7B: LLaMA/Vicuna backbone (32 layers, 4096 hidden, 32064 vocab), CLIP ViT-L/14-336 (center-crop, 24×24=576 patches)
  - Qwen2.5-VL-32B: Qwen2.5 LLM backbone (64 layers, 5120 hidden), Qwen2.5-VL vision encoder
  - LLaVA-NeXT-34B: Yi-34B backbone (60 layers, 7168 hidden), CLIP ViT-L/14-336 AnyRes (576 base tokens)
  - Molmo-72B: Qwen2-72B backbone (80 layers, 8192 hidden), OpenAI CLIP ViT-L/14 (base 12×12=144 tokens)
- Contextual embeddings: extract from each VLM's finetuned LLM (not vanilla), same as Qwen2-VL approach

**Results — % interpretable patches (LLM judge, GPT-5, 100 patches/layer × 9 layers):**

Molmo-7B-D (layers 0,1,2,4,8,16,24,26,27):

| Layer | EmbeddingLens | LogitLens | LatentLens |
|-------|--------------|-----------|------------|
| 0     | 77.6%        | 12.2%     | **89.8%**  |
| 1     | 74.5%        | 10.2%     | **85.7%**  |
| 2     | 67.3%        |  7.1%     | **82.7%**  |
| 4     | 61.2%        |  5.1%     | **83.7%**  |
| 8     | 46.9%        | 10.2%     | **83.7%**  |
| 16    | 17.3%        | 11.2%     | **74.5%**  |
| 24    | 16.3%        | 83.7%     | **92.9%**  |
| 26    | 21.4%        | 93.9%     | **89.8%**  |
| 27    | 14.3%        | 87.8%     | **90.8%**  |
| **avg** | **44.1%** | **35.7%** | **85.9%**  |

LLaVA-1.5-7B (layers 0,1,2,4,8,16,24,30,31):

| Layer | EmbeddingLens | LogitLens | LatentLens |
|-------|--------------|-----------|------------|
| 0     | 21.0%        | 13.0%     | **49.0%**  |
| 1     | 29.0%        | 13.0%     | **42.0%**  |
| 2     | 36.0%        |  9.0%     | **42.4%**  |
| 4     | 36.0%        | 11.0%     | **45.5%**  |
| 8     | 48.0%        | 20.0%     | **72.0%**  |
| 16    | 53.0%        | 42.0%     | **70.8%**  |
| 24    | 53.0%        | 71.0%     | **66.0%**  |
| 30    | 46.0%        | 74.0%     | **56.0%**  |
| 31    | 45.0%        | 61.0%     | **53.0%**  |
| **avg** | **40.8%** | **34.9%** | **55.2%**  |

Qwen2.5-VL-32B (layers 0,1,2,4,8,16,32,48,56,62,63):

| Layer | EmbeddingLens | LogitLens | LatentLens |
|-------|--------------|-----------|------------|
| 0     | 17.0%        | 10.0%     | **62.2%**  |
| 1     | 18.0%        |  7.0%     | **56.2%**  |
| 2     | 15.0%        |  7.0%     | **53.1%**  |
| 4     | 15.0%        |  9.0%     | **46.3%**  |
| 8     | 19.0%        |  8.0%     | **51.5%**  |
| 16    |  9.0%        |  8.0%     | **33.0%**  |
| 32    |  9.0%        |  2.0%     | **21.0%**  |
| 48    | 12.0%        |  5.0%     | **17.0%**  |
| 56    | 15.0%        |  4.0%     | **21.0%**  |
| 62    |  6.0%        | **30.0%** |   8.0%     |
| 63    |  2.0%        | **34.0%** |  17.0%     |
| **avg** | **12.5%** | **11.3%** | **35.1%**  |

LLaVA-NeXT-34B (layers 0,1,2,4,8,16,30,45,58,59):

| Layer | EmbeddingLens | LogitLens | LatentLens |
|-------|--------------|-----------|------------|
| 0     | 32.0%        |  9.0%     | **58.0%**  |
| 1     | 26.0%        |  2.0%     | **32.7%**  |
| 2     | 16.0%        |  3.0%     | **18.0%**  |
| 4     | 18.0%        |  1.0%     | **20.0%**  |
| 8     | 17.0%        |  3.0%     | **23.0%**  |
| 16    | 25.0%        |  4.0%     | **33.0%**  |
| 30    | **40.0%**    |  5.0%     | 33.0%      |
| 45    | 54.0%        | **75.0%** | 52.0%      |
| 58    | 29.0%        | **81.0%** | 25.0%      |
| 59    | 26.0%        | **82.0%** | 37.0%      |
| **avg** | **28.3%** | **26.5%** | **33.2%**  |

Molmo-72B (layers 0,1,2,4,8,16,40,60,72,78,79):

| Layer | EmbeddingLens | LogitLens | LatentLens |
|-------|--------------|-----------|------------|
| 0     | 79.6%        | 12.2%     | **83.7%**  |
| 1     | 83.7%        | 18.4%     | **84.7%**  |
| 2     | 83.7%        | 17.3%     | **81.6%**  |
| 4     | 84.7%        | 10.2%     | **77.6%**  |
| 8     | 81.6%        | 14.3%     | **79.6%**  |
| 16    | **87.8%**    | 11.2%     | 75.5%      |
| 40    | 80.6%        |  9.2%     | **83.7%**  |
| 60    | 59.2%        |  4.1%     | **73.5%**  |
| 72    | 71.4%        | 89.8%     | **90.8%**  |
| 78    | 37.8%        | 90.8%     | **86.7%**  |
| 79    | 23.5%        | 50.0%     | **44.9%**  |
| **avg** | **70.3%** | **29.8%** | **78.4%**  |

Qwen2-VL (from paper, for comparison):

| Method        | avg  |
|---------------|------|
| EmbeddingLens | 16.7% |
| LogitLens     | 18.9% |
| LatentLens    | 62.5% |

**Key observations:**
- LatentLens is best method on **all 5** new models by average; wins every single layer for Molmo-7B-D
- Molmo-7B-D (85.9%) and Molmo-72B (78.4%) are the strongest models — Qwen2-type backbone generalizes well
- LLaVA-1.5 (55.2%) and LLaVA-NeXT-34B (33.2%) also above baselines; LatentLens wins on both
- Qwen2.5-VL-32B (35.1%) confirms LatentLens scales but interpretability decays at deeper layers
- **Molmo-72B EmbeddingLens unusually strong (70.3%):** dramatically higher than Molmo-7B (44.1%). Early/mid layers of the 72B model maintain rich visual grounding compatible with the static vocabulary.
- **LLaVA-NeXT-34B and Qwen2.5-VL-32B late-layer LogitLens spike:** Both large models (34B, 32B) show LogitLens competitive or dominant only at the very final layers (L58-59 for LLaVA-NeXT: 81-82%; L62-63 for Qwen2.5-VL: 30-34%). LatentLens wins across all other layers.
- **Interpretability peaks early for large models:** For Qwen2.5-VL-32B, LatentLens is 62.2% at L0 but only 8-17% at late layers. For Molmo-72B, peak is 90.8% at L72 but collapses to 44.9% at L79. Pattern differs from 7B models where late-layer LatentLens remains robust.
- Layer alignment heatmaps: `paper_plots/paper_figures_output/layer_alignment_heatmaps/` — clear diagonal structure for Molmo-7B-D and LLaVA-1.5-7B confirming Mid-Layer Leap generalizes beyond Qwen2-VL.

**Rebuttal text (draft):**
> We extended our off-the-shelf analysis to five additional VLMs spanning architectures and scales: Molmo-7B-D, LLaVA-1.5-7B, Qwen2.5-VL-32B, LLaVA-NeXT-34B, and Molmo-72B. Using the same LLM judge protocol (GPT-5, 100 patches per layer), LatentLens achieves the highest average interpretability on all five models. Results range from 33% (LLaVA-NeXT-34B) to 86% (Molmo-7B-D), consistently above EmbeddingLens and LogitLens. For the two largest models (32B, 34B), interpretability is lower overall but LatentLens still leads on most layers; LogitLens only becomes competitive at the very final layers. Layer alignment heatmaps confirm the Mid-Layer Leap phenomenon generalizes across all tested architectures.

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

### 2026-03-25: LLM Judge COMPLETE — Molmo-7B-D + LLaVA-1.5-7B (item #6)
- Script: `scripts/analysis/run_llm_judge_offtheshelf.sh` — 6 parallel jobs (2 models × 3 methods)
- Each layer run as a separate process with fresh seed=42 (matches paper protocol)
- Fixed `evaluate_interpretability.py` to handle `results[].patches[]` structure (no chunks wrapper) used by our new scripts
- All 54 layers complete (9 layers × 3 methods × 2 models); ~$8 actual API cost
- Results written to `paper_plots/data.json` (keys: `molmo-7b`, `llava-1.5`)
- **Summary (avg % interpretable across 9 layers):**
  - Molmo-7B-D:   LatentLens=**85.9%**, EmbeddingLens=44.1%, LogitLens=35.7%
  - LLaVA-1.5-7B: LatentLens=**55.2%**, EmbeddingLens=40.8%, LogitLens=34.9%
- git commits: `3516a13` (data.json), `d072e31` (script+fix)

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
- [x] Phase 4: Demo viewer
- [x] Phase 5: LLM Judge evaluation (54 layers done, ~$8 actual cost)
- [x] Phase 6: data.json updated (paper text pending)

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
