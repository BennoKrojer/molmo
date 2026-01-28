# Repository Cleanup Candidates

Audit performed: 2026-01-03

This document lists files/folders that appear to be legacy, redundant, or candidates for removal.
**Review carefully before deleting anything!**

---

## 1. DEFINITELY SAFE TO REMOVE (test/debug files)

### Root-level test files
```
test_preprocessing_pipeline.py
test_fsdp_openvision.py
test_real_data_pipeline.py
test_eval_mode_fix.py
test_matching_layers.sh
openvision2_test.py
quick_check_layer_corruption.py
```

### scripts/ folder test files
```
scripts/test_inference_match.py
scripts/test_leftright_dataset.py
scripts/test_inference_preprocessing.py
scripts/test_single_inference.py
scripts/test_checkpoint_loading.py
```

### scripts/analysis/ debug files
```
scripts/analysis/debug_contextual_nearest_neighbors_allLayers.py
scripts/analysis/debug_embedding_comparison.py
scripts/analysis/debug_allLayers_bug.py
scripts/analysis/compare_debug_vs_alllayers.py
scripts/analysis/compare_old_vs_new_results.py
scripts/analysis/test_distributed_search.py
```

### llm_judge debug files
```
llm_judge/debug_single_contextual.sh
llm_judge/test.json
```

### paper_plots test file
```
paper_plots/test_run.py
```

---

## 2. REDUNDANT SCRIPT VERSIONS (older versions superseded)

### scripts/analysis/ - Multiple versions of same functionality
| Keep (Primary) | Remove (Legacy/Older) |
|----------------|----------------------|
| `contextual_nearest_neighbors_allLayers_singleGPU.py` | `contextual_nearest_neighbors.py` (single layer) |
| `contextual_nearest_neighbors_allLayers_singleGPU.py` | `contextual_nearest_neighbors_allLayers.py` (multi-GPU) |
| `contextual_nearest_neighbors_allLayers_singleGPU.py` | `contextual_nearest_neighbors_allLayers_slower.py` |
| `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` | `general_and_nearest_neighbors_pixmo_cap.py` |
| `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` | `general_and_nearest_neighbors.py` |

### Interactive viewers (unused?)
```
scripts/analysis/interactive_contextual_nearest_neighbors_viewer.py
scripts/analysis/interactive_logitlens_viewer.py
scripts/analysis/interactive_nearest_neighbors_viewer_pixmo_cap.py
scripts/analysis/interactive_nearest_neighbors_viewer.py
```
→ Check if these are used or replaced by the HTML viewer

---

## 3. RUN SCRIPTS - Many may be redundant

### Understanding the run script landscape:
- **`run_all_combinations_*.sh`** = Main 3x3 models (9 combinations)
- **`run_all_missing.sh`** = Ablations + Qwen2-VL (NOT main models!)
- **`run_all_ablations_*.sh`** = Older scripts for ablations (superseded by run_all_missing.sh)

### Superseded by `run_all_missing.sh` (ablation scripts):
```
run_all_ablations_nn.sh              → Phase 1 in run_all_missing.sh
run_all_ablations_logitlens.sh       → Phase 2 in run_all_missing.sh
run_all_ablations_contextual_nn.sh   → Phase 3 in run_all_missing.sh
```

### Still needed for main 3x3 models:
```
run_all_combinations_nn.sh           → KEEP (main models)
run_all_combinations_logitlens.sh    → KEEP (main models)
run_all_combinations_contextual_nn.sh → KEEP (main models)
```

### REFACTORING OPPORTUNITY: `run_all_missing.sh` is 958 lines!
Consider splitting into:
- `run_ablations.sh` - just ablations
- `run_qwen2vl.sh` - just Qwen2-VL  
- `run_all_missing.sh` - thin wrapper calling both

### One-time use scripts (already executed?)
```
run_missing_nn_layers.sh
run_qwen_layers_25-27_logitlens.sh
run_parallel_sameToken_similarity.sh
run_parallel_sameToken_text_similarity.sh
```

### Unused/legacy run scripts
```
run_general_nn_scripts.sh            → Contains commented-out examples only
run_contextual_nn.sh                 → Contains commented-out examples only
run_logitlens.sh                     → Contains commented-out examples only
run_main_training.sh                 → Contains commented-out examples only
run_all_combinations_contextual_nn_matching_layers.sh  → Experimental?
run_all_ablations_overlap_comparison.sh  → ?
run_all_combinations_interpretability.sh → ?
run_all_ablations_interpretability.sh   → ?
```

### Captioning scripts (still needed?)
```
run_all_ablations_captions.sh
run_all_combinations_captions.sh
run_all_ablations_captioning_metric_from_captions.sh
run_all_combinations_captioning_metric.sh
run_captioning_metric.sh
```

---

## 4. ONE-TIME USE SCRIPTS (HuggingFace upload/cleanup)

```
upload_hf.py
upload_hf_sharing.py
upload_contextual_nn.py
cleanup_hf_repo.py
delete_all_hf_repo.py
check_hf_status.py
download_hf_ckpt.py
```
→ Keep if you might need to re-upload; otherwise remove

---

## 5. MISCELLANEOUS CLEANUP CANDIDATES

### Root-level files that may not be needed
```
fix_corrupted_json.py           → One-time fix script
fix_viewer_files.py             → One-time fix script
make_htmls.sh                   → Legacy viewer generation?
monitor_parallel.sh             → Debug script
precompute_and_run_contextual_nn.sh → Legacy workflow
build_contextual_caches.sh      → Legacy (use precompute_contextual_caches.py)
analyze_caption_lengths.py      → One-time analysis
check_first_sentence_captions.py → One-time analysis
save_validation_samples.py      → One-time utility
comparison_sample.json          → Test data
translation_cache.json          → Can be regenerated
vg_phrases.txt                  → Can be regenerated?
regenerate_qwen2vl.sh           → Superseded by run_all_missing.sh
```

### Paper folder legacy
```
paper/outdated/                 → Old paper versions (can archive)
```

### scripts/ folder exploratory files
```
scripts/explore_pixmo_points.py
scripts/explore_simple_leftright.py
scripts/visualize_left_right_results.py
scripts/visualize_left_right_training_data.py
scripts/count_left_right_examples.py
scripts/move_ablation_captioning_results.py
scripts/merge_captions_into_nn_json.py
```

---

## 6. CONFIGS CLEANUP

### configs/rest/ - These are ablation configs
Keep: All ablation configs are referenced in run_all_missing.sh

### Unused configs?
```
configs/baseline_pixmo-captions_*_openvision2-l-14-336.yaml  → OpenVision2 experiments
```
→ Check if these models were ever trained

---

## 7. FOLDERS TO REVIEW

### hf_sharing/
Contains sample data for HuggingFace upload. Can be removed after upload is complete.

### downstream_eval/
Contains a single evaluation result. Archive or remove if not needed.

### logs/
Can be cleaned periodically (keep recent logs only).

---

## RECOMMENDED CLEANUP ORDER

1. **Phase 1**: Remove all test/debug files (safe)
2. **Phase 2**: Remove redundant script versions (careful)
3. **Phase 3**: Archive one-time scripts to `archive/` folder
4. **Phase 4**: Clean up run scripts (keep `run_all_missing.sh` + `generate_demo.sh` as primary)
5. **Phase 5**: Clean up paper/outdated/

---

## SCRIPTS REFERENCED IN README (KEEP!)

### Analysis scripts (core functionality):
```
scripts/analysis/create_unified_viewer.py          # Demo viewer
scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py  # Static NN
scripts/analysis/logitlens.py                      # LogitLens
scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py   # LN-Lens
scripts/analysis/create_contextual_embeddings.py   # Contextual embeddings
scripts/analysis/precompute_contextual_caches.py   # Cache precomputation
scripts/analysis/generate_ablation_viewers.py      # Ablation viewers
scripts/analysis/add_models_to_viewer.py           # Link models to index
scripts/train.py                                   # Training
```

### Run scripts:
```
generate_demo.sh                     # Unified demo generation
run_all_missing.sh                   # Ablations + Qwen2-VL (958 lines - needs refactoring!)
run_all_combinations_nn.sh           # Main 3x3 models - NN
run_all_combinations_logitlens.sh    # Main 3x3 models - LogitLens
run_all_combinations_contextual_nn.sh # Main 3x3 models - LN-Lens
```

### LLM Judge scripts:
```
llm_judge/run_all_parallel_nn.sh
llm_judge/run_all_parallel_logitlens.sh
llm_judge/run_all_parallel_contextual.sh
llm_judge/run_all_parallel_nn_ablations.sh
```

---

## REGENERATION STATUS

**Note**: The Qwen2-VL regeneration that was running appears to have been interrupted.
When you return, restart with:
```bash
./run_all_missing.sh --qwen2vl-only --force-qwen2vl
```


