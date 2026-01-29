# V-Lens: Interpreting Vision Tokens in LLMs

Code for "The surprising interpretability of vision tokens in LLMs".

> **Note**: This repo is forked from [Molmo](https://github.com/allenai/molmo). Most training infrastructure comes from there. This README focuses on our interpretability study.

## Overview

We study how frozen LLMs process visual soft prompt tokens from vision encoders. We train connector-only models (MLP) mapping vision tokens to LLM embedding space, then analyze interpretability using three methods:

1. **Input Embedding Matrix** - Nearest neighbors in LLM input embedding matrix
2. **LogitLens** - Applying LM head to intermediate representations (output embedding matrix)
3. **LN-Lens (ours)** - Nearest neighbors in contextual text embeddings

## Setup

```bash
# Create environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Set paths
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create symlinks to data storage (CRITICAL - required for all scripts)
ln -s /mnt/research/scratch/bkroje/molmo_data molmo_data
ln -s /mnt/research/scratch/bkroje/analysis_results analysis_results
```

> **Note**: Both symlinks are required. Without `molmo_data`, training/analysis scripts will fail.
> Without `analysis_results`, viewer generation will fail.

---

## 1. Training

### Main Study: 3×3 Model Grid

| | ViT-L/14-336 (CLIP) | DINOv2-L-336 | SigLIP |
|---|---|---|---|
| **LLaMA3-8B** | ✓ | ✓ | ✓ |
| **OLMo-7B** | ✓ | ✓ | ✓ |
| **Qwen2-7B** | ✓ | ✓ | ✓ |

**Training details:**
- Dataset: PixMo-Cap (captioning)
- Steps: 12,000
- Frozen: LLM + Vision Encoder
- Trained: MLP connector only
- Batch size: 8 (global)

```bash
# Example: Train OLMo-7B + ViT-L/14-336
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py \
    configs/baseline_pixmo-captions_olmo-7b_vit-l-14-336.yaml
```

Checkpoints saved to: `molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_{llm}_{vision}/`

> **Note**: `qwen2-7b_vit-l-14-336` uses `_seed10` suffix.

### Ablations

Located in `molmo_data/checkpoints/ablations/` and `configs/rest/`:

| Ablation | Description |
|---|---|
| `first-sentence` | Train on first sentence of captions only |
| `linear` | Linear connector instead of MLP |
| `seed10`, `seed11` | Different random seeds |
| `unfreeze` | Fine-tune LLM (not frozen) |
| `earlier-vit-layers-6`, `earlier-vit-layers-10` | Use earlier ViT layers |
| `pixmo_topbottom` | Different task (top/bottom localization) |

### Off-the-Shelf Models

We also analyze pre-trained models without our connector training:

- **Qwen2-VL-7B-Instruct** - See `scripts/analysis/qwen2_vl/`

---

## 2. Contextual Embeddings Dataset

We extract contextual text embeddings from LLMs processing Visual Genome phrases. These serve as candidates for LN-Lens.

**Location:** `molmo_data/contextual_llm_embeddings_vg/`

| LLM | Layers | Unique Tokens | Embeddings/Layer |
|---|---|---|---|
| OLMo-7B | 1, 2, 4, 8, 16, 24, 30, 31 | ~26k | ~300k |
| LLaMA3-8B | 1, 2, 4, 8, 16, 24, 30, 31 | ~26k | ~300k |
| Qwen2-7B | 1, 2, 4, 8, 16, 24, 26, 27 | ~26k | ~300k |
| Qwen2-VL-7B | 1, 2, 4, 8, 16, 24, 26, 27 | ~26k | ~300k |

```bash
# Extract contextual embeddings for an LLM
python scripts/analysis/create_contextual_embeddings.py \
    --model Qwen/Qwen2-7B \
    --dataset vg \
    --layers 1 2 4 8 16 24 26 27 \
    --num-captions -1  # all phrases

# Build search cache (required before running LN-Lens)
python scripts/analysis/precompute_contextual_caches.py \
    --contextual-dir molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B
```

**Storage:** Float8 compression (~25% of float32 size).

---

## 3. Interpretability Tools

All tools analyze vision tokens across layers: **0, 1, 2, 4, 8, 16, 24, N-2, N-1**  
(where N=32 for OLMo/LLaMA, N=28 for Qwen2)

### Input Embedding Matrix (Vocabulary Nearest Neighbors)

Finds nearest neighbors in LLM input embedding matrix.

```bash
# Run on all 9 model combinations
./run_all_combinations_nn.sh

# Or single model:
torchrun --nproc_per_node=4 scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py \
    --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded \
    --llm_layer 0,1,2,4,8,16,24,30,31
```

**Output:** `analysis_results/nearest_neighbors/`

### LogitLens

Applies LM head to intermediate layer representations.

```bash
./run_all_combinations_logitlens.sh
```

**Output:** `analysis_results/logit_lens/`

### LN-Lens (Contextual Nearest Neighbors)

Our main method. Finds nearest neighbors in contextual text embeddings.

```bash
# RECOMMENDED: Parallel single-GPU approach (8x faster, uses all 8 GPUs independently)
./run_parallel_contextual_nn.sh vg

# Alternative: Sequential torchrun approach
./run_all_combinations_contextual_nn.sh vg
```

The parallel approach (`run_parallel_contextual_nn.sh`) launches 8 independent single-GPU jobs, avoiding FSDP synchronization overhead. Each job uses `scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py` which preloads images to GPU for maximum throughput.

**Output:** `analysis_results/contextual_nearest_neighbors/` - **Primary dataset, use this one.** Contains all 9 models + `_lite10` versions (demo images 0-9). The `_vg/` variant is deprecated/incomplete.

---

## 4. Evaluation: LLM Judge

GPT-4o evaluates whether nearest neighbor tokens are semantically meaningful for image patches.

```bash
# Run on all model combinations
./llm_judge/run_all_parallel_nn.sh          # Input Embedding Matrix
./llm_judge/run_all_parallel_logitlens.sh   # LogitLens
./llm_judge/run_all_parallel_contextual.sh  # LN-Lens

# Ablations
./llm_judge/run_all_parallel_nn_ablations.sh
```

**Output:** `analysis_results/llm_judge_*/`

Each evaluation samples patches, sends cropped regions + candidate tokens to GPT-4o, and measures accuracy.

---

## 5. Analysis & Visualization

### Interactive Demo

Explore all interpretability results in a unified HTML viewer:

```bash
# Generate complete demo (main + ablations) with ONE command
./generate_demo.sh --num-images 10

# Or with custom output directory
./generate_demo.sh --output-dir analysis_results/my_demo --num-images 10

# Or run Python script directly (same result)
python scripts/analysis/create_unified_viewer.py \
    --output-dir analysis_results/unified_viewer_lite \
    --num-images 10
```

Open `analysis_results/unified_viewer_lite/index.html` in a browser.

**What the demo includes:**
- 9 main model viewers (3 LLMs × 3 Vision Encoders)
- 10 ablation model viewers (automatically linked if they exist in `ablations/` folder)
- Unified index.html with navigation to all models

**NOTE:** `create_unified_viewer.py` now handles ablations automatically - you do NOT need
to run multiple scripts. Use `--no-ablations` flag to skip ablations section if needed.

**Individual scripts (for advanced use only):**
- `generate_ablation_viewers.py` - Generate ablation image viewers (run once, then `create_unified_viewer.py` links them)
- `add_models_to_viewer.py` - Legacy script, functionality now in `create_unified_viewer.py`

### Layer Evolution Analysis

Analyze how interpretability changes across layers:

```bash
# Located in scripts/analysis/layer_evolution/
python scripts/analysis/layer_evolution/analyze_concreteness.py
python scripts/analysis/layer_evolution/analyze_visual_attributes.py
```

---

## Directory Structure

```
molmo_data/
├── checkpoints/                    # Trained model checkpoints
│   ├── train_mlp-only_pixmo_cap_resize_{llm}_{vision}/
│   └── ablations/
├── contextual_llm_embeddings_vg/   # Contextual embeddings (VG corpus)
│   ├── allenai_OLMo-7B-1024-preview/
│   ├── meta-llama_Meta-Llama-3-8B/
│   ├── Qwen_Qwen2-7B/
│   └── Qwen_Qwen2-VL-7B-Instruct/

analysis_results/
├── nearest_neighbors/              # Input Embedding Matrix results
├── logit_lens/                     # LogitLens results
├── contextual_nearest_neighbors/   # LN-Lens results (CC corpus)
├── contextual_nearest_neighbors_vg/# LN-Lens results (VG corpus)
├── llm_judge_*/                    # GPT-4o evaluation results
├── unified_viewer_lite/            # HTML visualization
└── ablations_comparison/           # Ablation analysis

scripts/analysis/
├── create_contextual_embeddings.py
├── general_and_nearest_neighbors_pixmo_cap_multi-gpu.py
├── logitlens.py
├── contextual_nearest_neighbors_allLayers_singleGPU.py
├── create_unified_viewer.py
├── qwen2_vl/                       # Qwen2-VL specific scripts
└── layer_evolution/                # Layer-wise analysis
```

---

## Reproducing Results

Full pipeline for one model:

```bash
# 1. Train connector
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py \
    configs/baseline_pixmo-captions_olmo-7b_vit-l-14-336.yaml

# 2. Extract contextual embeddings (if not already done for this LLM)
python scripts/analysis/create_contextual_embeddings.py \
    --model allenai/OLMo-7B-1024-preview --dataset vg --layers 1 2 4 8 16 24 30 31

# 3. Build cache
python scripts/analysis/precompute_contextual_caches.py \
    --contextual-dir molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview

# 4. Run interpretability tools
./run_all_combinations_nn.sh
./run_all_combinations_logitlens.sh
./run_all_combinations_contextual_nn.sh vg

# 5. Run LLM judge evaluation
./llm_judge/run_all_parallel_nn.sh
./llm_judge/run_all_parallel_logitlens.sh
./llm_judge/run_all_parallel_contextual.sh

# 6. Generate visualization (single command)
./generate_demo.sh --num-images 10
```

---

## Citation

```bibtex
@article{krojer2026latentlens,
  title={LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs},
  author={Krojer, Benno and Nayak, Shravan and Ma{\~n}as, Oscar and Adlakha, Vaibhav and Elliott, Desmond and Reddy, Siva and Mosbach, Marius},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

> **Note**: Update the arXiv ID after upload.


