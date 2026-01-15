# Qwen2-VL Analysis Scripts

This folder contains analysis scripts for off-the-shelf Qwen2-VL models from HuggingFace.

## Key Findings

### Vision Token Processing in Qwen2-VL

Based on our investigation, Qwen2-VL processes vision tokens differently than Molmo:

1. **Merger Module**: Qwen2-VL has a "merger" in the visual encoder that reduces patches
   - Example: 256 patches (16×16) → 64 vision tokens in the LLM
   - This is different from Molmo which preserves more spatial resolution

2. **Vision Token Positions**: Vision tokens are marked by `<|image_pad|>` tokens (ID: 151655)
   - Vision tokens appear at positions 0 to (num_image_pad - 1) in hidden states
   - Number of vision tokens varies by image size (e.g., 64, 256, 777, 999 tokens)

3. **Architecture**:
   ```
   model.visual         - Vision encoder (Qwen2VisionTransformerPretrainedModel)
     ├── patch_embed    - Patch embedding
     ├── rotary_pos_emb - Rotary position embeddings  
     ├── blocks         - Transformer blocks
     └── merger         - Reduces patches to fewer tokens
   
   model.model          - LLM (Qwen2VLModel)
     ├── embed_tokens   - Token embeddings (152064, 3584)
     ├── layers         - 28 transformer layers
     ├── norm           - Final layer norm
     └── rotary_emb     - Rotary embeddings
   ```

4. **Hidden States**: 29 layers total (embedding layer + 28 transformer layers)
   - Hidden dimension: 3584
   - Vision token norms increase from ~35-55 at early layers to ~200-240 at later layers

## Scripts

### `extract_vision_tokens_allLayers_singleGPU.py` ⭐ (Main Script)

Extract vision token features from ALL LLM layers. This is the Qwen2-VL equivalent of 
`contextual_nearest_neighbors_allLayers_singleGPU.py` for Molmo.

**Usage:**
```bash
source ../../env/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)

# Extract features from selected layers
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_allLayers_singleGPU.py \
    --num-images 100 \
    --layers "0,8,16,24,28" \
    --output-dir analysis_results/qwen2_vl/vision_features

# Extract from all layers
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_allLayers_singleGPU.py \
    --num-images 100 \
    --layers "all" \
    --output-dir analysis_results/qwen2_vl/vision_features

# Also save full feature tensors (large files)
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_allLayers_singleGPU.py \
    --num-images 100 \
    --layers "0,8,16,24,28" \
    --save-features \
    --output-dir analysis_results/qwen2_vl/vision_features
```

**Output:**
- `extraction_summary_*.json`: Per-image statistics for each layer
- `features_layer*.pt`: Full feature tensors (if `--save-features`)

### `extract_vision_tokens_all_layers.py`

Simpler extraction script with test mode.

**Usage:**
```bash
# Test extraction with synthetic image
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_all_layers.py --test-only

# Extract from real images
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_all_layers.py \
    --num-images 10 --use-real-images
```

### `investigate_vision_tokens.py`

Deep investigation script to understand Qwen2-VL's vision token processing.

### `inspect_model_structure.py` / `find_vision_encoder.py` / `test_vision_extraction.py`

Debug/investigation scripts used to understand the model architecture.

### `contextual_nearest_neighbors.py`

Finds nearest contextual text embeddings for visual tokens (multi-GPU version).
Requires pre-computed contextual embeddings from the underlying Qwen2 LLM.

**Note**: Since Qwen2-VL uses a finetuned LLM, contextual embeddings from the base Qwen2 LLM
may not be perfectly aligned. This is a known limitation.

## Prerequisites

1. **Install required packages:**
   ```bash
   pip install transformers qwen-vl-utils
   ```

2. **Environment setup:**
   ```bash
   source ../../env/bin/activate
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## Differences from Molmo Analysis

| Aspect | Molmo | Qwen2-VL |
|--------|-------|----------|
| Model Loading | Custom checkpoint | HuggingFace |
| Vision Tokens | ~576 tokens (24×24) | Variable (64-1000+) |
| Token Reduction | Uses `use_n_token_only` | Built-in merger |
| Position Tracking | `image_input_idx` tensor | `<\|image_pad\|>` markers |
| Hidden Dim | Varies by model | 3584 (7B) |
| Num Layers | Varies by model | 29 (28 transformer + 1 embed) |

## Contextual Embeddings for Qwen2-VL

Since Qwen2-VL uses a **finetuned** version of Qwen2's LLM (not vanilla Qwen2), we need to
extract contextual embeddings from Qwen2-VL's LLM backbone directly. This ensures the text
embeddings are properly aligned with the vision tokens.

### `create_contextual_embeddings_qwen2vl.py`

Extract contextual text embeddings from Qwen2-VL's LLM backbone (pure text, no vision).

**Quick test (100 captions):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py \
    --num-captions 100 --layers 8 16 24 --batch-size 8
```

**Full extraction (same layers as vanilla Qwen2: 1,2,4,8,16,24,26,27):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py \
    --num-captions 1000000 \
    --layers 1 2 4 8 16 24 26 27 \
    --batch-size 32 \
    --output-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-VL-7B-Instruct
```

**After extraction, build caches for fast loading:**
```bash
python scripts/analysis/precompute_contextual_caches.py \
    --contextual-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-VL-7B-Instruct \
    --num-workers 1
```

## Next Steps

1. **Nearest Neighbor Analysis**: Use extracted features to find nearest neighbors in:
   - Static text embedding space (vocabulary)
   - Contextual text embedding space (from Qwen2-VL's LLM backbone)

2. **Layer Evolution Analysis**: Analyze how vision tokens evolve across layers

3. **Interpretability**: Check if vision tokens are interpretable using LatentLens methods

## Notes

- GPU Memory: ~15GB for Qwen2-VL-7B-Instruct
- Processing Speed: ~1.5 images/second (single GPU)
- The model uses dynamic resolution - larger images have more vision tokens
