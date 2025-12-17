# Qwen2-VL Analysis Scripts

This folder contains analysis scripts for off-the-shelf Qwen2-VL models from HuggingFace.

## Overview

These scripts analyze how Qwen2-VL (an off-the-shelf vision-language model) processes visual tokens and how they relate to contextual text embeddings, similar to the analysis done for trained Molmo models.

## Prerequisites

1. **Install required packages:**
   ```bash
   pip install transformers qwen-vl-utils
   ```

2. **Create contextual embeddings** (if not already done):
   ```bash
   python scripts/analysis/create_contextual_embeddings.py \
       --model Qwen/Qwen2-7B \
       --layers 8 16 24 \
       --output-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B
   ```

3. **Precompute caches** (required for fast loading):
   ```bash
   python scripts/analysis/precompute_contextual_caches.py \
       --contextual-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B \
       --num-workers 1
   ```

## Scripts

### `contextual_nearest_neighbors.py`

Finds nearest contextual text embeddings for visual tokens extracted from Qwen2-VL.

**Usage:**

**IMPORTANT**: Must use `torchrun` (not `python3`) because the script uses distributed processing!

```bash
# Basic example (2 GPUs)
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29528 \
    scripts/analysis/qwen2_vl/contextual_nearest_neighbors.py \
    --model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --contextual-dir "molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B" \
    --contextual-layer "16" \
    --visual-layer 0 \
    --num-images 100 \
    --top-k 5

# Multiple contextual layers
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29528 \
    scripts/analysis/qwen2_vl/contextual_nearest_neighbors.py \
    --model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --contextual-dir "molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B" \
    --contextual-layer "8,16,24" \
    --visual-layer 0 \
    --num-images 300
```

**Arguments:**
- `--model-name`: HuggingFace model name (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--contextual-dir`: Directory with contextual embeddings (required)
- `--contextual-layer`: Layer index(es) to use, comma-separated (e.g., `8,16,24`)
- `--visual-layer`: Visual layer to extract (0 = vision encoder, >0 = LLM layer)
- `--num-images`: Number of images to process (default: 100)
- `--split`: Dataset split (`train` or `validation`, default: `validation`)
- `--top-k`: Number of nearest neighbors (default: 5)
- `--max-contextual-per-token`: Max embeddings per token for memory management
- `--output-dir`: Output directory (default: `analysis_results/qwen2_vl/contextual_nearest_neighbors`)

**Output:**
Results are saved as JSON files in the output directory, containing:
- For each image: visual token positions and their nearest contextual neighbors
- Similarity scores and metadata for each neighbor
- Inter-neighbor similarities

## Differences from Molmo Analysis

1. **Model Loading**: Qwen2-VL is loaded directly from HuggingFace, not from checkpoints
2. **Preprocessing**: Uses Qwen2-VL's processor instead of Molmo's preprocessor
3. **Visual Token Extraction**: Accesses vision encoder or LLM layers through HuggingFace API
4. **Architecture**: Qwen2-VL has a different architecture than Molmo, so extraction methods differ

## Notes

- The script uses distributed processing (multi-GPU) similar to the Molmo analysis scripts
- Visual token extraction may need adjustment based on the specific Qwen2-VL model version
- Ensure you have sufficient GPU memory for the model size (7B models require ~14GB per GPU)
- The script processes contextual layers sequentially to save memory

## Troubleshooting

**Issue**: Cannot extract vision features
- **Solution**: Check the model architecture. The script tries multiple methods to access vision encoder. You may need to inspect the model structure and adjust the extraction function.

**Issue**: Wrong number of visual tokens
- **Solution**: The script estimates visual token count. You may need to adjust `num_image_tokens` based on your specific model configuration or image preprocessing.

**Issue**: Out of memory
- **Solution**: Reduce `--num-images`, use `--max-contextual-per-token`, or use fewer GPUs.

