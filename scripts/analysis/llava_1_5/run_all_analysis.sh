#!/bin/bash
# Run all 3 interpretability methods for LLaVA-1.5-7B.
# Prerequisites: contextual embeddings must be extracted first.
# Run from repo root: bash scripts/analysis/llava_1_5/run_all_analysis.sh

cd "$(dirname "$0")/../../.."
source /home/nlp/users/bkroje/vl_embedding_spaces/env/bin/activate

GPU=${1:-0}
NUM_IMAGES=${2:-100}
LAYERS="0,1,2,4,8,16,24,30,31"
CTX_DIR="molmo_data/contextual_llm_embeddings_vg/llava-hf_llava-1.5-7b-hf"

echo "=== LLaVA-1.5-7B Full Analysis ==="
echo "GPU: $GPU, Images: $NUM_IMAGES, Layers: $LAYERS"
echo ""

# 1. EmbeddingLens (Nearest Neighbors)
echo "--- [1/3] EmbeddingLens ---"
CUDA_VISIBLE_DEVICES=$GPU python scripts/analysis/llava_1_5/nearest_neighbors.py \
    --num-images $NUM_IMAGES --layers "$LAYERS"

# 2. LogitLens
echo "--- [2/3] LogitLens ---"
CUDA_VISIBLE_DEVICES=$GPU python scripts/analysis/llava_1_5/logitlens.py \
    --num-images $NUM_IMAGES --layers "$LAYERS"

# 3. LatentLens (Contextual NN) - requires contextual embeddings
if [ -d "$CTX_DIR" ]; then
    echo "--- [3/3] LatentLens ---"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/analysis/llava_1_5/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --contextual-dir "$CTX_DIR" \
        --visual-layer "$LAYERS" --num-images $NUM_IMAGES
else
    echo "--- [3/3] LatentLens SKIPPED (contextual embeddings not ready) ---"
    echo "  Run: bash scripts/analysis/llava_1_5/run_extract_contextual.sh"
fi

echo ""
echo "=== DONE ==="
