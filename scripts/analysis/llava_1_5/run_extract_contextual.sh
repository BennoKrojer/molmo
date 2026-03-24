#!/bin/bash
# Extract contextual embeddings from LLaVA-1.5-7B's finetuned Vicuna backbone.
# Uses 4 GPUs in parallel (8 shards, 4 at a time).
# Run from repo root: bash scripts/analysis/llava_1_5/run_extract_contextual.sh

cd "$(dirname "$0")/../../.."
source /home/nlp/users/bkroje/vl_embedding_spaces/env/bin/activate

SCRIPT="scripts/analysis/llava_1_5/create_contextual_embeddings.py"
NUM_SHARDS=8
GPUS=(0 1 2 6)  # Use GPUs 0-2,6 (skip GPU 3, in use)

echo "=== LLaVA-1.5-7B Contextual Embedding Extraction ==="
echo "Shards: $NUM_SHARDS, GPUs: ${GPUS[*]}"
echo "Started: $(date)"

# Run first batch of 4 shards
echo "--- Launching shards 0-3 ---"
for i in 0 1 2 3; do
    gpu_idx=${GPUS[$i]}
    echo "Shard $i on GPU $gpu_idx"
    CUDA_VISIBLE_DEVICES=$gpu_idx python $SCRIPT \
        --dataset vg --num-captions -1 \
        --shard $i --num-shards $NUM_SHARDS \
        --embedding-dtype float8 \
        --layers 1 2 4 8 16 24 30 31 &
done
wait
echo "Shards 0-3 complete: $(date)"

# Run second batch of 4 shards
echo "--- Launching shards 4-7 ---"
for i in 4 5 6 7; do
    gpu_idx=${GPUS[$((i - 4))]}
    echo "Shard $i on GPU $gpu_idx"
    CUDA_VISIBLE_DEVICES=$gpu_idx python $SCRIPT \
        --dataset vg --num-captions -1 \
        --shard $i --num-shards $NUM_SHARDS \
        --embedding-dtype float8 \
        --layers 1 2 4 8 16 24 30 31 &
done
wait
echo "Shards 4-7 complete: $(date)"

# Merge shards and build caches
echo "--- Merging shards ---"
python $SCRIPT --merge-shards --num-shards $NUM_SHARDS --dataset vg

echo "=== DONE: $(date) ==="
