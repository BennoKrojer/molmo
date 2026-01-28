#!/bin/bash

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base path for contextual embeddings
CONTEXTUAL_BASE="molmo_data/contextual_llm_embeddings_vg"

# Number of workers for cache building
# NOTE: Each layer has 300k+ tiny .npy files. Too many workers causes disk contention.
# Recommended: 2-4 workers. More workers may actually be SLOWER due to disk thrashing.
NUM_WORKERS=4

echo "=========================================="
echo "Step 1: Precomputing Embedding Caches"
echo "=========================================="
echo ""

# Run the cache precomputation script
python scripts/analysis/precompute_contextual_caches.py \
    --contextual-base "$CONTEXTUAL_BASE" \
    --num-workers $NUM_WORKERS

# Check if cache building succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Cache precomputation failed!"
    echo "Fix the errors above before running the analysis."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running Contextual NN Analysis"
echo "=========================================="
echo ""

# Run the main analysis script
./run_all_combinations_contextual_nn.sh vg

echo ""
echo "=========================================="
echo "All Done!"
echo "=========================================="

