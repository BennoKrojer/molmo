#!/bin/bash

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define ablation checkpoints
ABLATION_CHECKPOINTS=(
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    "train_mlp-only_pixmo_points_resize_olmo-7b_vit-l-14-336"
)

# Define the layers to analyze (passed as comma-separated to python script)
LAYERS="0,1,2,3,4,8,12,16,20,24,28,29,30,31"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4
MASTER_PORT=29525

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py"

# Loop through all ablation checkpoints
for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint_path"
        echo "Skipping ${checkpoint_name}"
        continue
    fi
    
    echo "=========================================="
    echo "Processing ablation: ${checkpoint_name}"
    echo "Checkpoint: ${checkpoint_path}"
    echo "=========================================="
    
    # Run all layers in a single call (model loaded once, layers processed sequentially)
    echo "Running analysis for layers: ${LAYERS} (model loaded once)..."
    
    torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
        "$SCRIPT_PATH" \
        --ckpt-path "$checkpoint_path" \
        --llm_layer "$LAYERS" \
        # --generate-captions \
        --output-base-dir "analysis_results/nearest_neighbors/ablations"
    
    # Check if command succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for ${checkpoint_name}"
    else
        echo "SUCCESS: Completed ${checkpoint_name}"
    fi
    
    echo ""
done

echo "=========================================="
echo "All ablations processed!"
echo "=========================================="

