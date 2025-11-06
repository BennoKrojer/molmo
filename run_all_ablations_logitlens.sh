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

# Define the layers to analyze (as comma-separated groups or individual layers)
# Using groups of layers for efficiency
LAYER_GROUPS=("0,4,8,12" "16,20,24,28" "1,2,3" "29,30,31,32")

# Logit lens parameters
TOP_K=5
NUM_IMAGES=300

# CUDA settings
export CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC=4
MASTER_PORT=29526

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/logitlens.py"

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
    
    # Loop through all layer groups
    for layers in "${LAYER_GROUPS[@]}"; do
        echo "Running logit lens for layers: ${layers}..."
        
        torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
            "$SCRIPT_PATH" \
            --ckpt-path "$checkpoint_path" \
            --layers "$layers" \
            --top-k $TOP_K \
            --num-images $NUM_IMAGES \
            --output-dir "analysis_results/logitlens/ablations"
        
        # Check if command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for ${checkpoint_name} at layers ${layers}"
        else
            echo "SUCCESS: Completed ${checkpoint_name} at layers ${layers}"
        fi
    done
    
    echo ""
done

echo "=========================================="
echo "All ablations processed!"
echo "=========================================="

