#!/bin/bash

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define ablation checkpoints
ABLATION_CHECKPOINTS=(
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    # "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    # "train_mlp-only_pixmo_points_resize_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze"
)

# CUDA settings
export CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC=4
MASTER_PORT=29527

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/minimal_val_captions.py"

# Loop through all ablation checkpoints
for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint_path"
        echo "Skipping ${checkpoint_name}"
        continue
    fi
    
    # Check if captions already exist
    output_file="analysis_results/captions/ablations/${checkpoint_name}_step12000-unsharded/generated_captions.json"
    if [ -f "$output_file" ]; then
        echo "=========================================="
        echo "Captions already exist for: ${checkpoint_name}"
        echo "Skipping. Delete the file to regenerate."
        echo "=========================================="
        echo ""
        continue
    fi
    
    echo "=========================================="
    echo "Generating captions for: ${checkpoint_name}"
    echo "Checkpoint: ${checkpoint_path}"
    echo "=========================================="
    
    # Run caption generation
    torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
        "$SCRIPT_PATH" \
        --ckpt-path "$checkpoint_path" \
        --num-images 300 \
        --max-tokens 500 \
        --output-base-dir "analysis_results/captions/ablations"
    
    # Check if command succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for ${checkpoint_name}"
    else
        echo "SUCCESS: Completed ${checkpoint_name}"
    fi
    
    echo ""
done

echo "=========================================="
echo "All ablation captions generated!"
echo "=========================================="

