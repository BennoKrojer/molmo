#!/bin/bash

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define the LLMs and vision encoders
LLMS=("llama3-8b" "olmo-7b" "qwen2-7b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip" "openvision2-l-14-336")

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

# Loop through all combinations
for llm in "${LLMS[@]}"; do
    for vision_encoder in "${VISION_ENCODERS[@]}"; do
        # Special case: qwen2-7b with vit-l-14-336 uses seed10
        if [ "$llm" == "qwen2-7b" ] && [ "$vision_encoder" == "vit-l-14-336" ]; then
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}_seed10"
        else
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}"
        fi
        
        checkpoint_path="molmo_data/checkpoints/${checkpoint_name}/step12000-unsharded"
        
        # Check if checkpoint exists
        if [ ! -d "$checkpoint_path" ]; then
            echo "WARNING: Checkpoint not found: $checkpoint_path"
            echo "Skipping ${llm} + ${vision_encoder}"
            continue
        fi
        
        echo "=========================================="
        echo "Processing: ${llm} + ${vision_encoder}"
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
                --num-images $NUM_IMAGES
            
            # Check if command succeeded
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed for ${llm} + ${vision_encoder} at layers ${layers}"
            else
                echo "SUCCESS: Completed ${llm} + ${vision_encoder} at layers ${layers}"
            fi
        done
        
        echo ""
    done
done

echo "=========================================="
echo "All combinations processed!"
echo "=========================================="

