#!/bin/bash

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define the LLMs and vision encoders
# LLMS=("llama3-8b" "olmo-7b" "qwen2-7b")
# VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")
LLMS=("olmo-7b")
VISION_ENCODERS=("vit-l-14-336")

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4
MASTER_PORT=29526

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/minimal_val_captions.py"

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
        
        # Check if captions already exist
        output_file="analysis_results/captions/${checkpoint_name}_step12000-unsharded/generated_captions.json"
        if [ -f "$output_file" ]; then
            echo "=========================================="
            echo "Captions already exist for: ${checkpoint_name}"
            echo "Skipping. Delete the file to regenerate."
            echo "=========================================="
            echo ""
            continue
        fi
        
        echo "=========================================="
        echo "Generating captions for: ${llm} + ${vision_encoder}"
        echo "Checkpoint: ${checkpoint_path}"
        echo "=========================================="
        
        # Run caption generation
        torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
            "$SCRIPT_PATH" \
            --ckpt-path "$checkpoint_path" \
            --num-images 300 \
            --max-tokens 500 \
            --output-base-dir "analysis_results/captions"
        
        # Check if command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for ${llm} + ${vision_encoder}"
        else
            echo "SUCCESS: Completed ${llm} + ${vision_encoder}"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "All captions generated!"
echo "=========================================="

