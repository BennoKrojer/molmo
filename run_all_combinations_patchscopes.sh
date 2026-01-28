#!/bin/bash
# Run Patchscopes Identity analysis on all model combinations
#
# Patchscopes (Ghandeharioun et al., ICML 2024) uses an identity prompt
# to decode what token is encoded in a hidden representation.
#
# Identity prompt: "cat->cat; 1135->1135; hello->hello; ?"
# Method: l→l patching (same source and target layer)

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define the LLMs and vision encoders
LLMS=("llama3-8b" "olmo-7b" "qwen2-7b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

# Parameters
TOP_K=5
NUM_IMAGES=100
BATCH_SIZE=16

# Standard layer set: 0, 1, 2, 4, 8, 16, 24, N-2, N-1
# For 32-layer models: 0,1,2,4,8,16,24,30,31
# For 28-layer models (Qwen2): 0,1,2,4,8,16,24,26,27
LAYERS_32="0,1,2,4,8,16,24,30,31"
LAYERS_28="0,1,2,4,8,16,24,26,27"

# CUDA settings (single GPU)
export CUDA_VISIBLE_DEVICES=0

# Base script path
SCRIPT_PATH="scripts/analysis/patchscopes/patchscopes_identity.py"

echo "=========================================="
echo "PATCHSCOPES IDENTITY ANALYSIS"
echo "=========================================="
echo "Identity prompt: cat->cat; 1135->1135; hello->hello; ?"
echo "Method: l→l patching"
echo "Images: $NUM_IMAGES"
echo "Top-k: $TOP_K"
echo "=========================================="
echo ""

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

        # Select layers based on model
        if [ "$llm" == "qwen2-7b" ]; then
            LAYERS=$LAYERS_28
        else
            LAYERS=$LAYERS_32
        fi

        echo "=========================================="
        echo "Processing: ${llm} + ${vision_encoder}"
        echo "Checkpoint: ${checkpoint_path}"
        echo "Layers: ${LAYERS}"
        echo "=========================================="

        python "$SCRIPT_PATH" \
            --ckpt-path "$checkpoint_path" \
            --layers "$LAYERS" \
            --top-k $TOP_K \
            --num-images $NUM_IMAGES \
            --batch-size $BATCH_SIZE

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
echo "All combinations processed!"
echo "Results saved to: analysis_results/patchscopes/"
echo "=========================================="
