#!/bin/bash

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define the LLMs and vision encoders
LLMS=("llama3-8b" "olmo-7b" "qwen2-7b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip" "openvision2-l-14-336")

# Interpretability parameters
NN_LAYER=0
CONTEXTUAL_LAYER=8
VISUAL_LAYER=0
THRESHOLD=1.5
OUTPUT_DIR="analysis_results/interpretability_heuristic"

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/compute_interpretability_heuristic.py"

echo "=========================================="
echo "Computing Interpretability Heuristic"
echo "=========================================="
echo "Parameters:"
echo "  NN Layer: ${NN_LAYER}"
echo "  Contextual Layer: ${CONTEXTUAL_LAYER}"
echo "  Visual Layer: ${VISUAL_LAYER}"
echo "  Threshold: ${THRESHOLD}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Track statistics
total_models=0
successful_models=0
failed_models=0

# Loop through all combinations
for llm in "${LLMS[@]}"; do
    for vision_encoder in "${VISION_ENCODERS[@]}"; do
        total_models=$((total_models + 1))
        
        # Special case: qwen2-7b with vit-l-14-336 uses seed10
        if [ "$llm" == "qwen2-7b" ] && [ "$vision_encoder" == "vit-l-14-336" ]; then
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}_seed10"
        else
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}"
        fi
        
        echo "=========================================="
        echo "[${total_models}/12] Processing: ${llm} + ${vision_encoder}"
        echo "Checkpoint: ${checkpoint_name}"
        echo "=========================================="
        
        # Check if output already exists
        output_file="${OUTPUT_DIR}/${checkpoint_name}_step12000-unsharded/interpretability_heuristic_nn${NN_LAYER}_contextual${CONTEXTUAL_LAYER}_threshold${THRESHOLD}.json"
        if [ -f "$output_file" ]; then
            echo "✓ Results already exist: $output_file"
            echo "SKIPPING"
            successful_models=$((successful_models + 1))
            echo ""
            continue
        fi
        
        # Run analysis
        python3 "$SCRIPT_PATH" \
            --checkpoint-name "$checkpoint_name" \
            --nn-layer $NN_LAYER \
            --contextual-layer $CONTEXTUAL_LAYER \
            --visual-layer $VISUAL_LAYER \
            --threshold $THRESHOLD \
            --output-dir "$OUTPUT_DIR"
        
        # Check if command succeeded
        if [ $? -eq 0 ]; then
            echo "✓ SUCCESS: Completed ${llm} + ${vision_encoder}"
            successful_models=$((successful_models + 1))
        else
            echo "❌ ERROR: Failed for ${llm} + ${vision_encoder}"
            failed_models=$((failed_models + 1))
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "All Combinations Processed!"
echo "=========================================="
echo "Total models: ${total_models}"
echo "Successful: ${successful_models}"
echo "Failed: ${failed_models}"
echo "=========================================="

