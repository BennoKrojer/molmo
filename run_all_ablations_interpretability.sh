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

# Interpretability parameters
NN_LAYER=0
CONTEXTUAL_LAYER=8
VISUAL_LAYER=0
THRESHOLD=1.5
OUTPUT_DIR="analysis_results/interpretability_heuristic/ablations"

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/compute_interpretability_heuristic.py"

echo "=========================================="
echo "Computing Interpretability Heuristic for Ablations"
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

# Loop through all ablation checkpoints
for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    total_models=$((total_models + 1))
    
    # Check if checkpoint exists
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint_path"
        echo "Skipping ${checkpoint_name}"
        failed_models=$((failed_models + 1))
        continue
    fi
    
    echo "=========================================="
    echo "[${total_models}/${#ABLATION_CHECKPOINTS[@]}] Processing ablation: ${checkpoint_name}"
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
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run analysis
    python3 "$SCRIPT_PATH" \
        --checkpoint-name "$checkpoint_name" \
        --checkpoint-base "molmo_data/checkpoints/ablations" \
        --nn-layer $NN_LAYER \
        --contextual-layer $CONTEXTUAL_LAYER \
        --visual-layer $VISUAL_LAYER \
        --threshold $THRESHOLD \
        --output-dir "$OUTPUT_DIR"
    
    # Check if command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: Completed ${checkpoint_name}"
        successful_models=$((successful_models + 1))
    else
        echo "❌ ERROR: Failed for ${checkpoint_name}"
        failed_models=$((failed_models + 1))
    fi
    
    echo ""
done

echo "=========================================="
echo "All Ablations Processed!"
echo "=========================================="
echo "Total models: ${total_models}"
echo "Successful: ${successful_models}"
echo "Failed: ${failed_models}"
echo "=========================================="

