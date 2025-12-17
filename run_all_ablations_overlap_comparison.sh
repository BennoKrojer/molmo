#!/bin/bash

source ../../env/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)

# Original checkpoint
ORIGINAL_CHECKPOINT="train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336"
ORIGINAL_JSON="analysis_results/nearest_neighbors/${ORIGINAL_CHECKPOINT}_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json"

# Ablation checkpoints
ABLATION_CHECKPOINTS=(
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    "train_mlp-only_pixmo_points_resize_olmo-7b_vit-l-14-336"
)

# Configuration
TOP_K=5
SIMILARITY_THRESHOLD=0.1
OUTPUT_BASE="analysis_results/ablations_comparison"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Running Overlap Comparison for All Ablations"
echo "=========================================="
echo "Original checkpoint: ${ORIGINAL_CHECKPOINT}"
echo "Top-K: ${TOP_K}"
echo "Similarity threshold: ${SIMILARITY_THRESHOLD}"
echo "Ablation checkpoints: ${#ABLATION_CHECKPOINTS[@]} total"
echo "=========================================="
echo ""

# Check if original JSON exists
if [ ! -f "$ORIGINAL_JSON" ]; then
    echo "ERROR: Original JSON not found: $ORIGINAL_JSON"
    exit 1
fi

# Loop through all ablation checkpoints
for ablation_checkpoint in "${ABLATION_CHECKPOINTS[@]}"; do
    ablation_json="analysis_results/nearest_neighbors/ablations/${ablation_checkpoint}_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json"
    
    # Check if ablation JSON exists
    if [ ! -f "$ablation_json" ]; then
        echo "WARNING: Ablation JSON not found: $ablation_json"
        echo "Skipping ${ablation_checkpoint}"
        continue
    fi
    
    echo "=========================================="
    echo "Comparing: ${ablation_checkpoint}"
    echo "=========================================="
    
    # Create output filename
    safe_checkpoint_name=$(echo "$ablation_checkpoint" | sed 's/[^a-zA-Z0-9_-]/_/g')
    output_file="${OUTPUT_BASE}/${safe_checkpoint_name}_vs_original_layer0.json"
    
    # Run comparison
    python3 scripts/analysis/ablations_comparison.py \
        --original "$ORIGINAL_JSON" \
        --ablation "$ablation_json" \
        --top-k $TOP_K \
        --similarity-threshold $SIMILARITY_THRESHOLD \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Comparison saved to $output_file"
    else
        echo "ERROR: Failed comparison for ${ablation_checkpoint}"
    fi
    
    echo ""
done

echo "=========================================="
echo "All comparisons complete!"
echo "=========================================="
echo ""
echo "Results are in: ${OUTPUT_BASE}/"
echo ""
echo "Each comparison file contains:"
echo "  - Mean overlap statistics"
echo "  - High similarity (>${SIMILARITY_THRESHOLD}) statistics"
echo "  - Low similarity (<=${SIMILARITY_THRESHOLD}) statistics"
echo "  - Overlap distributions"
echo "=========================================="

