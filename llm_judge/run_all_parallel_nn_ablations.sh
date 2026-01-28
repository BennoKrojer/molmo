#!/bin/bash
#
# Run LLM judge evaluation on all ablation variants in parallel
#

set -e

# Setup environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Read API key
API_KEY=$(cat llm_judge/api_key.txt)

# Configuration
NUM_IMAGES=300
NUM_SAMPLES=1
SPLIT="validation"
USE_CROPPED_REGION=true  # Set to true to use cropped region prompt
SEED=42  # Random seed for reproducibility
SKIP_IF_EXISTS=false  # Set to true to skip ANY existing runs (complete or incomplete)
SKIP_IF_COMPLETE=true  # Set to true to skip only complete runs (will regenerate incomplete)
RESUME=true  # Set to true to resume incomplete runs instead of regenerating from scratch
BASE_DIR="analysis_results/nearest_neighbors/ablations"
OUTPUT_BASE="analysis_results/llm_judge_nearest_neighbors/ablations"

# Ablation checkpoints
ABLATION_CHECKPOINTS=(
    # "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    # "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    # "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    # "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    # "train_mlp-only_pixmo_points_resize_olmo-7b_vit-l-14-336"
    # "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze"
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm"
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336"
)

# Layers to evaluate (focus on layer 0 for input layer interpretability)
LAYERS=(0)

echo "=========================================="
echo "Parallel LLM Judge Evaluation - Ablations"
echo "=========================================="
echo "Images per model: $NUM_IMAGES"
echo "Patches per image: $NUM_SAMPLES"
echo "Use cropped region: $USE_CROPPED_REGION"
echo "Skip if exists: $SKIP_IF_EXISTS"
echo "Skip if complete: $SKIP_IF_COMPLETE"
echo "Resume incomplete: $RESUME"
echo "Random seed: $SEED"
echo "Split: $SPLIT"
echo "Layers: ${LAYERS[@]}"
echo "Ablation checkpoints: ${#ABLATION_CHECKPOINTS[@]} total"
echo "Layers per model: ${#LAYERS[@]} (${LAYERS[*]})"
echo "Execution: Models in parallel, layers sequentially per model"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to process all layers for a single ablation checkpoint
process_ablation_layers() {
    local checkpoint_name="$1"
    
    # Check if nearest neighbors directory exists
    nn_dir="$BASE_DIR/${checkpoint_name}_step12000-unsharded"
    
    if [ ! -d "$nn_dir" ]; then
        echo "WARNING: Nearest neighbors directory not found: $nn_dir"
        echo "Skipping ${checkpoint_name}"
        return 1
    fi
    
    # All ablations use olmo-7b and vit-l-14-336
    local llm="olmo-7b"
    local encoder="vit-l-14-336"
    
    echo "Processing ${checkpoint_name}: ${#LAYERS[@]} layers (${LAYERS[*]})"
    
    # Run evaluation for each layer SEQUENTIALLY
    for layer in "${LAYERS[@]}"; do
        nn_file="$nn_dir/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer${layer}.json"
        
        if [ ! -f "$nn_file" ]; then
            echo "  Layer $layer: WARNING - File not found: $nn_file"
            echo "  Layer $layer: Skipping..."
            continue
        fi
        
        echo "  Layer $layer: Starting evaluation..."
        
        # Create a safe identifier for the log file
        safe_checkpoint_name=$(echo "$checkpoint_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
        
        python3 llm_judge/run_single_model_with_viz.py \
            --llm "$llm" \
            --vision-encoder "$encoder" \
            --checkpoint-name "$checkpoint_name" \
            --layer $layer \
            --api-key "$API_KEY" \
            --base-dir "$BASE_DIR" \
            --output-base "$OUTPUT_BASE" \
            --num-images $NUM_IMAGES \
            --num-samples $NUM_SAMPLES \
            --split "$SPLIT" \
            --seed $SEED \
            $([ "$USE_CROPPED_REGION" = true ] && echo "--use-cropped-region") \
            $([ "$SKIP_IF_EXISTS" = true ] && echo "--skip-if-exists") \
            $([ "$SKIP_IF_COMPLETE" = true ] && echo "--skip-if-complete") \
            $([ "$RESUME" = true ] && echo "--resume") \
            > "$OUTPUT_BASE/log_${safe_checkpoint_name}_layer${layer}.txt" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  Layer $layer: Completed successfully"
        else
            echo "  Layer $layer: FAILED (check log file)"
        fi
    done
    
    echo "Completed all layers for ${checkpoint_name}"
}

# Run all ablation checkpoints in parallel (but layers sequentially within each)
pids=()
for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    echo "Starting ablation: $checkpoint_name"
    process_ablation_layers "$checkpoint_name" &
    pids+=($!)
done

echo ""
echo "All processes started (${#pids[@]} total)"
echo "Waiting for completion..."
echo ""

# Wait for all processes
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        ((failed++))
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Completed: $((${#pids[@]} - failed))/${#pids[@]}"
if [ $failed -gt 0 ]; then
    echo "Failed: $failed"
    echo "Check log files in $OUTPUT_BASE for details"
fi

echo ""
echo "Results are in: $OUTPUT_BASE/"
echo ""
echo "Each ablation has:"
echo "  - results_${SPLIT}.json (evaluation results)"
echo "  - visualizations/ (inspection images)"
echo "=========================================="
