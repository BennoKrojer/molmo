#!/bin/bash
#
# Run LLM judge evaluation on all model combinations in parallel
#

set -e

# Setup environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Read API key
API_KEY=$(cat llm_judge/api_key.txt)

# Configuration
NUM_IMAGES=100
NUM_SAMPLES=1
SPLIT="validation"
USE_CROPPED_REGION=true  # Set to true to use cropped region prompt
SEED=42  # Random seed for reproducibility
SKIP_IF_EXISTS=false  # Set to true to skip ANY existing runs (complete or incomplete)
SKIP_IF_COMPLETE=true  # Set to true to skip only complete runs (will regenerate incomplete)
RESUME=true  # Set to true to resume incomplete runs instead of regenerating from scratch
BASE_DIR="analysis_results/nearest_neighbors"
OUTPUT_BASE="analysis_results/llm_judge_nearest_neighbors"

# Model combinations
LLMS=("olmo-7b" "qwen2-7b" "llama3-8b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

# Layers to evaluate: 1, 2, 3 for fine-grained early alignment, then every 4th layer
LAYERS=(1 2 3 4 8 12 16 20 24 28 32)

echo "=========================================="
echo "Parallel LLM Judge Evaluation"
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
echo "Model combinations: ${#LLMS[@]} LLMs × ${#VISION_ENCODERS[@]} encoders = $((${#LLMS[@]} * ${#VISION_ENCODERS[@]})) total"
echo "Layers per model: ${#LAYERS[@]} (${LAYERS[*]})"
echo "Execution: Models in parallel, layers sequentially per model"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to process all layers for a single model combination
process_model_layers() {
    local llm="$1"
    local encoder="$2"
    
    # Special case: qwen2-7b with vit-l-14-336 uses seed10
    if [ "$llm" == "qwen2-7b" ] && [ "$encoder" == "vit-l-14-336" ]; then
        checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}_seed10"
    else
        checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}"
    fi
    
    # Check if nearest neighbors directory exists
    nn_dir="$BASE_DIR/${checkpoint_name}_step12000-unsharded"
    
    if [ ! -d "$nn_dir" ]; then
        echo "WARNING: Nearest neighbors directory not found: $nn_dir"
        echo "Skipping ${llm} + ${encoder}"
        return 1
    fi
    
    echo "Processing ${llm} + ${encoder}: ${#LAYERS[@]} layers (${LAYERS[*]})"
    
    # Run evaluation for each layer SEQUENTIALLY
    for layer in "${LAYERS[@]}"; do
        nn_file="$nn_dir/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer${layer}.json"
        
        if [ ! -f "$nn_file" ]; then
            echo "  Layer $layer: WARNING - File not found: $nn_file"
            echo "  Layer $layer: Skipping..."
            continue
        fi
        
        echo "  Layer $layer: Starting evaluation..."
        
        python3 llm_judge/run_single_model_with_viz.py \
            --llm "$llm" \
            --vision-encoder "$encoder" \
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
            > "$OUTPUT_BASE/log_${llm}_${encoder}_layer${layer}.txt" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  Layer $layer: Completed successfully"
        else
            echo "  Layer $layer: FAILED (check log file)"
        fi
    done
    
    echo "Completed all layers for ${llm} + ${encoder}"
}

# Run all model combinations in parallel (but layers sequentially within each model)
pids=()
for llm in "${LLMS[@]}"; do
    for encoder in "${VISION_ENCODERS[@]}"; do
        echo "Starting model combination: $llm + $encoder"
        process_model_layers "$llm" "$encoder" &
        pids+=($!)
    done
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
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results are in: $OUTPUT_BASE/"
echo ""
echo "Each model has:"
echo "  - results_${SPLIT}.json (evaluation results)"
echo "  - visualizations/ (inspection images)"
echo "=========================================="

