#!/bin/bash
#
# Run LLM judge evaluation on all model combinations using logit lens results in parallel
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
BASE_DIR="analysis_results/logit_lens"
OUTPUT_BASE="analysis_results/llm_judge_logitlens"

# Model combinations
LLMS=("olmo-7b" "qwen2-7b" "llama3-8b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

echo "=========================================="
echo "Parallel LLM Judge Evaluation (Logit Lens)"
echo "=========================================="
echo "Images per model: $NUM_IMAGES"
echo "Patches per image: $NUM_SAMPLES"
echo "Use cropped region: $USE_CROPPED_REGION"
echo "Skip if exists: $SKIP_IF_EXISTS"
echo "Skip if complete: $SKIP_IF_COMPLETE"
echo "Resume incomplete: $RESUME"
echo "Random seed: $SEED"
echo "Split: $SPLIT"
echo "Model combinations: ${#LLMS[@]} LLMs × ${#VISION_ENCODERS[@]} encoders = $((${#LLMS[@]} * ${#VISION_ENCODERS[@]})) total"
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
    
    # Check if logit lens directory exists
    logitlens_dir="$BASE_DIR/${checkpoint_name}_step12000-unsharded"
    
    if [ ! -d "$logitlens_dir" ]; then
        echo "WARNING: Logit lens directory not found: $logitlens_dir"
        echo "Skipping ${llm} + ${encoder}"
        return 1
    fi
    
    # Find all available layers for this model
    available_layers=($(ls "$logitlens_dir" | grep "^logit_lens_layer[0-9]*_topk5_multi-gpu.json$" | sed 's/logit_lens_layer//' | sed 's/_topk5_multi-gpu.json//' | sort -n))
    
    if [ ${#available_layers[@]} -eq 0 ]; then
        echo "WARNING: No logit lens files found in: $logitlens_dir"
        echo "Skipping ${llm} + ${encoder}"
        return 1
    fi
    
    echo "Processing ${llm} + ${encoder}: ${#available_layers[@]} layers (${available_layers[*]})"
    
    # Run evaluation for each layer SEQUENTIALLY
    for layer in "${available_layers[@]}"; do
        logitlens_file="$logitlens_dir/logit_lens_layer${layer}_topk5_multi-gpu.json"
        
        if [ ! -f "$logitlens_file" ]; then
            echo "WARNING: Logit lens file not found: $logitlens_file"
            continue
        fi
        
        echo "  Layer $layer: Starting evaluation..."
        
        python3 llm_judge/run_single_model_with_viz_logitlens.py \
            --llm "$llm" \
            --vision-encoder "$encoder" \
            --api-key "$API_KEY" \
            --base-dir "$BASE_DIR" \
            --output-base "$OUTPUT_BASE" \
            --num-images $NUM_IMAGES \
            --num-samples $NUM_SAMPLES \
            --split "$SPLIT" \
            --seed $SEED \
            --layer "layer${layer}" \
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
