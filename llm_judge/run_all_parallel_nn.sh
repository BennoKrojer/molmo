#!/bin/bash
#
# Run LLM judge evaluation on all model combinations in parallel
#

# Don't use set -e here - we want to handle errors explicitly
set -u  # Fail on undefined variables

# Setup environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Read API key
if [ ! -f "llm_judge/api_key.txt" ]; then
    echo "ERROR: API key file not found: llm_judge/api_key.txt"
    exit 1
fi
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

# Layers to evaluate - includes missing layers (30,31 for OLMo/Llama, 26,27 for Qwen)
# Skip logic will automatically skip already-completed layers
LAYERS=(0 1 2 3 4 8 12 16 20 24 26 27 28 30 31)

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
    local log_prefix="[${llm}+${encoder}]"
    
    # Special case: qwen2-7b with vit-l-14-336 uses seed10
    if [ "$llm" == "qwen2-7b" ] && [ "$encoder" == "vit-l-14-336" ]; then
        checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}_seed10"
    else
        checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}"
    fi
    
    # Check if nearest neighbors directory exists
    nn_dir="$BASE_DIR/${checkpoint_name}_step12000-unsharded"
    
    if [ ! -d "$nn_dir" ]; then
        echo "$log_prefix ERROR: Nearest neighbors directory not found: $nn_dir" >&2
        echo "$log_prefix Skipping ${llm} + ${encoder}" >&2
        return 1
    fi
    
    echo "$log_prefix Processing: ${#LAYERS[@]} layers (${LAYERS[*]})"
    
    # Run evaluation for each layer SEQUENTIALLY
    local layer_failed=0
    for layer in "${LAYERS[@]}"; do
        nn_file="$nn_dir/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer${layer}.json"
        
        if [ ! -f "$nn_file" ]; then
            echo "$log_prefix Layer $layer: ERROR - File not found: $nn_file" >&2
            echo "$log_prefix Layer $layer: Skipping..." >&2
            layer_failed=1
            continue
        fi
        
        echo "$log_prefix Layer $layer: Starting evaluation..."
        local log_file="$OUTPUT_BASE/log_${llm}_${encoder}_layer${layer}.txt"
        
        # Run with tee to show output in real-time AND save to log file
        # Capture exit code from Python script (not tee)
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
            $([ "$RESUME" = true ] && echo "--resume") 2>&1 | tee "$log_file"
        
        local exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            echo "$log_prefix Layer $layer: Completed successfully"
        else
            echo "$log_prefix Layer $layer: FAILED with exit code $exit_code" >&2
            echo "$log_prefix Layer $layer: Last 20 lines of log:" >&2
            tail -n 20 "$log_file" >&2 || true
            layer_failed=1
        fi
    done
    
    if [ $layer_failed -eq 0 ]; then
        echo "$log_prefix Completed all layers for ${llm} + ${encoder}"
        return 0
    else
        echo "$log_prefix Some layers failed for ${llm} + ${encoder}" >&2
        return 1
    fi
}

# Run all model combinations in parallel (but layers sequentially within each model)
pids=()
model_names=()
for llm in "${LLMS[@]}"; do
    for encoder in "${VISION_ENCODERS[@]}"; do
        echo "Starting model combination: $llm + $encoder"
        process_model_layers "$llm" "$encoder" &
        pids+=($!)
        model_names+=("${llm}+${encoder}")
    done
done

echo ""
echo "All processes started (${#pids[@]} total)"
echo "Waiting for completion..."
echo ""

# Wait for all processes with progress reporting
failed=0
failed_models=()
completed=0
total_processes=${#pids[@]}
start_time=$(date +%s)

while [ ${#pids[@]} -gt 0 ]; do
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        model_name=${model_names[$i]}
        
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            # Process finished, wait for it to get exit code
            if wait "$pid" 2>/dev/null; then
                echo "[PROGRESS] $model_name: Completed successfully"
                ((completed++))
            else
                exit_code=$?
                echo "[PROGRESS] $model_name: FAILED with exit code $exit_code" >&2
                failed_models+=("$model_name")
                ((failed++))
            fi
            
            # Remove from arrays
            unset 'pids[$i]'
            unset 'model_names[$i]'
        fi
    done
    
    # Rebuild arrays to remove gaps
    pids=("${pids[@]}")
    model_names=("${model_names[@]}")
    
    # Show progress if still waiting
    if [ ${#pids[@]} -gt 0 ]; then
        elapsed=$(($(date +%s) - start_time))
        echo "[PROGRESS] Still running: ${#pids[@]} processes, Completed: $completed, Failed: $failed, Elapsed: ${elapsed}s"
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Completed: $completed/$total_processes"
if [ $failed -gt 0 ]; then
    echo "Failed: $failed"
    echo "Failed models:"
    for model in "${failed_models[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Check log files in $OUTPUT_BASE for details:"
    for model in "${failed_models[@]}"; do
        llm=$(echo "$model" | cut -d'+' -f1)
        encoder=$(echo "$model" | cut -d'+' -f2)
        for layer in "${LAYERS[@]}"; do
            log_file="$OUTPUT_BASE/log_${llm}_${encoder}_layer${layer}.txt"
            if [ -f "$log_file" ]; then
                echo "  - $log_file"
            fi
        done
    done
fi

echo ""
echo "Results are in: $OUTPUT_BASE/"
echo ""
echo "Each model has:"
echo "  - results_${SPLIT}.json (evaluation results)"
echo "  - visualizations/ (inspection images)"
echo "=========================================="

