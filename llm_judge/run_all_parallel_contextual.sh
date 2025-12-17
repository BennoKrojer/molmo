#!/bin/bash
#
# Run LLM judge evaluation on all model combinations using contextual NN results in parallel
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
DEBUG=false  # Set to true to enable debug logging
BASE_DIR="analysis_results/contextual_nearest_neighbors"
OUTPUT_BASE="analysis_results/llm_judge_contextual_nn"

# Model combinations
LLMS=("olmo-7b" "llama3-8b" "qwen2-7b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

echo "=========================================="
echo "Parallel LLM Judge Evaluation (Contextual NN)"
echo "=========================================="
echo "Images per model: $NUM_IMAGES"
echo "Patches per image: $NUM_SAMPLES"
echo "Use cropped region: $USE_CROPPED_REGION"
echo "Random seed: $SEED"
echo "Split: $SPLIT"
echo "Model combinations: ${#LLMS[@]} LLMs × ${#VISION_ENCODERS[@]} encoders = $((${#LLMS[@]} * ${#VISION_ENCODERS[@]})) total"
echo "Execution: Models in parallel, contextual layers sequentially per model"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to process all contextual layers for a single model combination
process_model_layers() {
    local llm="$1"
    local encoder="$2"
    
    # Special case: qwen2-7b with vit-l-14-336 uses seed10
    if [ "$llm" == "qwen2-7b" ] && [ "$encoder" == "vit-l-14-336" ]; then
        checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}_seed10"
    else
        checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}"
    fi
    
    # Check if contextual neighbors directory exists
    ctxt_dir="$BASE_DIR/${checkpoint_name}_step12000-unsharded"
    
    if [ ! -d "$ctxt_dir" ]; then
        echo "WARNING: Contextual neighbors directory not found: $ctxt_dir"
        echo "Skipping ${llm} + ${encoder}"
        return 1
    fi
    
    # Find all available contextual layers for this model
    # Files can be either:
    #   - New format: contextual_neighbors_visual{v}_allLayers_multi-gpu.json (contains all layers)
    #   - Old format: contextual_neighbors_visual{v}_contextual{c}_multi-gpu.json (one layer per file)
    
    # Check for allLayers format first (try both with and without _multi-gpu suffix)
    allLayers_file=$(ls "$ctxt_dir" | grep "^contextual_neighbors_visual[0-9]*_allLayers.json$" | head -1)
    if [ -z "$allLayers_file" ]; then
        # Fallback to old naming with _multi-gpu suffix
        allLayers_file=$(ls "$ctxt_dir" | grep "^contextual_neighbors_visual[0-9]*_allLayers_multi-gpu.json$" | head -1)
    fi
    
    if [ -n "$allLayers_file" ]; then
        # Extract available layers from the JSON file's contextual_layers_used field
        available_layers=($(python3 -c "import json; f=open('$ctxt_dir/$allLayers_file'); data=json.load(f); layers=set(data.get('contextual_layers_used', [])); layers.add(0); print(' '.join(map(str, sorted(layers))))" 2>/dev/null))
        
        if [ ${#available_layers[@]} -eq 0 ]; then
            echo "WARNING: Could not extract contextual layers from allLayers file: $allLayers_file"
            echo "Skipping ${llm} + ${encoder}"
            return 1
        fi
    else
        # Fallback to old format: extract layer numbers from filenames
        available_layers=($(ls "$ctxt_dir" | grep "^contextual_neighbors_visual[0-9]*_contextual[0-9]*_multi-gpu.json$" | sed 's/.*_contextual\([0-9]*\)_multi-gpu.json/\1/' | sort -n | uniq))
        
        if [ ${#available_layers[@]} -eq 0 ]; then
            echo "WARNING: No contextual neighbor files found in: $ctxt_dir"
            echo "Skipping ${llm} + ${encoder}"
            return 1
        fi
        
        # Ensure layer0 is included
        if [[ ! " ${available_layers[@]} " =~ " 0 " ]]; then
            available_layers=(0 "${available_layers[@]}")
        fi
    fi
    
    echo "Processing ${llm} + ${encoder}: ${#available_layers[@]} contextual layers (${available_layers[*]})"
    
    # Run evaluation for each contextual layer SEQUENTIALLY
    for layer in "${available_layers[@]}"; do
        # Check if output already exists (skip if it does)
        model_name="${llm}_${encoder}"
        if [ "$llm" == "qwen2-7b" ] && [ "$encoder" == "vit-l-14-336" ]; then
            model_name="${llm}_${encoder}_seed10"
        fi
        output_dir_name="llm_judge_${model_name}_contextual${layer}_gpt5"
        if [ "$USE_CROPPED_REGION" = true ]; then
            output_dir_name="${output_dir_name}_cropped"
        fi
        output_json="$OUTPUT_BASE/${output_dir_name}/results_${SPLIT}.json"
        
        if [ -f "$output_json" ]; then
            echo "  Contextual layer $layer: Output already exists, skipping..."
            continue
        fi
        
        echo "  Contextual layer $layer: Starting evaluation..."
        
        log_file="$OUTPUT_BASE/log_${llm}_${encoder}_contextual${layer}.txt"
        
        # Run the script and capture output
        python3 llm_judge/run_single_model_with_viz_contextual.py \
            --llm "$llm" \
            --vision-encoder "$encoder" \
            --api-key_file llm_judge/api_key.txt \
            --base-dir "$BASE_DIR" \
            --output-base "$OUTPUT_BASE" \
            --num-images $NUM_IMAGES \
            --num-samples $NUM_SAMPLES \
            --split "$SPLIT" \
            --seed $SEED \
            --layer "contextual${layer}" \
            $([ "$USE_CROPPED_REGION" = true ] && echo "--use-cropped-region") \
            $([ "$DEBUG" = true ] && echo "--debug") \
            > "$log_file" 2>&1
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "  Contextual layer $layer: Completed successfully"
        else
            echo ""
            echo "============================================================"
            echo "ERROR: Contextual layer $layer FAILED for ${llm} + ${encoder}"
            echo "============================================================"
            echo "Exit code: $exit_code"
            echo "ERROR OUTPUT (full error shown below):"
            echo "------------------------------------------------------------"
            cat "$log_file"
            echo "------------------------------------------------------------"
            echo "Full error log saved to: $log_file"
            echo "============================================================"
            echo ""
            # Don't continue processing this model combination on error - fail loudly
            return 1
        fi
    done
    
    echo "Completed all contextual layers for ${llm} + ${encoder}"
}

# Run all model combinations in parallel (but contextual layers sequentially within each model)
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

# Wait for all processes and collect failed PIDs
failed_pids=()
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    if ! wait $pid; then
        failed_pids+=($pid)
        echo ""
        echo "============================================================"
        echo "ERROR: Process $pid FAILED"
        echo "============================================================"
        echo "Check log files in $OUTPUT_BASE for details"
        echo "============================================================"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
completed=$((${#pids[@]} - ${#failed_pids[@]}))
echo "Completed: $completed/${#pids[@]}"
if [ ${#failed_pids[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "ERROR: ${#failed_pids[@]} process(es) FAILED!"
    echo "============================================================"
    echo "Failed PIDs: ${failed_pids[*]}"
    echo ""
    echo "To see errors, check these log files:"
    for llm in "${LLMS[@]}"; do
        for encoder in "${VISION_ENCODERS[@]}"; do
            for log in "$OUTPUT_BASE/log_${llm}_${encoder}_contextual"*.txt; do
                if [ -f "$log" ] && [ -s "$log" ]; then
                    # Check if log contains error indicators
                    if grep -qi "error\|exception\|traceback\|failed" "$log"; then
                        echo ""
                        echo "ERRORS FOUND in: $log"
                        echo "Last 30 lines:"
                        echo "------------------------------------------------------------"
                        tail -30 "$log"
                        echo "------------------------------------------------------------"
                    fi
                fi
            done
        done
    done
    echo ""
    echo "============================================================"
    # Exit with error code so user knows something failed
    exit 1
fi

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Results are in: $OUTPUT_BASE/"
echo ""
echo "Each model+layer combination has:"
echo "  - results_${SPLIT}.json (evaluation results)"
echo "  - visualizations/ (inspection images)"
echo "=========================================="

