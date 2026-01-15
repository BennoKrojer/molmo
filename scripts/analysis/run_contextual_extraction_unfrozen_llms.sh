#!/bin/bash
#
# Launch contextual embedding extraction for the two unfrozen LLM ablation checkpoints.
#
# This script runs extraction in parallel across GPUs for both:
# 1. train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze (captioning task, unfrozen LLM)
# 2. train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm (topbottom task, unfrozen LLM)
#
# Usage:
#   # Run all shards in parallel (recommended: 8 GPUs per checkpoint)
#   ./scripts/analysis/run_contextual_extraction_unfrozen_llms.sh
#
#   # Or run just one checkpoint at a time:
#   ./scripts/analysis/run_contextual_extraction_unfrozen_llms.sh unfreeze
#   ./scripts/analysis/run_contextual_extraction_unfrozen_llms.sh topbottom
#
#   # After all shards complete, merge:
#   ./scripts/analysis/run_contextual_extraction_unfrozen_llms.sh merge
#
#   # Test with small number of captions:
#   ./scripts/analysis/run_contextual_extraction_unfrozen_llms.sh test

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate environment
source ../../env/bin/activate

# Checkpoint paths
CKPT_UNFREEZE="molmo_data/checkpoints/ablations/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze/step12000-unsharded"
CKPT_TOPBOTTOM="molmo_data/checkpoints/ablations/train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm/step12000-unsharded"

# Output directories (will be created automatically)
OUTPUT_UNFREEZE="molmo_data/contextual_llm_embeddings_vg/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze"
OUTPUT_TOPBOTTOM="molmo_data/contextual_llm_embeddings_vg/train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm"

# Layers to extract (OLMo-7B has 32 layers, matching existing contextual embedding configs)
LAYERS="1 2 4 8 16 24 30 31"

# Number of shards (should match number of available GPUs)
NUM_SHARDS=8

# Log directory
LOG_DIR="logs/contextual_extraction_$(date +%Y%m%d_%H%M%S)"

# Python script
SCRIPT="scripts/analysis/create_contextual_embeddings_from_molmo_ckpt.py"

run_checkpoint() {
    local ckpt_path=$1
    local output_dir=$2
    local name=$3

    echo "========================================"
    echo "Starting extraction for: $name"
    echo "Checkpoint: $ckpt_path"
    echo "Output: $output_dir"
    echo "Shards: $NUM_SHARDS"
    echo "Layers: $LAYERS"
    echo "========================================"

    mkdir -p "$LOG_DIR"

    # Launch each shard on a separate GPU
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        local log_file="$LOG_DIR/${name}_shard${shard}.log"

        echo "Launching shard $shard on GPU $shard..."

        CUDA_VISIBLE_DEVICES=$shard python $SCRIPT \
            --ckpt-path "$ckpt_path" \
            --output-dir "$output_dir" \
            --layers $LAYERS \
            --shard $shard \
            --num-shards $NUM_SHARDS \
            --embedding-dtype float8 \
            --batch-size 32 \
            > "$log_file" 2>&1 &

        echo "  PID: $!, log: $log_file"
    done

    echo ""
    echo "All $NUM_SHARDS shards launched for $name"
    echo "Monitor progress with: tail -f $LOG_DIR/${name}_shard*.log"
}

merge_checkpoint() {
    local ckpt_path=$1
    local output_dir=$2
    local name=$3

    echo "========================================"
    echo "Merging shards for: $name"
    echo "========================================"

    python $SCRIPT \
        --ckpt-path "$ckpt_path" \
        --output-dir "$output_dir" \
        --layers $LAYERS \
        --merge-shards \
        --num-shards $NUM_SHARDS

    echo "Merge complete for $name"
}

run_test() {
    local ckpt_path=$1
    local output_dir=$2
    local name=$3

    echo "========================================"
    echo "Test run for: $name"
    echo "========================================"

    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/${name}_test.log"

    CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
        --ckpt-path "$ckpt_path" \
        --output-dir "${output_dir}_test" \
        --layers $LAYERS \
        --num-captions 1000 \
        --embedding-dtype float8 \
        --batch-size 16 \
        2>&1 | tee "$log_file"

    echo "Test complete for $name"
    echo "Output: ${output_dir}_test"
}

# Parse command line arguments
case "${1:-all}" in
    unfreeze)
        run_checkpoint "$CKPT_UNFREEZE" "$OUTPUT_UNFREEZE" "unfreeze"
        ;;
    topbottom)
        run_checkpoint "$CKPT_TOPBOTTOM" "$OUTPUT_TOPBOTTOM" "topbottom"
        ;;
    all)
        # Check available GPUs
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
        if [ "$NUM_GPUS" -lt "$NUM_SHARDS" ]; then
            echo "WARNING: Only $NUM_GPUS GPUs available, need $NUM_SHARDS"
            echo "Running checkpoints sequentially..."
            run_checkpoint "$CKPT_UNFREEZE" "$OUTPUT_UNFREEZE" "unfreeze"
            echo ""
            echo "Waiting for unfreeze to complete before starting topbottom..."
            wait
            run_checkpoint "$CKPT_TOPBOTTOM" "$OUTPUT_TOPBOTTOM" "topbottom"
        else
            # Can run both in parallel if we have enough GPUs (16+)
            echo "Found $NUM_GPUS GPUs"
            if [ "$NUM_GPUS" -ge 16 ]; then
                echo "Running both checkpoints in parallel..."
                run_checkpoint "$CKPT_UNFREEZE" "$OUTPUT_UNFREEZE" "unfreeze"
                run_checkpoint "$CKPT_TOPBOTTOM" "$OUTPUT_TOPBOTTOM" "topbottom"
            else
                echo "Running checkpoints sequentially..."
                run_checkpoint "$CKPT_UNFREEZE" "$OUTPUT_UNFREEZE" "unfreeze"
                echo ""
                echo "Waiting for unfreeze to complete before starting topbottom..."
                wait
                run_checkpoint "$CKPT_TOPBOTTOM" "$OUTPUT_TOPBOTTOM" "topbottom"
            fi
        fi
        ;;
    merge)
        merge_checkpoint "$CKPT_UNFREEZE" "$OUTPUT_UNFREEZE" "unfreeze"
        echo ""
        merge_checkpoint "$CKPT_TOPBOTTOM" "$OUTPUT_TOPBOTTOM" "topbottom"
        ;;
    test)
        run_test "$CKPT_UNFREEZE" "$OUTPUT_UNFREEZE" "unfreeze"
        ;;
    test-topbottom)
        run_test "$CKPT_TOPBOTTOM" "$OUTPUT_TOPBOTTOM" "topbottom"
        ;;
    *)
        echo "Usage: $0 [unfreeze|topbottom|all|merge|test|test-topbottom]"
        echo ""
        echo "Commands:"
        echo "  unfreeze     - Run extraction for unfrozen LLM (captioning task)"
        echo "  topbottom    - Run extraction for unfrozen LLM (topbottom task)"
        echo "  all          - Run extraction for both checkpoints"
        echo "  merge        - Merge shards after extraction completes"
        echo "  test         - Test run with 1000 captions (unfreeze checkpoint)"
        echo "  test-topbottom - Test run with 1000 captions (topbottom checkpoint)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo "Log directory: $LOG_DIR"
echo ""
echo "After all shards complete, run: $0 merge"
