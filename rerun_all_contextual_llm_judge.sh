#!/bin/bash
#
# RERUN ALL CONTEXTUAL LLM JUDGE
#
# This script reruns all contextual LLM judge evaluations after fixing the bug
# where visual0_allLayers.json was used for ALL layers instead of visual{N}.
#
# What it runs:
#   1. Main 9 models × 9 layers = 81 runs (via run_all_parallel_contextual.sh)
#   2. Ablations: 9 variants × 9 layers = 81 runs (directly calling run_llm_judge.py)
#   3. Qwen2-VL: 9 layers = 9 runs (directly calling run_llm_judge.py)
#
# Total: 171 LLM judge runs
# Expected runtime: ~12-24 hours (API rate limited)
#
# Usage:
#   ./rerun_all_contextual_llm_judge.sh           # Run everything
#   ./rerun_all_contextual_llm_judge.sh --dry-run # Show what would be run
#

set -e

# Parse args
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
    esac
done

# Setup
cd "$(dirname "$0")"
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Log setup
LOG_DIR="logs/rerun_contextual_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/master.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Configuration - EXPLICIT, NO GLOBS
NUM_IMAGES=100
NUM_SAMPLES=1
SPLIT="validation"
SEED=42

# Layers for OLMo models (32 layers total, sample 9)
OLMO_LAYERS=(0 1 2 4 8 16 24 30 31)

# Layers for Qwen2 models (28 layers total, sample 9)
QWEN2_LAYERS=(0 1 2 4 8 16 24 26 27)

# Ablation checkpoint names - EXPLICIT LIST
ABLATION_CHECKPOINTS=(
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_earlier-vit-layers-6"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_earlier-vit-layers-10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze"
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm"
)

# Derive model_name from checkpoint_name (consistent with run_all_missing.sh)
get_model_name() {
    local ckpt="$1"
    echo "$ckpt" | sed 's/train_mlp-only_pixmo_cap_resize_//' | \
                   sed 's/train_mlp-only_pixmo_cap_//' | \
                   sed 's/train_mlp-only_pixmo_topbottom_/topbottom_/' | \
                   sed 's/train_mlp-only_//'
}

echo "=========================================="
echo "RERUN ALL CONTEXTUAL LLM JUDGE"
echo "=========================================="
echo "Log: $LOG_FILE"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

log "Starting rerun of all contextual LLM judge evaluations"
log "This fixes the bug where visual0_allLayers.json was used for ALL layers"

# ============================================================
# PHASE 1: Main 9 models × 9 layers = 81 runs
# ============================================================
log ""
log "========== PHASE 1: Main Models (9 × 9 = 81 runs) =========="

if [ "$DRY_RUN" = true ]; then
    log "DRY-RUN: Would run llm_judge/run_all_parallel_contextual.sh"
else
    log "Running llm_judge/run_all_parallel_contextual.sh..."
    log "This runs all 9 model combinations in parallel, layers sequentially per model"

    # Run in background and save PID
    bash llm_judge/run_all_parallel_contextual.sh > "$LOG_DIR/main_models.log" 2>&1 &
    MAIN_PID=$!
    log "Main models started with PID $MAIN_PID"
    log "Log: $LOG_DIR/main_models.log"
fi

# ============================================================
# PHASE 2: Ablations = 9 variants × 9 layers = 81 runs
# ============================================================
log ""
log "========== PHASE 2: Ablations (${#ABLATION_CHECKPOINTS[@]} × 9 = $((${#ABLATION_CHECKPOINTS[@]} * 9)) runs) =========="

mkdir -p analysis_results/llm_judge_contextual_nn/ablations

# Function to process all layers for one ablation model
process_ablation() {
    local checkpoint_name="$1"
    local log_file="$2"
    local model_name=$(get_model_name "$checkpoint_name")

    echo "[$(date '+%H:%M:%S')] Starting $model_name" >> "$log_file"

    for layer in "${OLMO_LAYERS[@]}"; do
        result_dir="analysis_results/llm_judge_contextual_nn/ablations/llm_judge_${model_name}_contextual${layer}_gpt5_cropped"

        if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
            echo "[$(date '+%H:%M:%S')] SKIP: $model_name layer $layer (exists)" >> "$log_file"
            continue
        fi

        echo "[$(date '+%H:%M:%S')] Running $model_name layer $layer..." >> "$log_file"

        python3 llm_judge/run_llm_judge.py \
            --analysis-type contextual \
            --llm olmo-7b \
            --vision-encoder vit-l-14-336 \
            --checkpoint-name "$checkpoint_name" \
            --model-name "$model_name" \
            --base-dir analysis_results/contextual_nearest_neighbors/ablations \
            --output-base analysis_results/llm_judge_contextual_nn/ablations \
            --num-images $NUM_IMAGES \
            --num-samples $NUM_SAMPLES \
            --split $SPLIT \
            --seed $SEED \
            --layer "$layer" \
            --use-cropped-region \
            >> "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] ✓ $model_name layer $layer done" >> "$log_file"
        else
            echo "[$(date '+%H:%M:%S')] ✗ $model_name layer $layer FAILED" >> "$log_file"
        fi
    done

    echo "[$(date '+%H:%M:%S')] Completed $model_name (all layers)" >> "$log_file"
}

# Launch all ablation models in parallel
ABLATION_PIDS=()
for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    model_name=$(get_model_name "$checkpoint_name")
    ablation_log="$LOG_DIR/ablation_${model_name}.log"

    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN: Would process $checkpoint_name"
    else
        log "LAUNCHING: $model_name (layers in parallel)"
        process_ablation "$checkpoint_name" "$ablation_log" &
        ABLATION_PIDS+=($!)
    fi
done

if [ "$DRY_RUN" = false ]; then
    log "Started ${#ABLATION_PIDS[@]} ablation processes"
fi

# ============================================================
# PHASE 3: Qwen2-VL = 9 layers
# ============================================================
log ""
log "========== PHASE 3: Qwen2-VL (9 layers) =========="

mkdir -p analysis_results/llm_judge_contextual_nn/qwen2-vl

QWEN2VL_CTX_DIR="analysis_results/contextual_nearest_neighbors/qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"

if [ ! -d "$QWEN2VL_CTX_DIR" ]; then
    log "SKIP: Qwen2-VL contextual NN data not found at $QWEN2VL_CTX_DIR"
else
    QWEN2VL_PIDS=()

    for layer in "${QWEN2_LAYERS[@]}"; do
        result_dir="analysis_results/llm_judge_contextual_nn/qwen2-vl/llm_judge_qwen2vl_contextual${layer}_gpt5_cropped"

        if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
            log "SKIP: Qwen2-VL layer $layer (exists)"
            continue
        fi

        if [ "$DRY_RUN" = true ]; then
            log "DRY-RUN: Would run Qwen2-VL layer $layer"
        else
            log "LAUNCHING: Qwen2-VL layer $layer"
            python3 llm_judge/run_llm_judge.py \
                --analysis-type contextual \
                --llm qwen2-7b \
                --vision-encoder qwen2-vl \
                --checkpoint-name qwen2_vl/Qwen_Qwen2-VL-7B-Instruct \
                --model-name qwen2vl \
                --base-dir analysis_results/contextual_nearest_neighbors \
                --output-base analysis_results/llm_judge_contextual_nn/qwen2-vl \
                --num-images $NUM_IMAGES \
                --num-samples $NUM_SAMPLES \
                --split $SPLIT \
                --seed $SEED \
                --layer "$layer" \
                --use-cropped-region \
                >> "$LOG_DIR/qwen2vl_layer${layer}.log" 2>&1 &
            QWEN2VL_PIDS+=($!)
        fi
    done

    if [ "$DRY_RUN" = false ]; then
        log "Started ${#QWEN2VL_PIDS[@]} Qwen2-VL processes"
    fi
fi

# ============================================================
# Wait for completion
# ============================================================
if [ "$DRY_RUN" = false ]; then
    log ""
    log "========== Waiting for all jobs to complete =========="
    log "Main models PID: ${MAIN_PID:-N/A}"
    log "Ablation PIDs: ${#ABLATION_PIDS[@]} processes"
    log "Qwen2-VL PIDs: ${#QWEN2VL_PIDS[@]} processes"
    log ""
    log "Monitor progress:"
    log "  tail -f $LOG_DIR/main_models.log"
    log "  tail -f $LOG_DIR/ablation_*.log"
    log "  tail -f $LOG_DIR/qwen2vl_*.log"
    log ""
    log "Check output:"
    log "  ls analysis_results/llm_judge_contextual_nn/ | wc -l"
    log "  ls analysis_results/llm_judge_contextual_nn/ablations/ | wc -l"
    log ""

    # Wait for main models
    if [ -n "$MAIN_PID" ]; then
        log "Waiting for main models (PID $MAIN_PID)..."
        if wait $MAIN_PID; then
            log "✓ Main models completed successfully"
        else
            log "✗ Main models FAILED - check $LOG_DIR/main_models.log"
        fi
    fi

    # Wait for ablations
    if [ ${#ABLATION_PIDS[@]} -gt 0 ]; then
        log "Waiting for ${#ABLATION_PIDS[@]} ablation processes..."
        FAILED=0
        for pid in "${ABLATION_PIDS[@]}"; do
            if ! wait $pid; then
                ((FAILED++))
            fi
        done
        if [ $FAILED -eq 0 ]; then
            log "✓ All ablations completed successfully"
        else
            log "✗ $FAILED ablation process(es) FAILED"
        fi
    fi

    # Wait for Qwen2-VL
    if [ ${#QWEN2VL_PIDS[@]} -gt 0 ]; then
        log "Waiting for ${#QWEN2VL_PIDS[@]} Qwen2-VL processes..."
        FAILED=0
        for pid in "${QWEN2VL_PIDS[@]}"; do
            if ! wait $pid; then
                ((FAILED++))
            fi
        done
        if [ $FAILED -eq 0 ]; then
            log "✓ All Qwen2-VL layers completed successfully"
        else
            log "✗ $FAILED Qwen2-VL layer(s) FAILED"
        fi
    fi
fi

# ============================================================
# Summary
# ============================================================
log ""
log "=========================================="
log "RERUN COMPLETE"
log "=========================================="
log "Log directory: $LOG_DIR"
log ""
log "Expected results:"
log "  Main: 81 (9 models × 9 layers)"
log "  Ablations: 81 (9 variants × 9 layers)"
log "  Qwen2-VL: 9 (9 layers)"
log "  Total: 171"
log ""
log "Next steps:"
log "1. Check for failures: grep -i 'failed\|error' $LOG_DIR/*.log"
log "2. Count results:"
log "   ls analysis_results/llm_judge_contextual_nn/*.json 2>/dev/null | wc -l"
log "   ls analysis_results/llm_judge_contextual_nn/ablations/ | wc -l"
log "3. Regenerate plots with updated data"
log "=========================================="

echo ""
echo "=========================================="
echo "RERUN COMPLETE"
echo "=========================================="
echo "Results in: analysis_results/llm_judge_contextual_nn/"
echo "Logs in: $LOG_DIR/"
echo "=========================================="
