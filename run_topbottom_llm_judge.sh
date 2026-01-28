#!/bin/bash
#
# Run LLM judge on both topbottom ablations (frozen and unfrozen)
# Following exact pattern from rerun_all_contextual_llm_judge.sh
#

set -e

cd "$(dirname "$0")"
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration - matching existing norms
NUM_IMAGES=100
NUM_SAMPLES=1
SPLIT="validation"
SEED=42
OLMO_LAYERS=(0 1 2 4 8 16 24 30 31)

LOG_DIR="logs/topbottom_llm_judge_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "LLM Judge - Topbottom Ablations"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo "Layers: ${OLMO_LAYERS[*]}"
echo "Images: $NUM_IMAGES"
echo ""

# Function to run all layers for one ablation (sequentially)
run_ablation() {
    local checkpoint_name="$1"
    local model_name="$2"
    local log_file="$3"

    echo "[$(date '+%H:%M:%S')] Starting $model_name" | tee -a "$log_file"

    for layer in "${OLMO_LAYERS[@]}"; do
        echo "[$(date '+%H:%M:%S')] Running $model_name layer $layer..." | tee -a "$log_file"

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
            echo "[$(date '+%H:%M:%S')] ✓ $model_name layer $layer done" | tee -a "$log_file"
        else
            echo "[$(date '+%H:%M:%S')] ✗ $model_name layer $layer FAILED" | tee -a "$log_file"
        fi
    done

    echo "[$(date '+%H:%M:%S')] Completed $model_name (all layers)" | tee -a "$log_file"
}

# Run both in parallel (each model runs its layers sequentially)
run_ablation \
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336" \
    "topbottom_olmo-7b_vit-l-14-336" \
    "$LOG_DIR/frozen.log" &
FROZEN_PID=$!

run_ablation \
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm" \
    "topbottom_olmo-7b_vit-l-14-336_unfreeze-llm" \
    "$LOG_DIR/unfrozen.log" &
UNFROZEN_PID=$!

echo "Started:"
echo "  Frozen: PID $FROZEN_PID (log: $LOG_DIR/frozen.log)"
echo "  Unfrozen: PID $UNFROZEN_PID (log: $LOG_DIR/unfrozen.log)"
echo ""
echo "Waiting for completion..."

wait $FROZEN_PID
FROZEN_EXIT=$?
echo "Frozen completed with exit code: $FROZEN_EXIT"

wait $UNFROZEN_PID
UNFROZEN_EXIT=$?
echo "Unfrozen completed with exit code: $UNFROZEN_EXIT"

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "Results:"
ls -la analysis_results/llm_judge_contextual_nn/ablations/ | grep topbottom
echo ""
echo "Total topbottom directories: $(ls analysis_results/llm_judge_contextual_nn/ablations/ | grep topbottom | wc -l)"
