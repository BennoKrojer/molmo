#!/bin/bash
# Run Tuned Lens training + inference + LLM judge for the 3 worst LogitLens models:
#   llama3-8b + siglip      (avg 7.1%)
#   llama3-8b + dinov2      (avg 7.2%)
#   qwen2-7b  + siglip      (avg 10.8%)
#
# Each model runs on its own GPU (0, 1, 2) in parallel.
# After training+inference finish, LLM judge runs sequentially.
#
# Usage:
#   cd /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo
#   bash scripts/analysis/run_all_tunedlens.sh
#
# Monitor:
#   tail -f analysis_results/tunedlens_llama3_siglip.log
#   tail -f analysis_results/tunedlens_llama3_dinov2.log
#   tail -f analysis_results/tunedlens_qwen2_siglip.log

set -e
cd "$(dirname "$0")/../.."

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

API_KEY=$(cat llm_judge/api_key.txt)

CKPT_BASE="molmo_data/checkpoints"
PROBES_BASE="analysis_results/tunedlens_probes"
TUNED_LENS_BASE="analysis_results/tuned_lens"
JUDGE_BASE="analysis_results/llm_judge_tunedlens"

NUM_TRAIN_IMAGES=200
EPOCHS=3
NUM_INFER_IMAGES=100
TOP_K=5

# LLaMA3 layers: 32 transformer blocks → analyze 0,1,2,4,8,16,24,30,31
LLAMA3_LAYERS="0,1,2,4,8,16,24,30,31"
# Qwen2 layers:  28 transformer blocks → analyze 0,1,2,4,8,16,24,26,27
QWEN2_LAYERS="0,1,2,4,8,16,24,26,27"

mkdir -p analysis_results "$PROBES_BASE" "$TUNED_LENS_BASE" "$JUDGE_BASE"

echo "=========================================================="
echo "Tuned Lens: Training + Inference for 3 worst LogitLens models"
echo "=========================================================="
echo "Train images : $NUM_TRAIN_IMAGES × $EPOCHS epochs"
echo "Infer images : $NUM_INFER_IMAGES"
echo ""

# ---- Helper: train + infer for one model on a given GPU ----
run_model() {
    local GPU=$1
    local LLM=$2
    local ENCODER=$3
    local LAYERS=$4
    local LOG=$5

    local CKPT_NAME="train_mlp-only_pixmo_cap_resize_${LLM}_${ENCODER}"
    local CKPT_PATH="${CKPT_BASE}/${CKPT_NAME}/step12000-unsharded"
    local PROBES_PATH="${PROBES_BASE}/${CKPT_NAME}/probes.pt"

    echo "[$(date '+%H:%M:%S')] Starting ${LLM} + ${ENCODER} on GPU ${GPU}" | tee -a "$LOG"

    # --- Train ---
    echo "[$(date '+%H:%M:%S')] Training probes..." | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/analysis/train_tunedlens.py \
        --ckpt-path "$CKPT_PATH" \
        --layers "$LAYERS" \
        --num-train-images $NUM_TRAIN_IMAGES \
        --epochs $EPOCHS \
        --output-dir "$PROBES_BASE" \
        >> "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] Training done." | tee -a "$LOG"

    # --- Infer ---
    echo "[$(date '+%H:%M:%S')] Running inference..." | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/analysis/tunedlens.py \
        --ckpt-path "$CKPT_PATH" \
        --probes-path "$PROBES_PATH" \
        --layers "$LAYERS" \
        --num-images $NUM_INFER_IMAGES \
        --top-k $TOP_K \
        --output-dir "$TUNED_LENS_BASE" \
        >> "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] Inference done." | tee -a "$LOG"
}

# ---- Launch 3 models in parallel on GPUs 0, 1, 2 ----
LOG0="analysis_results/tunedlens_llama3_siglip.log"
LOG1="analysis_results/tunedlens_llama3_dinov2.log"
LOG2="analysis_results/tunedlens_qwen2_siglip.log"

echo "" > "$LOG0"
echo "" > "$LOG1"
echo "" > "$LOG2"

run_model 0 llama3-8b siglip      "$LLAMA3_LAYERS" "$LOG0" &
PID0=$!
run_model 1 llama3-8b dinov2-large-336 "$LLAMA3_LAYERS" "$LOG1" &
PID1=$!
run_model 2 qwen2-7b  siglip      "$QWEN2_LAYERS"  "$LOG2" &
PID2=$!

echo "Launched:"
echo "  GPU 0: llama3-8b + siglip      (PID $PID0) → $LOG0"
echo "  GPU 1: llama3-8b + dinov2      (PID $PID1) → $LOG1"
echo "  GPU 2: qwen2-7b  + siglip      (PID $PID2) → $LOG2"
echo ""
echo "Monitor with: tail -f $LOG0"
echo ""

# Wait for all train+infer jobs
wait $PID0 && echo "✓ llama3+siglip done" || echo "✗ llama3+siglip FAILED"
wait $PID1 && echo "✓ llama3+dinov2 done" || echo "✗ llama3+dinov2 FAILED"
wait $PID2 && echo "✓ qwen2+siglip done"  || echo "✗ qwen2+siglip FAILED"

echo ""
echo "=========================================================="
echo "LLM JUDGE EVALUATION"
echo "=========================================================="

# Run LLM judge for each model sequentially (reuses run_single_model_with_viz_logitlens.py)
run_judge() {
    local LLM=$1
    local ENCODER=$2

    local CKPT_NAME="train_mlp-only_pixmo_cap_resize_${LLM}_${ENCODER}"
    local TUNED_DIR="${TUNED_LENS_BASE}/${CKPT_NAME}_step12000-unsharded"
    local AVAILABLE_LAYERS
    AVAILABLE_LAYERS=($(ls "$TUNED_DIR" | grep "^logit_lens_layer[0-9]*_topk${TOP_K}_multi-gpu.json$" \
        | sed "s/logit_lens_layer//" | sed "s/_topk${TOP_K}_multi-gpu.json//" | sort -n))

    if [ ${#AVAILABLE_LAYERS[@]} -eq 0 ]; then
        echo "WARNING: no tuned lens files found in $TUNED_DIR — skipping judge for ${LLM}+${ENCODER}"
        return 1
    fi

    echo "Running LLM judge for ${LLM} + ${ENCODER}  (${#AVAILABLE_LAYERS[@]} layers)..."
    for LAYER in "${AVAILABLE_LAYERS[@]}"; do
        python llm_judge/run_single_model_with_viz_logitlens.py \
            --llm "$LLM" \
            --vision-encoder "$ENCODER" \
            --api-key "$API_KEY" \
            --base-dir "$TUNED_LENS_BASE" \
            --output-base "$JUDGE_BASE" \
            --num-images 100 \
            --layer "layer${LAYER}" \
            --seed 42
    done
    echo "  ✓ Judge done for ${LLM} + ${ENCODER}"
}

run_judge llama3-8b siglip
run_judge llama3-8b dinov2-large-336
run_judge qwen2-7b  siglip

echo ""
echo "=========================================================="
echo "ALL DONE"
echo "=========================================================="
echo "Results:"
echo "  Probes:     $PROBES_BASE/"
echo "  TunedLens:  $TUNED_LENS_BASE/"
echo "  LLM judge:  $JUDGE_BASE/"
echo ""
echo "Compare with LogitLens:"
echo "  LogitLens judge: analysis_results/llm_judge_logitlens/"
