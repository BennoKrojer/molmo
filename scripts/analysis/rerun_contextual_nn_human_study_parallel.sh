#!/bin/bash
# Parallel re-run of contextual NN extraction for human study
# Uses "wait for model loaded" pattern from run_all_missing.sh to avoid OOM

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Models to process
declare -a MODELS=(
    "olmo-7b:vit-l-14-336:allenai_OLMo-7B-1024-preview:1,2,4,8,16,24,30,31"
    "olmo-7b:siglip:allenai_OLMo-7B-1024-preview:1,2,4,8,16,24,30,31"
    "olmo-7b:dinov2-large-336:allenai_OLMo-7B-1024-preview:1,2,4,8,16,24,30,31"
    "llama3-8b:vit-l-14-336:meta-llama_Llama-3.1-8B:1,2,4,8,16,24,30,31"
    "llama3-8b:siglip:meta-llama_Llama-3.1-8B:1,2,4,8,16,24,30,31"
    "llama3-8b:dinov2-large-336:meta-llama_Llama-3.1-8B:1,2,4,8,16,24,30,31"
    "qwen2-7b:vit-l-14-336_seed10:Qwen_Qwen2-7B:1,2,4,8,16,24,26,27"
    "qwen2-7b:siglip:Qwen_Qwen2-7B:1,2,4,8,16,24,26,27"
    "qwen2-7b:dinov2-large-336:Qwen_Qwen2-7B:1,2,4,8,16,24,26,27"
)

MISSING_INDICES="100,102,108,109,119,134,136,139,153,154,155,156,167,175,187,191,208,216,218,222,234,240,242,252,253,257,258,260,264,267,284,288,291,293,295,299"
VISUAL_LAYER="0"
MAX_GPUS=8

LOG_DIR="logs/contextual_nn_human_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "PARALLEL CONTEXTUAL NN EXTRACTION"
echo "=========================================="
echo "Output dir: $LOG_DIR"
echo "Models: ${#MODELS[@]}"
echo "Max GPUs: $MAX_GPUS"
echo ""

PIDS=()
NAMES=()
GPUS=()
gpu=0

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r llm encoder contextual_dir layers <<< "$model_spec"

    model_name="${llm}_${encoder}"
    ckpt_path="molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_${model_name}/step12000-unsharded"
    ctx_dir="molmo_data/contextual_llm_embeddings_vg/${contextual_dir}"
    output_dir="analysis_results/contextual_nearest_neighbors/human_study/${model_name}_step12000-unsharded"
    log_file="$LOG_DIR/${model_name}.log"

    echo "[$(date '+%H:%M:%S')] LAUNCHING: $model_name (GPU $gpu)"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --ckpt-path "$ckpt_path" \
        --contextual-dir "$ctx_dir" \
        --visual-layer "$VISUAL_LAYER" \
        --contextual-layers "$layers" \
        --image-indices "$MISSING_INDICES" \
        --output-dir "$output_dir" \
        --top-k 5 \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    NAMES+=("$model_name")
    GPUS+=($gpu)

    # Wait for model to load before launching next (avoid OOM from concurrent loads)
    echo "  Waiting for model load..."
    for wait_i in {1..60}; do
        sleep 5
        if grep -q "✓ Model loaded" "$log_file" 2>/dev/null; then
            echo "  ✓ Model loaded, continuing..."
            break
        fi
        if grep -q "Error\|Traceback" "$log_file" 2>/dev/null; then
            echo "  ✗ Error detected, continuing..."
            break
        fi
        if [ $wait_i -eq 60 ]; then
            echo "  Timeout waiting, continuing anyway..."
        fi
    done

    gpu=$(( (gpu + 1) % MAX_GPUS ))
done

echo ""
echo "[$(date '+%H:%M:%S')] All models launched, waiting for completion..."

# Wait for all jobs
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE: ${NAMES[$i]} (GPU ${GPUS[$i]})"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAILED: ${NAMES[$i]} (GPU ${GPUS[$i]}, exit $status)"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "COMPLETED: $((${#PIDS[@]} - FAILED))/${#PIDS[@]} succeeded"
echo "=========================================="
echo "Check logs in: $LOG_DIR"
