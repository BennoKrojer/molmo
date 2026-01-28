#!/bin/bash
#
# PARALLEL launcher: Run 8 independent single-GPU jobs for Patchscopes
# Each job processes ONE model combination on ONE GPU
#
# Layers: 0, 2, 4, 8, 16 (optimal for patchscopes)
# Patches: 10 random patches per image (matching lite viewer)
#

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

NUM_IMAGES=${1:-10}  # Default to 10 images for lite viewer

echo "========================================"
echo "PARALLEL Patchscopes Analysis"
echo "Images: $NUM_IMAGES"
echo "Layers: 0,2,4,8,16"
echo "Patches: 10 random per image"
echo "Strategy: 8 independent single-GPU jobs"
echo "========================================"
echo ""

# Script path
SCRIPT="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/patchscopes/patchscopes_descriptive.py"
LAYERS="0,2,4,8,16"
NUM_PATCHES=10
OUTPUT_DIR="analysis_results/patchscopes"

# All 9 combinations
declare -a COMBINATIONS=(
    "olmo-7b:vit-l-14-336"
    "olmo-7b:dinov2-large-336"
    "olmo-7b:siglip"
    "llama3-8b:vit-l-14-336"
    "llama3-8b:dinov2-large-336"
    "llama3-8b:siglip"
    "qwen2-7b:vit-l-14-336"
    "qwen2-7b:dinov2-large-336"
    "qwen2-7b:siglip"
)

# Log directory
LOG_DIR="logs/parallel_patchscopes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo ""

# Launch jobs
GPU=0
PIDS=()
COMBO_NAMES=()

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r llm vision <<< "$combo"

    # Build checkpoint path
    if [ "$llm" == "qwen2-7b" ] && [ "$vision" == "vit-l-14-336" ]; then
        ckpt_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision}_seed10"
    else
        ckpt_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision}"
    fi

    ckpt_path="molmo_data/checkpoints/${ckpt_name}/step12000-unsharded"

    # Check path exists
    if [ ! -d "$ckpt_path" ]; then
        echo "SKIP: Checkpoint not found: $ckpt_path"
        continue
    fi

    # Log file
    log_file="$LOG_DIR/${llm}_${vision}.log"

    echo "GPU $GPU: $llm + $vision"
    echo "  Checkpoint: $ckpt_path"
    echo "  Log: $log_file"

    # Launch in background
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python "$SCRIPT" \
        --ckpt-path "$ckpt_path" \
        --num-images $NUM_IMAGES \
        --layers "$LAYERS" \
        --num-patches $NUM_PATCHES \
        --output-dir "$OUTPUT_DIR" \
        --lite-suffix "_lite${NUM_IMAGES}" \
        --skip-html \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    COMBO_NAMES+=("$llm+$vision")

    GPU=$((GPU + 1))

    # STAGGER: Wait for model to load before launching next job (max 5 min timeout)
    echo "  Waiting for model to load (up to 5 min)..."
    for i in {1..60}; do
        sleep 5
        if grep -q "✓ Model loaded" "$log_file" 2>/dev/null; then
            echo "  ✓ Model loaded! Launching next..."
            break
        fi
        if [ $i -eq 60 ]; then
            echo "  Timeout waiting for model load, continuing anyway..."
        fi
    done

    # All 8 GPUs used for 9 combinations (one will wait)
    MAX_PARALLEL=8
    if [ $GPU -ge $MAX_PARALLEL ]; then
        echo ""
        echo "All $MAX_PARALLEL GPUs in use, waiting for a job to complete..."

        # Wait for any one job to finish
        wait -n ${PIDS[@]} 2>/dev/null || true

        # Find which one finished
        NEW_PIDS=()
        NEW_NAMES=()
        for i in "${!PIDS[@]}"; do
            if kill -0 ${PIDS[$i]} 2>/dev/null; then
                NEW_PIDS+=(${PIDS[$i]})
                NEW_NAMES+=("${COMBO_NAMES[$i]}")
            else
                # This job finished, free its GPU
                echo "✓ Completed: ${COMBO_NAMES[$i]}"
                GPU=$((GPU - 1))
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        COMBO_NAMES=("${NEW_NAMES[@]}")
    fi
done

echo ""
echo "========================================"
echo "All jobs launched! Waiting for completion..."
echo "========================================"
echo ""

# Wait for all remaining jobs
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    status=$?
    if [ $status -eq 0 ]; then
        echo "✓ Completed: ${COMBO_NAMES[$i]}"
    else
        echo "✗ FAILED: ${COMBO_NAMES[$i]} (exit code: $status)"
    fi
done

echo ""
echo "========================================"
echo "ALL DONE!"
echo "Check logs in: $LOG_DIR"
echo "Results in: $OUTPUT_DIR"
echo "========================================"
