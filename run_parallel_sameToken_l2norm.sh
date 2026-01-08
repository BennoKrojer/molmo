#!/bin/bash
#
# PARALLEL launcher: Run single-GPU jobs for L2 norm analysis
# Each job processes ONE model combination on ONE GPU
#
# Computes L2 norm of vision tokens across layers
#

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================"
echo "PARALLEL L2 Norm Analysis (Vision + Text)"
echo "Strategy: sequential single-GPU jobs"
echo "========================================"
echo ""

# Script paths
VISION_SCRIPT="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/sameToken_acrossLayers_l2norm.py"
TEXT_SCRIPT="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/sameToken_acrossLayers_text_l2norm.py"
NUM_IMAGES=100
TARGET_LAYERS="0,4,8,24"

# Output dirs
VISION_OUTPUT_DIR="analysis_results/sameToken_acrossLayers_l2norm"
TEXT_OUTPUT_DIR="analysis_results/sameToken_acrossLayers_text_l2norm"

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
LOG_DIR="logs/parallel_l2norm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo "Target layers: $TARGET_LAYERS"
echo ""

# Choose mode based on argument
MODE="${1:-both}"

if [ "$MODE" == "vision" ] || [ "$MODE" == "both" ]; then
    echo "========================================"
    echo "PHASE 1: Vision L2 Norm Analysis"
    echo "========================================"
    echo ""

    # Launch vision jobs
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

        # Check if checkpoint exists
        if [ ! -d "$ckpt_path" ]; then
            echo "SKIP: Checkpoint not found: $ckpt_path"
            continue
        fi

        # Log file
        log_file="$LOG_DIR/vision_${llm}_${vision}.log"

        echo "GPU $GPU: $llm + $vision (VISION)"
        echo "  Checkpoint: $ckpt_path"
        echo "  Log: $log_file"

        # Launch in background
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python "$VISION_SCRIPT" \
            --ckpt-path "$ckpt_path" \
            --num-images $NUM_IMAGES \
            --target-layers "$TARGET_LAYERS" \
            --output-dir "$VISION_OUTPUT_DIR" \
            > "$log_file" 2>&1 &

        PIDS+=($!)
        COMBO_NAMES+=("vision:$llm+$vision")

        GPU=$((GPU + 1))

        # STAGGER: Wait for model to load before launching next job (max 5 min timeout)
        echo "  Waiting for model to load (up to 5 min)..."
        for i in {1..60}; do
            sleep 5
            if grep -q "Model loaded on device" "$log_file" 2>/dev/null; then
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
    echo "Waiting for all vision jobs to complete..."

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
    echo "Vision L2 norm analysis complete!"
    echo ""
fi

if [ "$MODE" == "text" ] || [ "$MODE" == "both" ]; then
    echo "========================================"
    echo "PHASE 2: Text L2 Norm Analysis"
    echo "========================================"
    echo ""

    # Launch text jobs
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

        # Check if checkpoint exists
        if [ ! -d "$ckpt_path" ]; then
            echo "SKIP: Checkpoint not found: $ckpt_path"
            continue
        fi

        # Log file
        log_file="$LOG_DIR/text_${llm}_${vision}.log"

        echo "GPU $GPU: $llm + $vision (TEXT)"
        echo "  Checkpoint: $ckpt_path"
        echo "  Log: $log_file"

        # Launch in background
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python "$TEXT_SCRIPT" \
            --ckpt-path "$ckpt_path" \
            --num-images $NUM_IMAGES \
            --target-layers "$TARGET_LAYERS" \
            --output-dir "$TEXT_OUTPUT_DIR" \
            > "$log_file" 2>&1 &

        PIDS+=($!)
        COMBO_NAMES+=("text:$llm+$vision")

        GPU=$((GPU + 1))

        # STAGGER: Wait for model to load before launching next job (max 5 min timeout)
        echo "  Waiting for model to load (up to 5 min)..."
        for i in {1..60}; do
            sleep 5
            if grep -q "Model loaded on device" "$log_file" 2>/dev/null; then
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
    echo "Waiting for all text jobs to complete..."

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
    echo "Text L2 norm analysis complete!"
    echo ""
fi

echo "========================================"
echo "ALL DONE!"
echo "Check logs in: $LOG_DIR"
echo "Vision results in: $VISION_OUTPUT_DIR"
echo "Text results in: $TEXT_OUTPUT_DIR"
echo "========================================"
