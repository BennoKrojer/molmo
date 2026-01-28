#!/bin/bash
#
# Run contextual NN for human study images (indices 100-299)
# Adapted from run_parallel_contextual_nn.sh with --image-indices support
#

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================"
echo "Contextual NN for Human Study Images"
echo "Strategy: 8 independent single-GPU jobs"
echo "========================================"
echo ""

SCRIPT="scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py"
OUTPUT_DIR="analysis_results/contextual_nearest_neighbors_vg"
CONTEXTUAL_BASE="molmo_data/contextual_llm_embeddings_vg"

# Human study missing image indices (36 images from 100-299)
IMAGE_INDICES="100,102,108,109,119,134,136,139,153,154,155,156,167,175,187,191,208,216,218,222,234,240,242,252,253,257,258,260,264,267,284,288,291,293,295,299"

# All 9 combinations (same as run_parallel_contextual_nn.sh)
declare -a COMBINATIONS=(
    "olmo-7b:vit-l-14-336:allenai_OLMo-7B-1024-preview"
    "olmo-7b:dinov2-large-336:allenai_OLMo-7B-1024-preview"
    "olmo-7b:siglip:allenai_OLMo-7B-1024-preview"
    "llama3-8b:vit-l-14-336:meta-llama_Meta-Llama-3-8B"
    "llama3-8b:dinov2-large-336:meta-llama_Meta-Llama-3-8B"
    "llama3-8b:siglip:meta-llama_Meta-Llama-3-8B"
    "qwen2-7b:vit-l-14-336:Qwen_Qwen2-7B"
    "qwen2-7b:dinov2-large-336:Qwen_Qwen2-7B"
    "qwen2-7b:siglip:Qwen_Qwen2-7B"
)

# Function to get visual layers (includes layer 0 for vision backbone features)
get_visual_layers() {
    local ctx_dir=$1
    local layers="0"  # Always include layer 0 (vision backbone)
    for d in "$ctx_dir"/layer_*; do
        if [ -d "$d" ] && [ -f "$d/embeddings_cache.pt" ]; then
            layer=$(basename "$d" | sed 's/layer_//')
            layers="$layers,$layer"
        fi
    done
    echo "$layers"
}

# Log directory
LOG_DIR="logs/contextual_nn_human_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo ""

# Launch jobs
GPU=0
PIDS=()
COMBO_NAMES=()

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r llm vision ctx_subdir <<< "$combo"

    # Build paths (same logic as run_parallel_contextual_nn.sh)
    if [ "$llm" == "qwen2-7b" ] && [ "$vision" == "vit-l-14-336" ]; then
        ckpt_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision}_seed10"
    else
        ckpt_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision}"
    fi

    ckpt_path="molmo_data/checkpoints/${ckpt_name}/step12000-unsharded"
    ctx_dir="${CONTEXTUAL_BASE}/${ctx_subdir}"

    # Check paths exist
    if [ ! -d "$ckpt_path" ]; then
        echo "SKIP: Checkpoint not found: $ckpt_path"
        continue
    fi
    if [ ! -d "$ctx_dir" ]; then
        echo "SKIP: Contextual dir not found: $ctx_dir"
        continue
    fi

    # Get available layers
    layers=$(get_visual_layers "$ctx_dir")
    if [ -z "$layers" ]; then
        echo "SKIP: No layer caches in $ctx_dir"
        continue
    fi

    # Log file
    log_file="$LOG_DIR/${llm}_${vision}.log"

    echo "GPU $GPU: $llm + $vision"
    echo "  Checkpoint: $ckpt_path"
    echo "  Layers: $layers"
    echo "  Log: $log_file"

    # Launch in background with --image-indices
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python "$SCRIPT" \
        --ckpt-path "$ckpt_path" \
        --contextual-dir "$ctx_dir" \
        --visual-layer "$layers" \
        --image-indices "$IMAGE_INDICES" \
        --output-dir "$OUTPUT_DIR" \
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
        if grep -qE "Error|Traceback" "$log_file" 2>/dev/null; then
            echo "  ✗ Error detected, continuing..."
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
echo "========================================"
