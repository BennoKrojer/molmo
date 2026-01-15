#!/bin/bash
#
# Run contextual NN extraction for the missing human study images (36 images not in first 100).
# These images are needed to extract cosine similarities for the human annotation dataset.
#
# Usage:
#   ./scripts/analysis/run_contextual_nn_human_study_missing.sh
#

set -e

# Setup
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# The 36 missing image indices (human study images with index > 99)
MISSING_INDICES="100,102,108,109,119,134,136,139,153,154,155,156,167,175,187,191,208,216,218,222,234,240,242,252,253,257,258,260,264,267,284,288,291,293,295,299"

# Visual layer 0 (all human study instances use visual_layer=0)
VISUAL_LAYER="0"

# Output directory suffix to distinguish from main results
OUTPUT_SUFFIX="_human_study_supplement"

# Log directory
LOG_DIR="logs/contextual_nn_human_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Model configurations (9 models)
declare -a MODELS=(
    "olmo-7b_vit-l-14-336"
    "olmo-7b_dinov2-large-336"
    "olmo-7b_siglip"
    "llama3-8b_vit-l-14-336"
    "llama3-8b_dinov2-large-336"
    "llama3-8b_siglip"
    "qwen2-7b_vit-l-14-336_seed10"
    "qwen2-7b_dinov2-large-336"
    "qwen2-7b_siglip"
)

# Checkpoint paths (derive from model name)
get_ckpt_path() {
    local model=$1
    if [[ "$model" == *"seed10"* ]]; then
        echo "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_${model}/step12000-unsharded"
    else
        echo "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_${model}/step12000-unsharded"
    fi
}

# Contextual embeddings dir (derive from LLM)
get_contextual_dir() {
    local model=$1
    local llm="${model%%_*}"  # Extract LLM name (olmo-7b, llama3-8b, qwen2-7b)

    case "$llm" in
        "olmo-7b")
            echo "molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"
            ;;
        "llama3-8b")
            echo "molmo_data/contextual_llm_embeddings_vg/meta-llama_Meta-Llama-3-8B"
            ;;
        "qwen2-7b")
            echo "molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B"
            ;;
        *)
            echo "ERROR: Unknown LLM: $llm" >&2
            exit 1
            ;;
    esac
}

echo "=========================================="
echo "Contextual NN Extraction - Human Study Missing Images"
echo "=========================================="
echo "Missing images: 36 (indices 100-299)"
echo "Visual layer: $VISUAL_LAYER"
echo "Models: ${#MODELS[@]}"
echo "Log dir: $LOG_DIR"
echo "=========================================="
echo ""

# Launch all models in parallel (one per GPU)
pids=()
gpu=0

for model in "${MODELS[@]}"; do
    ckpt_path=$(get_ckpt_path "$model")
    contextual_dir=$(get_contextual_dir "$model")

    # Check paths exist
    if [ ! -d "$ckpt_path" ]; then
        echo "WARNING: Checkpoint not found: $ckpt_path"
        echo "Skipping $model"
        continue
    fi

    if [ ! -d "$contextual_dir" ]; then
        echo "WARNING: Contextual dir not found: $contextual_dir"
        echo "Skipping $model"
        continue
    fi

    log_file="$LOG_DIR/${model}.log"
    output_dir="analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_${model}_step12000-unsharded${OUTPUT_SUFFIX}"

    echo "Launching $model on GPU $gpu..."
    echo "  Checkpoint: $ckpt_path"
    echo "  Contextual: $contextual_dir"
    echo "  Output: $output_dir"
    echo "  Log: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --ckpt-path "$ckpt_path" \
        --contextual-dir "$contextual_dir" \
        --visual-layer "$VISUAL_LAYER" \
        --image-indices "$MISSING_INDICES" \
        --output-dir "$output_dir" \
        --top-k 5 \
        > "$log_file" 2>&1 &

    pids+=($!)
    echo "  PID: ${pids[-1]}"
    echo ""

    # Cycle through GPUs 0-7
    gpu=$(( (gpu + 1) % 8 ))
done

echo "=========================================="
echo "All ${#pids[@]} jobs launched"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Waiting for completion..."

# Wait for all jobs
failed=0
for i in "${!pids[@]}"; do
    if ! wait ${pids[$i]}; then
        echo "ERROR: Job $i (PID ${pids[$i]}) failed!"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=========================================="
if [ $failed -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "ERROR: $failed job(s) failed!"
    echo "Check logs in: $LOG_DIR"
fi
echo "=========================================="
echo ""
echo "Results in: analysis_results/contextual_nearest_neighbors/*${OUTPUT_SUFFIX}/"
