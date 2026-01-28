#!/bin/bash
# Run contextual NN for remaining 7 models that need 36 more images
# Extract to _supplement directory, then merge with main results

set -e

# Use absolute paths
PYTHON="/home/nlp/users/bkroje/vl_embedding_spaces/env/bin/python"
export PYTHONPATH="${PYTHONPATH}:/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo"

SCRIPT="scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py"
MERGE_SCRIPT="scripts/analysis/merge_contextual_nn_results.py"
IMAGE_INDICES="100,102,108,109,119,134,136,139,153,154,155,156,167,175,187,191,208,216,218,222,234,240,242,252,253,257,258,260,264,267,284,288,291,293,295,299"
OUTPUT_BASE="analysis_results/contextual_nearest_neighbors"
SUPPLEMENT_BASE="analysis_results/contextual_nearest_neighbors_supplement"
CONTEXTUAL_BASE="molmo_data/contextual_llm_embeddings_vg"

LOG_DIR="logs/remaining_contextual_nn_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p "$SUPPLEMENT_BASE"

echo "========================================"
echo "Contextual NN for 7 remaining models"
echo "========================================"
echo "Logs: $LOG_DIR"
echo ""

# Models that still need 36 more images
# Format: "short_name:encoder:llm_dir:ckpt_suffix:dir_suffix"
declare -a COMBINATIONS=(
    "llama3-8b:dinov2-large-336:meta-llama_Meta-Llama-3-8B::"
    "llama3-8b:siglip:meta-llama_Meta-Llama-3-8B::"
    "llama3-8b:vit-l-14-336:meta-llama_Meta-Llama-3-8B::"
    "olmo-7b:siglip:allenai_OLMo-7B-1024-preview::"
    "qwen2-7b:dinov2-large-336:Qwen_Qwen2-7B::"
    "qwen2-7b:siglip:Qwen_Qwen2-7B::"
    "qwen2-7b:vit-l-14-336:Qwen_Qwen2-7B:_seed10:_seed10"
)

# Function to get visual layers from contextual cache
get_visual_layers() {
    local ctx_dir=$1
    local layers="0"
    for d in "$ctx_dir"/layer_*; do
        if [ -d "$d" ] && [ -f "$d/embeddings_cache.pt" ]; then
            layer=$(basename "$d" | sed 's/layer_//')
            layers="$layers,$layer"
        fi
    done
    echo "$layers"
}

# Launch jobs with staggered model loading
gpu=0
for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r llm encoder llm_dir ckpt_suffix dir_suffix <<< "$combo"

    ckpt="molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_${llm}_${encoder}${ckpt_suffix}/step12000-unsharded"
    contextual_dir="${CONTEXTUAL_BASE}/${llm_dir}"
    dir_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}_step12000-unsharded${dir_suffix}"
    supplement_dir="${SUPPLEMENT_BASE}/${dir_name}"
    log_file="$LOG_DIR/${llm}_${encoder}${dir_suffix}.log"

    # Get available layers from contextual cache
    visual_layers=$(get_visual_layers "$contextual_dir")

    echo "GPU $gpu: ${llm}_${encoder}${dir_suffix}"
    echo "  Checkpoint: $ckpt"
    echo "  Contextual: $contextual_dir"
    echo "  Visual layers: $visual_layers"
    echo "  Supplement dir: $supplement_dir"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" \
        --ckpt-path "$ckpt" \
        --contextual-dir "$contextual_dir" \
        --output-dir "$supplement_dir" \
        --visual-layer "$visual_layers" \
        --image-indices "$IMAGE_INDICES" \
        --top-k 10 \
        --split train \
        > "$log_file" 2>&1 &

    # Wait for model to load before launching next (prevent OOM)
    echo "  Waiting for model to load..."
    for i in {1..60}; do
        sleep 5
        if grep -q "✓ Model loaded" "$log_file" 2>/dev/null; then
            echo "  ✓ Model loaded!"
            break
        fi
        if grep -q "Error\|Traceback" "$log_file" 2>/dev/null; then
            echo "  ✗ Error detected! Check log."
            break
        fi
    done
    echo ""

    gpu=$((gpu + 1))
done

echo "All extraction jobs launched. Waiting for completion..."
wait

echo ""
echo "========================================"
echo "Merging results..."
echo "========================================"

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r llm encoder llm_dir ckpt_suffix dir_suffix <<< "$combo"

    dir_name="train_mlp-only_pixmo_cap_resize_${llm}_${encoder}_step12000-unsharded${dir_suffix}"
    base_dir="${OUTPUT_BASE}/${dir_name}"
    supplement_dir="${SUPPLEMENT_BASE}/${dir_name}"

    if [ -d "$supplement_dir" ] && [ "$(ls -A $supplement_dir/*.json 2>/dev/null)" ]; then
        echo "Merging ${llm}_${encoder}${dir_suffix}..."
        $PYTHON "$MERGE_SCRIPT" --base-dir "$base_dir" --supplement-dir "$supplement_dir"
    else
        echo "Skipping ${llm}_${encoder}${dir_suffix} (no supplement files)"
    fi
done

echo ""
echo "Done!"
