#!/bin/bash
# Sequential re-run of contextual NN extraction for human study
# Runs models one at a time to avoid OOM issues

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

LOG_DIR="logs/contextual_nn_human_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "SEQUENTIAL CONTEXTUAL NN EXTRACTION"
echo "=========================================="
echo "Output dir: $LOG_DIR"
echo "Models: ${#MODELS[@]}"
echo ""

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r llm encoder contextual_dir layers <<< "$model_spec"

    model_name="${llm}_${encoder}"
    ckpt_path="molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_${model_name}/step12000-unsharded"
    ctx_dir="molmo_data/contextual_llm_embeddings_vg/${contextual_dir}"
    output_dir="analysis_results/contextual_nearest_neighbors/human_study/${model_name}_step12000-unsharded"
    log_file="$LOG_DIR/${model_name}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $model_name"
    echo "  Checkpoint: $ckpt_path"
    echo "  Contextual: $ctx_dir"
    echo "  Layers: $layers"
    echo ""

    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --ckpt-path "$ckpt_path" \
        --contextual-dir "$ctx_dir" \
        --visual-layer "$VISUAL_LAYER" \
        --contextual-layers "$layers" \
        --image-indices "$MISSING_INDICES" \
        --output-dir "$output_dir" \
        --top-k 5 \
        > "$log_file" 2>&1

    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE: $model_name"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAILED: $model_name (exit $status)"
    fi
    echo ""
done

echo "=========================================="
echo "ALL MODELS COMPLETED"
echo "=========================================="
echo "Check logs in: $LOG_DIR"
