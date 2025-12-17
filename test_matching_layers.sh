#!/bin/bash

# Quick test script to compare OLD vs NEW approach
# OLD: visual_layer=0 vs contextual_layer=8
# NEW: visual_layer=8 vs contextual_layer=8

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Test parameters
LLM="olmo-7b"
VISION_ENCODER="vit-l-14-336"
LAYER=8
NUM_IMAGES=8  # Need at least NPROC*2 to avoid distribution issues (2 GPUs * 2 = 4 minimum)

# CUDA settings - Use fewer GPUs for small test
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2
MASTER_PORT=29527

# Script path
SCRIPT_PATH="scripts/analysis/contextual_nearest_neighbors.py"

# Setup paths
if [ "$LLM" == "qwen2-7b" ] && [ "$VISION_ENCODER" == "vit-l-14-336" ]; then
    checkpoint_name="train_mlp-only_pixmo_cap_resize_${LLM}_${VISION_ENCODER}_seed10"
else
    checkpoint_name="train_mlp-only_pixmo_cap_resize_${LLM}_${VISION_ENCODER}"
fi

checkpoint_path="molmo_data/checkpoints/${checkpoint_name}/step12000-unsharded"

# Map LLM to contextual directory
if [ "$LLM" == "llama3-8b" ]; then
    contextual_dir="molmo_data/contextual_llm_embeddings_vg/meta-llama_Meta-Llama-3-8B"
elif [ "$LLM" == "olmo-7b" ]; then
    contextual_dir="molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"
elif [ "$LLM" == "qwen2-7b" ]; then
    contextual_dir="molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B"
else
    echo "ERROR: Unknown LLM: $LLM"
    exit 1
fi

output_dir="analysis_results/contextual_nearest_neighbors_test"

echo "="*80
echo "TEST: Comparing OLD vs NEW approach"
echo "="*80
echo "Model: ${LLM} + ${VISION_ENCODER}"
echo "Layer: ${LAYER}"
echo ""

echo "Running NEW approach: visual_layer=${LAYER} vs contextual_layer=${LAYER}"
echo "This compares MATCHING layers (layer ${LAYER} vision vs layer ${LAYER} text)"
echo ""

torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    "$SCRIPT_PATH" \
    --ckpt-path "$checkpoint_path" \
    --contextual-dir "$contextual_dir" \
    --contextual-layer "$LAYER" \
    --visual-layer "$LAYER" \
    --num-images $NUM_IMAGES \
    --output-dir "$output_dir"

if [ $? -eq 0 ]; then
    echo ""
    echo "="*80
    echo "SUCCESS! Now compare results:"
    echo "="*80
    echo ""
    echo "OLD result (layer 0 vision vs layer ${LAYER} text):"
    echo "  analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_${LLM}_${VISION_ENCODER}_step12000-unsharded/contextual_neighbors_visual0_contextual${LAYER}_multi-gpu.json"
    echo ""
    echo "NEW result (layer ${LAYER} vision vs layer ${LAYER} text):"
    echo "  ${output_dir}/train_mlp-only_pixmo_cap_resize_${LLM}_${VISION_ENCODER}_step12000-unsharded/contextual_neighbors_visual${LAYER}_contextual${LAYER}_multi-gpu.json"
    echo ""
    echo "Run the comparison script to see differences:"
    echo "  python3 scripts/analysis/compare_old_vs_new_results.py"
else
    echo "ERROR: Test failed"
    exit 1
fi

