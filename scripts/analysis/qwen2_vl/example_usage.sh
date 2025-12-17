#!/bin/bash

# Example usage for Qwen2-VL contextual nearest neighbors analysis

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set CUDA devices and distributed settings
# For 8 GPUs:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
MASTER_PORT=29528

# For 2 GPUs (alternative):
# export CUDA_VISIBLE_DEVICES=4,5
# NPROC=2
# MASTER_PORT=29528

# Script path
SCRIPT_PATH="scripts/analysis/qwen2_vl/contextual_nearest_neighbors.py"

# Example 1: Single contextual layer, vision encoder output (visual_layer=0)
# Using 8 GPUs - each GPU processes num_images/8 images
torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    --model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --contextual-dir "molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B" \
    --contextual-layer "16" \
    --visual-layer 0 \
    --num-images 100 \
    --top-k 5 \
    --output-dir "analysis_results/qwen2_vl/contextual_nearest_neighbors"

# Example 2: Multiple contextual layers, LLM layer 8 output
torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    --model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --contextual-dir "molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B" \
    --contextual-layer "8,16,24" \
    --visual-layer 8 \
    --num-images 300 \
    --top-k 5 \
    --output-dir "analysis_results/qwen2_vl/contextual_nearest_neighbors"

# Example 3: Using Visual Genome contextual embeddings (if available)
# torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
#     $SCRIPT_PATH \
#     --model-name "Qwen/Qwen2-VL-7B-Instruct" \
#     --contextual-dir "molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-7B" \
#     --contextual-layer "8,16,24" \
#     --visual-layer 0 \
#     --num-images 100 \
#     --output-dir "analysis_results/qwen2_vl/contextual_nearest_neighbors_vg"

