#!/bin/bash

# Example usage for Qwen2-VL analysis scripts
# This includes:
# 1. Vision token extraction from Qwen2-VL
# 2. Contextual embeddings extraction from Qwen2-VL's LLM backbone
# 3. Contextual nearest neighbors analysis

# Setup environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ============================================================================
# STEP 1: CONTEXTUAL EMBEDDINGS FROM QWEN2-VL LLM BACKBONE
# ============================================================================
# This extracts text embeddings from Qwen2-VL's LLM (no vision).
# Needed because Qwen2-VL uses a finetuned LLM, not vanilla Qwen2.

# Quick test (100 captions)
# CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py \
#     --num-captions 100 --layers 8 16 24 --batch-size 8

# Full extraction (same layers as vanilla Qwen2)
# CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/create_contextual_embeddings_qwen2vl.py \
#     --num-captions 1000000 \
#     --layers 1 2 4 8 16 24 26 27 \
#     --batch-size 32 \
#     --output-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-VL-7B-Instruct

# After extraction, build caches for fast loading:
# python scripts/analysis/precompute_contextual_caches.py \
#     --contextual-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-VL-7B-Instruct \
#     --num-workers 1

# ============================================================================
# STEP 2: VISION TOKEN EXTRACTION
# ============================================================================

# Extract vision tokens from all layers (smaller test run)
# CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_allLayers_singleGPU.py \
#     --num-images 10 \
#     --layers "0,8,16,24,28" \
#     --output-dir "analysis_results/qwen2_vl/vision_features"

# Extract vision tokens from all layers with full features saved
# CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_allLayers_singleGPU.py \
#     --num-images 100 \
#     --layers "all" \
#     --save-features \
#     --output-dir "analysis_results/qwen2_vl/vision_features"

# ============================================================================
# STEP 3: CONTEXTUAL NEAREST NEIGHBORS
# ============================================================================
# Uses the contextual embeddings from STEP 1 to find nearest neighbors for vision tokens.

# Multi-GPU contextual nearest neighbors (using Qwen2-VL contextual embeddings)
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29528 \
#     scripts/analysis/qwen2_vl/contextual_nearest_neighbors.py \
#     --model-name "Qwen/Qwen2-VL-7B-Instruct" \
#     --contextual-dir "molmo_data/contextual_llm_embeddings/Qwen_Qwen2-VL-7B-Instruct" \
#     --contextual-layer "16" \
#     --visual-layer 0 \
#     --num-images 100 \
#     --top-k 5

# Multiple contextual layers
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29528 \
#     scripts/analysis/qwen2_vl/contextual_nearest_neighbors.py \
#     --model-name "Qwen/Qwen2-VL-7B-Instruct" \
#     --contextual-dir "molmo_data/contextual_llm_embeddings/Qwen_Qwen2-VL-7B-Instruct" \
#     --contextual-layer "8,16,24" \
#     --visual-layer 0 \
#     --num-images 300

# ============================================================================
# SIMPLE EXTRACTION TEST
# ============================================================================

# Test extraction with synthetic image (quick verification)
# CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_all_layers.py --test-only

# Extract from real images with simpler script
# CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/extract_vision_tokens_all_layers.py \
#     --num-images 10 --use-real-images
