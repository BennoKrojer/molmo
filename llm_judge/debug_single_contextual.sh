#!/bin/bash
#
# Quick debug script to test a single model/layer combination
# Usage: ./debug_single_contextual.sh llama3-8b dinov2-large-336 1
#

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <llm> <vision_encoder> <contextual_layer> [num_images]"
    echo "Example: $0 llama3-8b dinov2-large-336 1 10"
    exit 1
fi

LLM="$1"
ENCODER="$2"
LAYER="$3"
NUM_IMAGES="${4:-10}"  # Default to 10 images for quick testing

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

API_KEY=$(cat llm_judge/api_key.txt)

echo "=========================================="
echo "Quick Debug Test"
echo "=========================================="
echo "LLM: $LLM"
echo "Vision Encoder: $ENCODER"
echo "Contextual Layer: $LAYER"
echo "Images: $NUM_IMAGES"
echo "=========================================="
echo ""

python3 llm_judge/run_single_model_with_viz_contextual.py \
    --llm "$LLM" \
    --vision-encoder "$ENCODER" \
    --api-key_file llm_judge/api_key.txt \
    --base-dir "analysis_results/contextual_nearest_neighbors" \
    --output-base "analysis_results/llm_judge_contextual_nn" \
    --num-images $NUM_IMAGES \
    --num-samples 1 \
    --split "validation" \
    --seed 42 \
    --layer "contextual${LAYER}" \
    --use-cropped-region \
    --debug

