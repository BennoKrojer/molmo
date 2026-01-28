#!/bin/bash
# Generate all interpretation type visualizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Activate environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=================================="
echo "Generating all visualizations"
echo "=================================="

# Single model visualizations
echo ""
echo "1. Generating single model visualizations..."
echo ""

python scripts/analysis/layer_evolution/visualize_interpretation_types.py \
    --llm olmo-7b --vision-encoder vit-l-14-336

python scripts/analysis/layer_evolution/visualize_interpretation_types.py \
    --llm qwen2-7b --vision-encoder vit-l-14-336

python scripts/analysis/layer_evolution/visualize_interpretation_types.py \
    --llm llama3-8b --vision-encoder vit-l-14-336

# Comparison visualizations
echo ""
echo "2. Generating comparison visualizations..."
echo ""

# Compare LLMs with same encoder
python scripts/analysis/layer_evolution/compare_interpretation_types.py \
    --models "olmo-7b:vit-l-14-336" "qwen2-7b:vit-l-14-336"

python scripts/analysis/layer_evolution/compare_interpretation_types.py \
    --models "olmo-7b:vit-l-14-336" "llama3-8b:vit-l-14-336"

# Compare all three LLMs
python scripts/analysis/layer_evolution/compare_interpretation_types.py \
    --models "olmo-7b:vit-l-14-336" "llama3-8b:vit-l-14-336" "qwen2-7b:vit-l-14-336" \
    --output analysis_results/layer_evolution/compare_all_llms_vit.pdf

echo ""
echo "=================================="
echo "All visualizations generated!"
echo "=================================="
echo ""
echo "Output directory: analysis_results/layer_evolution/"
ls -lh analysis_results/layer_evolution/*.pdf

