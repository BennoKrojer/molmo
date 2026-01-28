#!/bin/bash
# Run concreteness analysis for all model combinations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Activate environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=================================="
echo "Concreteness Analysis"
echo "Running for all model combinations"
echo "=================================="

# Define model combinations
LLMS=("olmo-7b" "llama3-8b" "qwen2-7b")
ENCODERS=("vit-l-14-336" "siglip" "dinov2-large-336")

# Counter for progress
total=$((${#LLMS[@]} * ${#ENCODERS[@]}))
current=0

# Run for each combination
for llm in "${LLMS[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
        current=$((current + 1))
        echo ""
        echo "[$current/$total] Processing: $llm + $encoder"
        echo "----------------------------------------"
        
        python scripts/analysis/layer_evolution/analyze_concreteness.py \
            --llm "$llm" \
            --vision-encoder "$encoder"
    done
done

echo ""
echo "=================================="
echo "All analyses complete!"
echo "=================================="
echo ""
echo "Results saved to: analysis_results/layer_evolution/"
echo ""
echo "Generated visualizations:"
ls -lh analysis_results/layer_evolution/concreteness*.pdf | awk '{print "  " $9 " (" $5 ")"}'

