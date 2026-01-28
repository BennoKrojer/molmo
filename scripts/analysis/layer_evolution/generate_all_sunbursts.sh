#!/bin/bash
# Generate all 30 sunburst data files and plots (10 models × 3 layer variants)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

source ../../env/bin/activate

# All 10 models (9 trained + Qwen2-VL)
MODELS=(
    "olmo-7b_vit-l-14-336"
    "olmo-7b_siglip"
    "olmo-7b_dinov2-large-336"
    "llama3-8b_vit-l-14-336"
    "llama3-8b_siglip"
    "llama3-8b_dinov2-large-336"
    "qwen2-7b_vit-l-14-336_seed10"
    "qwen2-7b_siglip"
    "qwen2-7b_dinov2-large-336"
    "qwen2vl"
)

# Display names for titles
declare -A DISPLAY_NAMES
DISPLAY_NAMES["olmo-7b_vit-l-14-336"]="OLMo-7B ViT-L"
DISPLAY_NAMES["olmo-7b_siglip"]="OLMo-7B SigLIP"
DISPLAY_NAMES["olmo-7b_dinov2-large-336"]="OLMo-7B DINOv2"
DISPLAY_NAMES["llama3-8b_vit-l-14-336"]="Llama3-8B ViT-L"
DISPLAY_NAMES["llama3-8b_siglip"]="Llama3-8B SigLIP"
DISPLAY_NAMES["llama3-8b_dinov2-large-336"]="Llama3-8B DINOv2"
DISPLAY_NAMES["qwen2-7b_vit-l-14-336_seed10"]="Qwen2-7B ViT-L"
DISPLAY_NAMES["qwen2-7b_siglip"]="Qwen2-7B SigLIP"
DISPLAY_NAMES["qwen2-7b_dinov2-large-336"]="Qwen2-7B DINOv2"
DISPLAY_NAMES["qwen2vl"]="Qwen2-VL-7B"

# Layer variants
LAYER_VARIANTS=("all" "early" "late")

declare -A LAYER_SUFFIX
LAYER_SUFFIX["all"]=""
LAYER_SUFFIX["early"]="_early"
LAYER_SUFFIX["late"]="_late"

declare -A LAYER_TITLE
LAYER_TITLE["all"]="All Layers"
LAYER_TITLE["early"]="Early Layers (0,1,2)"
LAYER_TITLE["late"]="Late Layers"

echo "Generating 30 sunburst data files and plots..."
echo ""

count=0
total=30

for model in "${MODELS[@]}"; do
    for layer_var in "${LAYER_VARIANTS[@]}"; do
        count=$((count + 1))
        suffix="_${model}${LAYER_SUFFIX[$layer_var]}"
        title="${DISPLAY_NAMES[$model]} - ${LAYER_TITLE[$layer_var]}"

        echo "[$count/$total] Processing: $model - $layer_var"

        # Generate data file
        echo "  Generating data..."
        python scripts/analysis/layer_evolution/generate_sunburst_data.py \
            --model "$model" \
            --layers "$layer_var" \
            --output-suffix "$suffix"

        # Generate plot
        echo "  Generating plot..."
        python scripts/analysis/layer_evolution/visualize_sunburst_interpretation_types.py \
            --data "sunburst_data${suffix}.pkl" \
            --suffix "$suffix" \
            --title "$title" \
            --no-paper-copy

        echo "  Done: sunburst_interpretation_types${suffix}.pdf"
        echo ""
    done
done

echo "All $total sunbursts generated!"
echo "Output directory: analysis_results/layer_evolution/"
