#!/bin/bash
# Run LLM judge evaluation with top-1 (pass@1) for 3 models × 3 methods.
#
# This addresses reviewer concern #4 (VLM judge bias: full words vs fragments).
# LatentLens returns full words while LogitLens/EmbeddingLens return subword tokens.
# By evaluating with only the top-1 candidate, we can show the gap is real.
#
# Models (same 3 as corpus ablation):
#   - OLMo+CLIP (olmo-7b_vit-l-14-336)
#   - LLaMA+SigLIP (llama3-8b_siglip)
#   - Qwen2+DINOv2 (qwen2-7b_dinov2-large-336)
#
# Methods: EmbeddingLens, LogitLens, LatentLens
#
# Usage:
#   ./scripts/analysis/corpus_ablation/run_topk1_evaluation.sh
#
# Estimated cost: ~$10-15 (3 models × 3 methods × 9 layers × 100 patches × $0.01/call)

set -e

cd "$(dirname "$0")/../../.."

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration
EVAL_SCRIPT="latentlens_release/reproduce/scripts/evaluate/evaluate_interpretability.py"
IMAGES_DIR="analysis_results/pixmo_cap_validation_indexed"
OUTPUT_BASE="analysis_results/llm_judge_topk1"
NUM_PATCHES=100
TOP_K=1
SEED=42

# Check images exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Indexed images not found at $IMAGES_DIR"
    echo "Run: python scripts/analysis/corpus_ablation/create_indexed_images.py"
    exit 1
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ~/.openai_key ]; then
        source ~/.openai_key
    elif [ -f "llm_judge/api_key.txt" ]; then
        export OPENAI_API_KEY=$(cat llm_judge/api_key.txt)
    else
        echo "ERROR: Set OPENAI_API_KEY or create ~/.openai_key"
        exit 1
    fi
fi

# Model definitions: name:results_suffix:model_name_for_preproc:layers
MODELS=(
    "olmo-vit:olmo-7b_vit-l-14-336:olmo-7b_vit-l-14-336:0 1 2 4 8 16 24 30 31"
    "llama-siglip:llama3-8b_siglip:llama3-8b_siglip:0 1 2 4 8 16 24 30 31"
    "qwen-dino:qwen2-7b_dinov2-large-336:qwen2-7b_dinov2-large-336:0 1 2 4 8 16 24 26 27"
)

# Method definitions: method_label:results_dir_prefix
METHODS=(
    "embeddinglens:nearest_neighbors"
    "logitlens:logit_lens"
    "latentlens:contextual_nearest_neighbors"
)

CKPT_PREFIX="train_mlp-only_pixmo_cap_resize_"
CKPT_SUFFIX="_step12000-unsharded"

# Function: run one model×method combo (all layers sequentially, each with fresh seed)
run_one_combo() {
    local model_name=$1
    local model_suffix=$2
    local preproc_name=$3
    local layers=$4
    local method_label=$5
    local results_prefix=$6

    local results_dir="analysis_results/${results_prefix}/${CKPT_PREFIX}${model_suffix}${CKPT_SUFFIX}"
    local output_dir="${OUTPUT_BASE}/${method_label}/${model_name}"
    local log_file="${output_dir}/run.log"

    if [ ! -d "$results_dir" ]; then
        echo "[${model_name}/${method_label}] WARN: Results not found: $results_dir — skipping"
        return
    fi

    # Skip if all layers already evaluated
    if [ -f "${output_dir}/evaluation_results.json" ]; then
        local n_layers=$(python3 -c "
import json
with open('${output_dir}/evaluation_results.json') as f:
    data = json.load(f)
print(len([e for e in data if 'interpretable_fraction' in e]))
" 2>/dev/null)
        local expected=$(echo $layers | wc -w)
        if [ "$n_layers" = "$expected" ]; then
            echo "[${model_name}/${method_label}] SKIP (all $expected layers done)"
            return
        fi
    fi

    mkdir -p "$output_dir"
    echo "[${model_name}/${method_label}] Starting (log: $log_file)"

    # Run each layer SEPARATELY with fresh seed=42 each time.
    # This matches the paper's protocol where each layer was a separate process.
    for layer in $layers; do
        # Check if this layer is already done
        local layer_done=$(python3 -c "
import json, os
f = '${output_dir}/evaluation_results.json'
if not os.path.exists(f):
    print('no')
else:
    with open(f) as fp:
        data = json.load(fp)
    done = any(e.get('layer') == $layer and 'interpretable_fraction' in e for e in data)
    print('yes' if done else 'no')
" 2>/dev/null)

        if [ "$layer_done" = "yes" ]; then
            echo "[${model_name}/${method_label}] Layer $layer: already done"
            continue
        fi

        echo "[${model_name}/${method_label}] Layer $layer..."
        python "$EVAL_SCRIPT" \
            --results-dir "$results_dir" \
            --images-dir "$IMAGES_DIR" \
            --output-dir "$output_dir/layer_${layer}" \
            --layers $layer \
            --num-patches "$NUM_PATCHES" \
            --model-name "$preproc_name" \
            --top-k "$TOP_K" \
            --seed "$SEED" \
            >> "$log_file" 2>&1
    done

    # Merge per-layer results into single evaluation_results.json
    python3 -c "
import json
from pathlib import Path

eval_dir = Path('$output_dir')
all_results = []
for layer_dir in sorted(eval_dir.iterdir()):
    if not layer_dir.is_dir() or not layer_dir.name.startswith('layer_'):
        continue
    f = layer_dir / 'evaluation_results.json'
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
        for entry in data:
            if 'interpretable_fraction' in entry:
                all_results.append(entry)

all_results.sort(key=lambda e: e['layer'])
with open(eval_dir / 'evaluation_results.json', 'w') as fp:
    json.dump(all_results, fp, indent=2)
print(f'Merged {len(all_results)} layers -> {eval_dir}/evaluation_results.json')
"
    echo "[${model_name}/${method_label}] DONE"
}

# Launch all 9 model×method combos in parallel (API-bound, no GPU needed)
PIDS=()

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_suffix preproc_name layers <<< "$model_spec"

    for method_spec in "${METHODS[@]}"; do
        IFS=':' read -r method_label results_prefix <<< "$method_spec"

        run_one_combo "$model_name" "$model_suffix" "$preproc_name" "$layers" "$method_label" "$results_prefix" &
        PIDS+=($!)
    done
done

echo ""
echo "Launched ${#PIDS[@]} parallel jobs. Waiting..."
echo ""

# Wait for all jobs
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "All done! ${#PIDS[@]} jobs launched, $FAILED failed."
echo "Results in: ${OUTPUT_BASE}/"
echo "=========================================="
