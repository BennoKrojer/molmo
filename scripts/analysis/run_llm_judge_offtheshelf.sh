#!/bin/bash
# Run LLM judge evaluation for off-the-shelf VLMs: Molmo-7B-D and LLaVA-1.5-7B.
#
# Rebuttal item #6 (reviewer BNgn: more off-the-shelf VLMs).
# 3 methods × 2 models × 9 layers × 100 patches = 5,400 evaluations (~$8-12).
#
# Methods: EmbeddingLens (nearest_neighbors), LogitLens (logit_lens), LatentLens (contextual_nearest_neighbors)
# Models:  Molmo-7B-D  (layers 0,1,2,4,8,16,24,26,27)
#          LLaVA-1.5-7B (layers 0,1,2,4,8,16,24,30,31)
#
# Each layer runs as a SEPARATE invocation with fresh seed=42 (matches paper protocol).
# Per-layer outputs are merged into evaluation_results.json per model×method.
#
# Usage:
#   ./scripts/analysis/run_llm_judge_offtheshelf.sh
#
# Output: analysis_results/llm_judge_offtheshelf/{method}/{model_name}/evaluation_results.json

set -e

cd "$(dirname "$0")/../.."

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Load API key
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

EVAL_SCRIPT="latentlens_release/reproduce/scripts/evaluate/evaluate_interpretability.py"
IMAGES_DIR="analysis_results/pixmo_cap_validation_indexed"
OUTPUT_BASE="analysis_results/llm_judge_offtheshelf"
NUM_PATCHES=100
SEED=42

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Indexed images not found at $IMAGES_DIR"
    echo "Run: python scripts/analysis/corpus_ablation/create_indexed_images.py"
    exit 1
fi

# Model definitions: display_name:results_subdir:model_name_for_preproc:layers
MODELS=(
    "molmo-7b:molmo_7b/allenai_Molmo-7B-D-0924:molmo-7b:0 1 2 4 8 16 24 26 27"
    "llava-1.5:llava_1_5/llava-hf_llava-1.5-7b-hf:llava-1.5:0 1 2 4 8 16 24 30 31"
    "qwen2.5-vl-32b:qwen2_5_vl/Qwen_Qwen2.5-VL-32B-Instruct:qwen2vl:0 1 2 4 8 16 32 48 56 62 63"
)

# Method definitions: method_label:results_dir_prefix
METHODS=(
    "embeddinglens:nearest_neighbors"
    "logitlens:logit_lens"
    "latentlens:contextual_nearest_neighbors"
)

# Function: run one model×method combo (each layer as separate invocation with fresh seed=42)
run_one_combo() {
    local model_name=$1
    local model_subdir=$2
    local preproc_name=$3
    local layers=$4
    local method_label=$5
    local results_prefix=$6

    local results_dir="analysis_results/${results_prefix}/${model_subdir}"
    local output_dir="${OUTPUT_BASE}/${method_label}/${model_name}"
    local log_file="${output_dir}/run.log"

    if [ ! -d "$results_dir" ]; then
        echo "[${model_name}/${method_label}] WARN: Results not found: $results_dir — skipping"
        return
    fi

    # Skip if all layers already evaluated
    if [ -f "${output_dir}/evaluation_results.json" ]; then
        local n_layers
        n_layers=$(python3 -c "
import json
with open('${output_dir}/evaluation_results.json') as f:
    data = json.load(f)
print(len([e for e in data if 'interpretable_fraction' in e]))
" 2>/dev/null)
        local expected
        expected=$(echo $layers | wc -w)
        if [ "$n_layers" = "$expected" ]; then
            echo "[${model_name}/${method_label}] SKIP (all $expected layers done)"
            return
        fi
    fi

    mkdir -p "$output_dir"
    echo "[${model_name}/${method_label}] Starting (log: $log_file)"

    # Run each layer SEPARATELY with fresh seed=42 each time.
    # Matches paper protocol where each layer was a separate process.
    for layer in $layers; do
        # Check if this layer is already done
        local layer_done
        layer_done=$(python3 -c "
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

# Launch all 6 model×method combos in parallel (API-bound, no GPU needed)
PIDS=()

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_subdir preproc_name layers <<< "$model_spec"

    for method_spec in "${METHODS[@]}"; do
        IFS=':' read -r method_label results_prefix <<< "$method_spec"

        run_one_combo "$model_name" "$model_subdir" "$preproc_name" "$layers" "$method_label" "$results_prefix" &
        PIDS+=($!)
    done
done

echo ""
echo "Launched ${#PIDS[@]} parallel jobs. Waiting..."
echo ""

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
