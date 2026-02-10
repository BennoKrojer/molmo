#!/bin/bash
#
# Re-run LLM judge for 3 SigLIP models (NN + LogitLens only)
# After fixing hardcoded grid_size=24 in crop_image_region (SigLIP uses 27x27)
#
# Context: The preprocessing fix re-run (2026-02-09) succeeded for DINOv2/CLIP/qwen2-vit
# models, but SigLIP models crashed because crop_image_region() hardcoded grid=24.
# SigLIP uses 27x27 grid, causing patches at row/col >24 to overflow the 512px image.
# Contextual NN was unaffected (doesn't crop patches).
#
# This script:
#   1. Runs unit tests to verify the fix before burning API credits
#   2. Deletes broken SigLIP NN results (all had 9-18/100 samples with wrong crops)
#   3. Deletes broken SigLIP LogitLens results (only layer 0 existed with 9/100)
#   4. Re-runs NN batch (will process SigLIP fresh + resume DINOv2 stragglers)
#   5. Re-runs LogitLens batch (will process SigLIP fresh)
#   6. Re-runs Contextual batch (will resume 2 stragglers: 99/100)
#
# Estimated time: ~6 hours (3 SigLIP models in parallel × 9 layers × 100 images)
#
# Usage:
#   cd /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo
#   bash llm_judge/rerun_siglip_fix.sh

set -euo pipefail

cd "$(dirname "$0")/.."  # cd to repo root

SIGLIP_MODELS=(
    "olmo-7b_siglip"
    "llama3-8b_siglip"
    "qwen2-7b_siglip"
)

echo "==========================================="
echo "SigLIP grid_size Fix: Re-run LLM Judge"
echo "==========================================="
echo ""

# ──────────────────────────────────────────────
# Step 0: Run unit tests BEFORE burning API $
# ──────────────────────────────────────────────
echo "Step 0: Running unit tests to verify fix..."
source ../../env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python3 -m pytest llm_judge/test_preprocessing_pipeline.py -v --tb=short 2>&1 | tail -20
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "FATAL: Unit tests failed! Fix before re-running."
    exit 1
fi
echo ""
echo "  All tests passed."
echo ""

# ──────────────────────────────────────────────
# Step 1: Delete broken SigLIP NN results
# ──────────────────────────────────────────────
echo "Step 1: Deleting broken SigLIP results..."
echo ""

NN_DIR="analysis_results/llm_judge_nearest_neighbors"
LOGIT_DIR="analysis_results/llm_judge_logitlens"

deleted_nn=0
deleted_logit=0

for model in "${SIGLIP_MODELS[@]}"; do
    # Delete NN results (broken: 9-18/100 with wrong crop coordinates)
    for dir in "$NN_DIR"/llm_judge_${model}_layer*_gpt5_cropped; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            deleted_nn=$((deleted_nn + 1))
        fi
    done
    # Delete NN logs
    for log in "$NN_DIR"/log_${model}_layer*.txt; do
        [ -f "$log" ] && rm -f "$log"
    done

    # Delete LogitLens results (broken: only layer 0 with 9/100)
    for dir in "$LOGIT_DIR"/llm_judge_${model}_layer*_gpt5_cropped; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            deleted_logit=$((deleted_logit + 1))
        fi
    done
    # Delete LogitLens logs
    for log in "$LOGIT_DIR"/log_${model}_layer*.txt; do
        [ -f "$log" ] && rm -f "$log"
    done
done

echo "  Deleted $deleted_nn NN directories"
echo "  Deleted $deleted_logit LogitLens directories"
echo ""

# ──────────────────────────────────────────────
# Step 2: Re-run NN (SigLIP fresh + DINOv2 stragglers)
# ──────────────────────────────────────────────
echo "Step 2/4: Re-running Nearest Neighbors..."
echo "  (SigLIP models fresh, DINOv2 stragglers resume)"
echo ""
bash llm_judge/run_all_parallel_nn.sh
echo ""
echo "  NN complete at $(date)"
echo ""

# ──────────────────────────────────────────────
# Step 3: Re-run LogitLens (SigLIP fresh)
# ──────────────────────────────────────────────
echo "Step 3/4: Re-running LogitLens..."
echo "  (SigLIP models fresh)"
echo ""
bash llm_judge/run_all_parallel_logitlens.sh
echo ""
echo "  LogitLens complete at $(date)"
echo ""

# ──────────────────────────────────────────────
# Step 4: Re-run Contextual (resume 2 stragglers)
# ──────────────────────────────────────────────
echo "Step 4/4: Re-running Contextual NN (resume stragglers)..."
echo ""
bash llm_judge/run_all_parallel_contextual.sh
echo ""
echo "  Contextual complete at $(date)"
echo ""

# ──────────────────────────────────────────────
# Step 5: Verify completeness
# ──────────────────────────────────────────────
echo "==========================================="
echo "Verifying completeness..."
echo "==========================================="

python3 -c "
import json, pathlib

models = {
    'olmo-7b_siglip': [0,1,2,4,8,16,24,30,31],
    'olmo-7b_dinov2-large-336': [0,1,2,4,8,16,24,30,31],
    'llama3-8b_siglip': [0,1,2,4,8,16,24,30,31],
    'llama3-8b_dinov2-large-336': [0,1,2,4,8,16,24,30,31],
    'qwen2-7b_siglip': [0,1,2,4,8,16,24,26,27],
    'qwen2-7b_dinov2-large-336': [0,1,2,4,8,16,24,26,27],
    'qwen2-7b_vit-l-14-336_seed10': [0,1,2,4,8,16,24,26,27],
}

methods = {
    'NN': ('analysis_results/llm_judge_nearest_neighbors', 'llm_judge_{model}_layer{layer}_gpt5_cropped'),
    'LogitLens': ('analysis_results/llm_judge_logitlens', 'llm_judge_{model}_layer{layer}_gpt5_cropped'),
    'Contextual': ('analysis_results/llm_judge_contextual_nn', 'llm_judge_{model}_contextual{layer}_gpt5_cropped'),
}

issues = []
ok = 0
for method_name, (base_dir, pattern) in methods.items():
    for model, layers in models.items():
        for layer in layers:
            dir_name = pattern.format(model=model, layer=layer)
            result_dir = pathlib.Path(base_dir) / dir_name.split('/')[-1]
            if not result_dir.exists():
                issues.append(f'  MISSING: {method_name} {model} layer{layer}')
                continue
            found = False
            for f in result_dir.glob('results_*.json'):
                d = json.load(open(f))
                total = d.get('total', len(d.get('results', d.get('responses', []))))
                if total < 95:
                    issues.append(f'  PARTIAL: {method_name} {model} layer{layer} ({total}/100)')
                else:
                    ok += 1
                found = True
                break
            if not found:
                issues.append(f'  NO_JSON: {method_name} {model} layer{layer}')

total_expected = sum(len(layers) for layers in models.values()) * 3
print(f'Complete: {ok}/{total_expected}')
if issues:
    print(f'Issues ({len(issues)}):')
    for i in issues:
        print(i)
else:
    print('All results complete!')
"

echo ""
echo "==========================================="
echo "Re-run complete at $(date)"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Update paper_plots/data.json with new numbers"
echo "  2. Re-generate affected figures"
echo "  3. Review paper text"
