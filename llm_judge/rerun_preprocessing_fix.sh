#!/bin/bash
#
# Re-run LLM judge for 7 models affected by preprocessing bug (2026-02-09).
#
# Bug: process_image_with_mask() used CLIP-style resize+pad for SigLIP/DINOv2 models
# (should be squash-resize), and "qwen2" substring match gave qwen2-7b_vit center-crop
# (should be resize+pad).
#
# This script:
# 1. Backs up old (buggy) results to *_preprocessing_bug_backup/
# 2. Deletes old results for the 7 affected models
# 3. Runs the 3 existing batch scripts (which skip unaffected models automatically)
#
# Affected models:
#   olmo-7b_siglip, olmo-7b_dinov2-large-336
#   llama3-8b_siglip, llama3-8b_dinov2-large-336
#   qwen2-7b_siglip, qwen2-7b_dinov2-large-336
#   qwen2-7b_vit-l-14-336_seed10
#
# Unaffected (CLIP, no qwen2 match — left alone):
#   olmo-7b_vit-l-14-336, llama3-8b_vit-l-14-336
#
# Usage:
#   cd /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo
#   bash llm_judge/rerun_preprocessing_fix.sh
#

set -euo pipefail

cd "$(dirname "$0")/.."  # cd to repo root

# The 7 affected model name patterns (as they appear in directory names)
AFFECTED_MODELS=(
    "olmo-7b_siglip"
    "olmo-7b_dinov2-large-336"
    "llama3-8b_siglip"
    "llama3-8b_dinov2-large-336"
    "qwen2-7b_siglip"
    "qwen2-7b_dinov2-large-336"
    "qwen2-7b_vit-l-14-336_seed10"
)

RESULT_DIRS=(
    "analysis_results/llm_judge_nearest_neighbors"
    "analysis_results/llm_judge_logitlens"
    "analysis_results/llm_judge_contextual_nn"
)

echo "=========================================="
echo "Preprocessing Bug Fix: Re-run LLM Judge"
echo "=========================================="
echo ""
echo "Affected models (7):"
for m in "${AFFECTED_MODELS[@]}"; do
    echo "  - $m"
done
echo ""

# Step 1: Backup and delete old results
echo "Step 1: Backing up and deleting old (buggy) results..."
echo ""

total_backed_up=0
for result_dir in "${RESULT_DIRS[@]}"; do
    if [ ! -d "$result_dir" ]; then
        echo "  WARNING: $result_dir does not exist, skipping"
        continue
    fi

    backup_dir="${result_dir}_preprocessing_bug_backup"
    mkdir -p "$backup_dir"

    for model in "${AFFECTED_MODELS[@]}"; do
        # Match directories like llm_judge_{model}_layer0_gpt5_cropped
        # or llm_judge_{model}_contextual0_gpt5_cropped
        for dir in "$result_dir"/llm_judge_${model}_*; do
            if [ -d "$dir" ]; then
                dirname=$(basename "$dir")
                echo "  Moving: $dirname → $(basename $backup_dir)/"
                mv "$dir" "$backup_dir/$dirname"
                total_backed_up=$((total_backed_up + 1))
            fi
        done
        # Also move log files
        for log in "$result_dir"/log_*; do
            if [ -f "$log" ]; then
                logname=$(basename "$log")
                # Check if this log belongs to an affected model
                for m in "${AFFECTED_MODELS[@]}"; do
                    if echo "$logname" | grep -q "$m"; then
                        mv "$log" "$backup_dir/$logname" 2>/dev/null || true
                        break
                    fi
                done
            fi
        done
    done
done

echo ""
echo "  Backed up $total_backed_up directories total"
echo ""

# Step 2: Re-run the 3 batch scripts
# The existing scripts will skip models that already have results (the 2 unaffected CLIP models)
# and only run the 7 models whose results we just deleted.

echo "Step 2: Re-running LLM judge evaluations..."
echo "  Running 3 methods sequentially (each runs 7 models in parallel internally)"
echo ""

# Ensure PYTHONPATH is set (batch scripts append to it with set -u)
export PYTHONPATH="${PYTHONPATH:-}"

echo "--- Method 1/3: Nearest Neighbors ---"
bash llm_judge/run_all_parallel_nn.sh
echo ""

echo "--- Method 2/3: LogitLens ---"
bash llm_judge/run_all_parallel_logitlens.sh
echo ""

echo "--- Method 3/3: Contextual NN ---"
bash llm_judge/run_all_parallel_contextual.sh
echo ""

echo "=========================================="
echo "Re-run complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Update paper_plots/data.json with new numbers"
echo "  2. Re-generate affected figures"
echo "  3. Review paper text for claims about Qwen2+CLIP outlier"
echo ""
echo "Backup of old (buggy) results:"
for result_dir in "${RESULT_DIRS[@]}"; do
    echo "  ${result_dir}_preprocessing_bug_backup/"
done
