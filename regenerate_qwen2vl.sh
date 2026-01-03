#!/bin/bash
# =============================================================================
# Regenerate ALL Qwen2-VL Analysis Data
# =============================================================================
# 
# ROOT CAUSE: All Qwen2-VL data was generated without --force-square, producing
# variable grid sizes (231-266 tokens) instead of consistent 16x16 = 256 tokens.
#
# This script regenerates:
#   1. Static NN (nearest_neighbors.py) - 300 images, 9 layers
#   2. LogitLens (logitlens.py) - 300 images, 9 layers  
#   3. Contextual NN (contextual_nearest_neighbors_allLayers_singleGPU.py) - 100 images
#   4. LLM Judge for Contextual NN (since data changed)
#
# Estimated time: ~3-4 hours total
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate environment
source ../../env/bin/activate 2>/dev/null || source env/bin/activate 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"

echo "=============================================="
echo "QWEN2-VL DATA REGENERATION"
echo "=============================================="
echo "This will regenerate ALL Qwen2-VL analysis data"
echo "with --force-square for consistent 16x16 grids."
echo ""
echo "WARNING: This will OVERWRITE existing data!"
echo ""

# Parse arguments
DRY_RUN=false
SKIP_LLM_JUDGE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-llm-judge)
            SKIP_LLM_JUDGE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./regenerate_qwen2vl.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Show what would be run without executing"
            echo "  --skip-llm-judge  Skip LLM Judge regeneration (saves API costs)"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE - commands will be shown but not executed]"
    echo ""
fi

# Common settings
NUM_IMAGES=300
NUM_IMAGES_CONTEXTUAL=100  # Contextual uses fewer images
LAYERS="0,1,2,4,8,16,24,26,27"
TOP_K=5

# Output directories
NN_OUTPUT="analysis_results/nearest_neighbors/qwen2_vl"
LOGITLENS_OUTPUT="analysis_results/logit_lens/qwen2_vl"
CONTEXTUAL_OUTPUT="analysis_results/contextual_nearest_neighbors/ablations"

# =============================================================================
# Step 1: Static NN
# =============================================================================
echo "[1/4] Regenerating Static NN..."
echo "  Output: $NN_OUTPUT"
echo "  Images: $NUM_IMAGES, Layers: $LAYERS"

CMD_NN="CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/nearest_neighbors.py \
    --num-images $NUM_IMAGES \
    --layers $LAYERS \
    --top-k $TOP_K \
    --output-dir $NN_OUTPUT \
    --force-square"

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would run:"
    echo "  $CMD_NN"
else
    echo "  Running..."
    eval $CMD_NN
    echo "  ✓ Done"
fi
echo ""

# =============================================================================
# Step 2: LogitLens
# =============================================================================
echo "[2/4] Regenerating LogitLens..."
echo "  Output: $LOGITLENS_OUTPUT"
echo "  Images: $NUM_IMAGES, Layers: $LAYERS"

CMD_LOGITLENS="CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/logitlens.py \
    --num-images $NUM_IMAGES \
    --layers $LAYERS \
    --top-k $TOP_K \
    --output-dir $LOGITLENS_OUTPUT \
    --force-square"

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would run:"
    echo "  $CMD_LOGITLENS"
else
    echo "  Running..."
    eval $CMD_LOGITLENS
    echo "  ✓ Done"
fi
echo ""

# =============================================================================
# Step 3: Contextual NN (LN-Lens)
# =============================================================================
echo "[3/4] Regenerating Contextual NN (LN-Lens)..."
echo "  Output: $CONTEXTUAL_OUTPUT/Qwen_Qwen2-VL-7B-Instruct"
echo "  Images: $NUM_IMAGES_CONTEXTUAL"

# Note: This script already has --force-square default=True now
CMD_CONTEXTUAL="CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py \
    --num-images $NUM_IMAGES_CONTEXTUAL \
    --output-dir $CONTEXTUAL_OUTPUT \
    --force-square"

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would run:"
    echo "  $CMD_CONTEXTUAL"
else
    echo "  Running..."
    eval $CMD_CONTEXTUAL
    echo "  ✓ Done"
fi
echo ""

# =============================================================================
# Step 4: LLM Judge for Contextual NN
# =============================================================================
if [ "$SKIP_LLM_JUDGE" = true ]; then
    echo "[4/4] Skipping LLM Judge (--skip-llm-judge flag set)"
else
    echo "[4/4] Regenerating LLM Judge for Contextual NN..."
    echo "  WARNING: This uses OpenAI API and incurs costs!"
    
    # LLM Judge needs to be re-run for each contextual layer
    CONTEXTUAL_LAYERS=(0 1 2 4 8 16 24 26 27)
    
    for layer in "${CONTEXTUAL_LAYERS[@]}"; do
        echo "  Layer $layer..."
        CMD_JUDGE="python llm_judge/run_single_model_with_viz_contextual.py \
            --model qwen2-vl \
            --layer $layer \
            --num-images 50"
        
        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY RUN] Would run: $CMD_JUDGE"
        else
            eval $CMD_JUDGE || echo "    Warning: Layer $layer failed"
        fi
    done
    echo "  ✓ Done"
fi
echo ""

# =============================================================================
# Verification
# =============================================================================
echo "=============================================="
echo "VERIFICATION"
echo "=============================================="

echo "Checking token counts in regenerated data..."

python3 -c "
import json
import sys

def check_file(path, name):
    try:
        with open(path) as f:
            data = json.load(f)
        results = data.get('results', [])
        if not results:
            print(f'  {name}: No results found')
            return False
        
        tokens = [r.get('num_vision_tokens', 0) for r in results[:10]]
        all_256 = all(t == 256 for t in tokens)
        status = '✓' if all_256 else '✗'
        print(f'  {name}: {status} First 10 token counts: {tokens}')
        return all_256
    except FileNotFoundError:
        print(f'  {name}: File not found')
        return False

ok = True
ok &= check_file('$NN_OUTPUT/Qwen_Qwen2-VL-7B-Instruct/nearest_neighbors_layer0_topk5.json', 'NN')
ok &= check_file('$LOGITLENS_OUTPUT/Qwen_Qwen2-VL-7B-Instruct/logit_lens_layer0_topk5.json', 'LogitLens')
ok &= check_file('$CONTEXTUAL_OUTPUT/Qwen_Qwen2-VL-7B-Instruct/contextual_neighbors_visual0_allLayers.json', 'Contextual NN')

if ok:
    print()
    print('✓ All data regenerated with consistent 16x16 grids!')
else:
    print()
    print('✗ Some data still has inconsistent grids - check the output above')
    sys.exit(1)
"

echo ""
echo "=============================================="
echo "NEXT STEPS"
echo "=============================================="
echo "1. Regenerate demo viewer: ./generate_demo.sh --num-images 10"
echo "2. Verify Qwen2-VL viewer has complete 16x16 grid"
echo "3. Commit and push: git add -A && git commit -m 'Regenerate Qwen2-VL data with --force-square'"
echo ""

