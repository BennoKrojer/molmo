#!/bin/bash
#
# Run LLM judge evaluation on human study patches
#

set -u  # Fail on undefined variables

# Get script directory and move to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Setup environment
source ../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Read API key
if [ ! -f "llm_judge/api_key.txt" ]; then
    echo "ERROR: API key file not found: llm_judge/api_key.txt"
    exit 1
fi
API_KEY=$(cat llm_judge/api_key.txt)

# Configuration - NN (token-level) data
DATA_JSON="human_correlations/interp_data_nn/data.json"
OUTPUT_DIR="human_correlations/llm_judge_results"
USE_CROPPED_REGION=true
RESUME=true

echo "=========================================="
echo "LLM Judge Evaluation on Human Study Data"
echo "=========================================="
echo "Data file: $DATA_JSON"
echo "Output directory: $OUTPUT_DIR"
echo "Use cropped region: $USE_CROPPED_REGION"
echo "Resume from existing: $RESUME"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python3 human_correlations/run_llm_judge_on_human_study.py \
    --data-json "$DATA_JSON" \
    --api-key "$API_KEY" \
    --output-dir "$OUTPUT_DIR" \
    $([ "$USE_CROPPED_REGION" = true ] && echo "--use-cropped-region") \
    $([ "$RESUME" = true ] && echo "--resume")

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation Complete!"
    echo "Results saved to: $OUTPUT_DIR/human_study_llm_results.json"
else
    echo "Evaluation FAILED with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

