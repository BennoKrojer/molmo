#!/bin/bash
#
# Build contextual embedding caches (ONE-TIME PREPROCESSING)
#
# This script converts ~7 million tiny .npy files into 24 fast-loading .pt cache files.
# Expected time: 4-8 hours (one time only!)
# After completion: All future analysis runs load caches in seconds instead of hours.
#

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "================================================================================"
echo "Building Contextual Embedding Caches"
echo "================================================================================"
echo ""
echo "This is a ONE-TIME preprocessing step that:"
echo "  • Loads ~7.3 million tiny .npy files (300k per layer × 24 layers)"
echo "  • Combines them into 24 fast-loading .pt cache files"
echo "  • Takes 4-8 hours (disk I/O bound)"
echo ""
echo "After this completes, every future analysis run will:"
echo "  • Load caches in SECONDS instead of HOURS"
echo "  • Save you many hours of repeated loading time"
echo ""
echo "Progress will be shown in real-time with timestamps and loading rates."
echo "================================================================================"
echo ""

read -p "Press Enter to start (or Ctrl+C to cancel)..."

# Run with 4 workers (optimal for disk I/O)
python3 scripts/analysis/precompute_contextual_caches.py \
    --contextual-base molmo_data/contextual_llm_embeddings \
    --num-workers 4

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Cache building complete!"
    echo "================================================================================"
    echo ""
    echo "You can now run your analysis scripts and they will load quickly from cache."
    echo "Example: ./run_all_combinations_contextual_nn.sh"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "✗ Cache building failed!"
    echo "================================================================================"
    echo ""
    echo "Check the error messages above. You may need to:"
    echo "  • Check disk space"
    echo "  • Verify the contextual embeddings directory exists"
    echo "  • Check file permissions"
    echo ""
    exit 1
fi

