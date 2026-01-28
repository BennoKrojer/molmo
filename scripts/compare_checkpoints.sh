#!/bin/bash
# Comprehensive checkpoint comparison script
# Generates captions with both original and stripped checkpoints and compares outputs

set -e

echo "=========================================="
echo "CHECKPOINT COMPARISON TEST"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Generate captions using the ORIGINAL checkpoint"
echo "  2. Generate captions using the STRIPPED checkpoint"
echo "  3. Compare outputs to verify they are identical"
echo ""

# Configuration
ORIGINAL_CKPT="/mnt/research/scratch/bkroje/molmo_data/checkpoints/TEST_STRIP/step12000-test.backup"
STRIPPED_CKPT="/mnt/research/scratch/bkroje/molmo_data/checkpoints/TEST_STRIP/step12000-test"
MAX_TOKENS=30
GPUS="0,1"
NUM_PROCS=2

echo "Configuration:"
echo "  Original checkpoint: $ORIGINAL_CKPT"
echo "  Stripped checkpoint: $STRIPPED_CKPT"
echo "  GPUs: $GPUS"
echo "  Processes: $NUM_PROCS"
echo "  Max tokens: $MAX_TOKENS"
echo ""

# Setup environment
cd /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create temp files
ORIGINAL_OUT=$(mktemp)
STRIPPED_OUT=$(mktemp)

echo "=========================================="
echo "STEP 1: Testing ORIGINAL checkpoint"
echo "=========================================="
echo ""

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_PROCS scripts/minimal_val_captions.py \
    --ckpt-path "$ORIGINAL_CKPT" \
    --max-tokens $MAX_TOKENS 2>&1 | tee /tmp/original_full.log
END_TIME=$(date +%s)
ORIGINAL_TIME=$((END_TIME - START_TIME))

# Extract image outputs
grep "^\[Image" /tmp/original_full.log | sort > "$ORIGINAL_OUT"

ORIGINAL_COUNT=$(wc -l < "$ORIGINAL_OUT")
echo ""
echo "✓ Original checkpoint completed in ${ORIGINAL_TIME}s"
echo "✓ Generated $ORIGINAL_COUNT captions"
echo ""
echo "Sample outputs from ORIGINAL checkpoint:"
head -n 3 "$ORIGINAL_OUT" | sed 's/^/  /'
echo "  ..."
echo ""

echo "=========================================="
echo "STEP 2: Testing STRIPPED checkpoint"
echo "=========================================="
echo ""

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_PROCS scripts/minimal_val_captions.py \
    --ckpt-path "$STRIPPED_CKPT" \
    --max-tokens $MAX_TOKENS 2>&1 | tee /tmp/stripped_full.log
END_TIME=$(date +%s)
STRIPPED_TIME=$((END_TIME - START_TIME))

# Extract image outputs
grep "^\[Image" /tmp/stripped_full.log | sort > "$STRIPPED_OUT"

STRIPPED_COUNT=$(wc -l < "$STRIPPED_OUT")
echo ""
echo "✓ Stripped checkpoint completed in ${STRIPPED_TIME}s"
echo "✓ Generated $STRIPPED_COUNT captions"
echo ""
echo "Sample outputs from STRIPPED checkpoint:"
head -n 3 "$STRIPPED_OUT" | sed 's/^/  /'
echo "  ..."
echo ""

echo "=========================================="
echo "STEP 3: COMPARING OUTPUTS"
echo "=========================================="
echo ""

if [ "$ORIGINAL_COUNT" -ne "$STRIPPED_COUNT" ]; then
    echo "❌ ERROR: Different number of outputs!"
    echo "   Original: $ORIGINAL_COUNT captions"
    echo "   Stripped: $STRIPPED_COUNT captions"
    exit 1
fi

echo "Comparing $ORIGINAL_COUNT captions line-by-line..."
echo ""

if diff -q "$ORIGINAL_OUT" "$STRIPPED_OUT" > /dev/null 2>&1; then
    echo "=========================================="
    echo "✅ SUCCESS: OUTPUTS ARE IDENTICAL!"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  • Both checkpoints generated $ORIGINAL_COUNT captions"
    echo "  • All captions are byte-for-byte identical"
    echo "  • Original checkpoint size: $(du -sh "$ORIGINAL_CKPT" | cut -f1)"
    echo "  • Stripped checkpoint size: $(du -sh "$STRIPPED_CKPT" | cut -f1)"
    echo ""
    echo "Conclusion:"
    echo "  The stripped checkpoint is functionally IDENTICAL to the original."
    echo "  You can safely use the stripped checkpoint and delete the original."
    echo ""
    echo "Space saved per checkpoint: $(du -sh "$ORIGINAL_CKPT" | cut -f1) → $(du -sh "$STRIPPED_CKPT" | cut -f1)"
    echo ""
else
    echo "=========================================="
    echo "❌ FAILURE: OUTPUTS DIFFER!"
    echo "=========================================="
    echo ""
    echo "The checkpoints produced different outputs."
    echo "Showing first 10 differences:"
    echo ""
    diff "$ORIGINAL_OUT" "$STRIPPED_OUT" | head -n 20 | sed 's/^/  /'
    echo ""
    echo "Full logs saved to:"
    echo "  Original: /tmp/original_full.log"
    echo "  Stripped: /tmp/stripped_full.log"
    echo ""
    echo "Full outputs saved to:"
    echo "  Original: $ORIGINAL_OUT"
    echo "  Stripped: $STRIPPED_OUT"
    echo ""
    echo "DO NOT delete the original checkpoint!"
    exit 1
fi

# Cleanup
rm -f "$ORIGINAL_OUT" "$STRIPPED_OUT"
echo "=========================================="
