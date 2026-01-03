#!/bin/bash
# =============================================================================
# V-Lens Demo Generator
# =============================================================================
# Single command to generate the complete interactive demo viewer.
# 
# Usage:
#   ./generate_demo.sh                    # Default: 10 images
#   ./generate_demo.sh --num-images 20    # Custom number of images
#   ./generate_demo.sh --output-dir path  # Custom output directory
#
# This script orchestrates:
#   1. Main model viewers (3x3 grid: 3 LLMs × 3 Vision Encoders)
#   2. Ablation model viewers (10 ablation variants)
#   3. Unified index.html with navigation to all models
# =============================================================================

set -e  # Exit on error

# Default values
OUTPUT_DIR="analysis_results/unified_viewer_lite"
NUM_IMAGES=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./generate_demo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR    Output directory (default: analysis_results/unified_viewer_lite)"
            echo "  --num-images N      Number of images per model (default: 10)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source ../../env/bin/activate 2>/dev/null || source env/bin/activate 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"

echo "=============================================="
echo "V-Lens Demo Generator"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Images per model: $NUM_IMAGES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Step 1: Generate main model viewers (3x3 grid)
echo "[1/3] Generating main model viewers (9 models)..."
python scripts/analysis/create_unified_viewer.py \
    --output-dir "$OUTPUT_DIR" \
    --num-images "$NUM_IMAGES" \
    2>&1 | tee logs/demo_step1_main.log

echo ""

# Step 2: Generate ablation model viewers
echo "[2/3] Generating ablation model viewers (10 models)..."
python scripts/analysis/generate_ablation_viewers.py \
    --output-dir "$OUTPUT_DIR" \
    --num-images "$NUM_IMAGES" \
    2>&1 | tee logs/demo_step2_ablations.log

echo ""

# Step 3: Update main index.html with ablation links
echo "[3/3] Updating main index with ablation links..."
python scripts/analysis/add_models_to_viewer.py \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee logs/demo_step3_index.log

echo ""
echo "=============================================="
echo "Demo generation complete!"
echo "=============================================="
echo ""
echo "Output: $OUTPUT_DIR/"
echo ""
echo "Contents:"
ls -la "$OUTPUT_DIR/" | head -20
echo ""
echo "To view: Open $OUTPUT_DIR/index.html in a browser"
echo ""


