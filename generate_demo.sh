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
#   ./generate_demo.sh --no-ablations     # Skip ablations section
#
# This script generates:
#   1. Main model viewers (3x3 grid: 3 LLMs x 3 Vision Encoders)
#   2. Ablation studies section (automatically linked if viewers exist)
#
# SIMPLIFIED: Now uses create_unified_viewer.py which handles ablations
#             automatically. No need to run multiple scripts!
# =============================================================================

set -e  # Exit on error

# Default values
OUTPUT_DIR="analysis_results/unified_viewer_lite"
NUM_IMAGES=10
EXTRA_ARGS=""

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
        --no-ablations)
            EXTRA_ARGS="$EXTRA_ARGS --no-ablations"
            shift
            ;;
        -h|--help)
            echo "Usage: ./generate_demo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR    Output directory (default: analysis_results/unified_viewer_lite)"
            echo "  --num-images N      Number of images per model (default: 10)"
            echo "  --no-ablations      Skip ablations section (main 3x3 grid only)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "NOTES:"
            echo "  - Ablations are AUTOMATICALLY included if their viewers exist in output-dir/ablations/"
            echo "  - To generate ablation viewers, run: python scripts/analysis/generate_ablation_viewers.py"
            echo "  - The main index.html links to both main models AND ablations by default"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
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
echo "Extra arguments: $EXTRA_ARGS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Generate complete viewer (main models + ablations links)
# NOTE: create_unified_viewer.py now handles ablations automatically!
echo "Generating unified viewer (main models + ablations)..."
python scripts/analysis/create_unified_viewer.py \
    --output-dir "$OUTPUT_DIR" \
    --num-images "$NUM_IMAGES" \
    $EXTRA_ARGS \
    2>&1 | tee logs/demo_generation.log

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
echo "To sync to website:"
echo "  rsync -av --delete $OUTPUT_DIR/ website/vlm_interp_demo/"
echo "  cd website && git add -A && git commit -m 'Update demo' && git push"
echo ""
