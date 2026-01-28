#!/bin/bash
# Fix missing visual layers for olmo-7b_vit-l-14-336
#
# Missing:
#   - visual31: completely missing (need all 136 images)
#   - visual4: only 100 images (need 36 supplement)
#   - visual8: only 100 images (need 36 supplement)

set -e

cd /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

CKPT_PATH="molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded"
CONTEXTUAL_DIR="molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"
OUTPUT_DIR="analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded"

# Supplement image indices (same as used in other models)
SUPPLEMENT_INDICES="100,102,108,109,119,134,136,139,153,154,155,156,167,175,187,191,208,216,218,222,234,240,242,252,253,257,258,260,264,267,284,288,291,293,295,299"

echo "=========================================="
echo "Fixing olmo-7b_vit-l-14-336 missing layers"
echo "=========================================="
echo "Checkpoint: $CKPT_PATH"
echo "Contextual: $CONTEXTUAL_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Extract visual31 for all 136 images
echo "[1/3] Extracting visual31 (136 images)..."
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
    --ckpt-path "$CKPT_PATH" \
    --contextual-dir "$CONTEXTUAL_DIR" \
    --visual-layer 31 \
    --num-images 136 \
    --output-dir "$OUTPUT_DIR" \
    --top-k 5

echo "✓ visual31 done"
echo ""

# Step 2: Extract visual4 supplement (36 images)
echo "[2/3] Extracting visual4 supplement (36 images)..."
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
    --ckpt-path "$CKPT_PATH" \
    --contextual-dir "$CONTEXTUAL_DIR" \
    --visual-layer 4 \
    --image-indices "$SUPPLEMENT_INDICES" \
    --output-dir "${OUTPUT_DIR}_visual4_supplement" \
    --top-k 5

echo "✓ visual4 supplement done"
echo ""

# Step 3: Extract visual8 supplement (36 images)
echo "[3/3] Extracting visual8 supplement (36 images)..."
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
    --ckpt-path "$CKPT_PATH" \
    --contextual-dir "$CONTEXTUAL_DIR" \
    --visual-layer 8 \
    --image-indices "$SUPPLEMENT_INDICES" \
    --output-dir "${OUTPUT_DIR}_visual8_supplement" \
    --top-k 5

echo "✓ visual8 supplement done"
echo ""

# Step 4: Merge supplements into main files
echo "[4/4] Merging supplements..."

python3 << 'MERGE_SCRIPT'
import json
from pathlib import Path

base_dir = Path("analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded")

for visual_layer in [4, 8]:
    base_file = base_dir / f"contextual_neighbors_visual{visual_layer}_allLayers.json"
    supp_dir = Path(f"{base_dir}_visual{visual_layer}_supplement")
    supp_file = supp_dir / f"contextual_neighbors_visual{visual_layer}_allLayers.json"

    if not supp_file.exists():
        print(f"WARNING: {supp_file} not found")
        continue

    print(f"Merging visual{visual_layer}...")

    with open(base_file) as f:
        base_data = json.load(f)
    with open(supp_file) as f:
        supp_data = json.load(f)

    base_indices = {r['image_idx'] for r in base_data['results']}
    supp_results = [r for r in supp_data['results'] if r['image_idx'] not in base_indices]

    print(f"  Base: {len(base_data['results'])} images")
    print(f"  Adding: {len(supp_results)} new images")

    base_data['results'].extend(supp_results)
    base_data['results'].sort(key=lambda x: x['image_idx'])

    with open(base_file, 'w') as f:
        json.dump(base_data, f, indent=2)

    print(f"  Final: {len(base_data['results'])} images")

print("Done!")
MERGE_SCRIPT

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

python3 -c "
import json
from pathlib import Path

base = Path('analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded')

for f in sorted(base.glob('contextual_neighbors_visual*_allLayers.json')):
    with open(f) as fp:
        data = json.load(fp)
    layer = f.name.split('visual')[1].split('_')[0]
    count = len(data['results'])
    status = '✓' if count == 136 else f'⚠ INCOMPLETE (have {count}, need 136)'
    print(f'  visual{layer}: {count} images {status}')
"

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
