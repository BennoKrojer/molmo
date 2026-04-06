#!/bin/bash
# Chinese LatentLens hypothesis test for Qwen2.5-VL-32B
# Tests whether late-layer interpretability recovers with Chinese contextual embeddings
#
# Prerequisites: analysis_results/chinese_latentlens_test/chinese_phrases.txt must exist
# GPUs: needs 4 GPUs (0-3) for model loading
#
# Steps:
# 1. Extract contextual embeddings for Chinese phrases (29K phrases, ~5 min)
# 2. Build caches
# 3. Rerun LatentLens search using Chinese cache (100 images × 11 layers)
# 4. Run LLM judge on the new results
set -e
cd "$(dirname "$0")/../.."

source ../../env/bin/activate
export PYTHONPATH=/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo:$PYTHONPATH

CHINESE_PHRASES="analysis_results/chinese_latentlens_test/chinese_phrases.txt"
CHINESE_EMB_DIR="molmo_data/contextual_llm_embeddings_chinese/Qwen_Qwen2.5-VL-32B-Instruct"
CHINESE_RESULTS_DIR="analysis_results/contextual_nearest_neighbors/qwen2_5_vl_chinese"
LAYERS="1 2 4 8 16 32 48 56 62 63"

# Verify Chinese phrases exist
if [ ! -f "$CHINESE_PHRASES" ]; then
    echo "ERROR: $CHINESE_PHRASES not found. Run translation first."
    exit 1
fi
NLINES=$(wc -l < "$CHINESE_PHRASES")
echo "Chinese phrases: $NLINES lines"

echo ""
echo "========================================"
echo "STEP 1: Extract contextual embeddings for Chinese phrases"
echo "========================================"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/analysis/qwen2_5_vl/create_contextual_embeddings.py \
    --vg-file "$CHINESE_PHRASES" \
    --num-captions -1 \
    --layers $LAYERS \
    --batch-size 32 \
    --dataset vg \
    --embedding-dtype float8 \
    --output-dir "$CHINESE_EMB_DIR"

echo ""
echo "========================================"
echo "STEP 2: Build caches"
echo "========================================"
python scripts/analysis/precompute_contextual_caches.py \
    --contextual-base "$(dirname "$CHINESE_EMB_DIR")" \
    --llm-names "$(basename "$CHINESE_EMB_DIR")" \
    --num-workers 4

echo ""
echo "========================================"
echo "STEP 3: Run LatentLens with Chinese cache"
echo "========================================"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/analysis/qwen2_5_vl/contextual_nearest_neighbors_allLayers_singleGPU.py \
    --contextual-dir "$CHINESE_EMB_DIR" \
    --visual-layer "0,1,2,4,8,16,32,48,56,62,63" \
    --num-images 100 \
    --output-dir "$CHINESE_RESULTS_DIR"

echo ""
echo "========================================"
echo "STEP 4: Run LLM judge"
echo "========================================"
# Load API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ~/.openai_key ]; then
        source ~/.openai_key
    fi
fi

EVAL_SCRIPT="latentlens_release/reproduce/scripts/evaluate/evaluate_interpretability.py"
IMAGES_DIR="analysis_results/pixmo_cap_validation_indexed"
JUDGE_OUTPUT="analysis_results/llm_judge_offtheshelf/latentlens_chinese/qwen2.5-vl-32b"
RESULTS_SUBDIR="qwen2_5_vl_chinese/Qwen_Qwen2.5-VL-32B-Instruct"

mkdir -p "$JUDGE_OUTPUT"
for layer in 0 1 2 4 8 16 32 48 56 62 63; do
    echo "  Layer $layer..."
    python "$EVAL_SCRIPT" \
        --results-dir "analysis_results/contextual_nearest_neighbors/$RESULTS_SUBDIR" \
        --images-dir "$IMAGES_DIR" \
        --output-dir "$JUDGE_OUTPUT/layer_${layer}" \
        --layers $layer \
        --num-patches 100 \
        --model-name "qwen2vl" \
        --seed 42
done

# Merge per-layer results
python3 -c "
import json
from pathlib import Path
eval_dir = Path('$JUDGE_OUTPUT')
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

echo ""
echo "========================================"
echo "DONE! Compare results:"
echo "========================================"
python3 -c "
import json
print('English LatentLens vs Chinese LatentLens (Qwen2.5-VL-32B):')
print(f'{\"Layer\":>8} {\"English\":>10} {\"Chinese\":>10} {\"Delta\":>8}')
print('-' * 40)
with open('analysis_results/llm_judge_offtheshelf/latentlens/qwen2.5-vl-32b/evaluation_results.json') as f:
    en = {e['layer']: e['interpretable_fraction']*100 for e in json.load(f)}
with open('$JUDGE_OUTPUT/evaluation_results.json') as f:
    zh = {e['layer']: e['interpretable_fraction']*100 for e in json.load(f)}
for layer in sorted(set(en.keys()) | set(zh.keys())):
    e = en.get(layer, 0)
    z = zh.get(layer, 0)
    d = z - e
    print(f'{layer:>8} {e:>9.1f}% {z:>9.1f}% {d:>+7.1f}%')
"
