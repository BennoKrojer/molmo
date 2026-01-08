#!/bin/bash
#
# PARALLEL: Extract max L2 norm vision token embeddings for all 9 models
# Uses 8 GPUs, runs all models in parallel (one waits for a free GPU)
#

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================"
echo "Extract Max Token Embeddings (PARALLEL)"
echo "========================================"
echo ""

SCRIPT="scripts/analysis/extract_max_token_embeddings.py"
OUTPUT_DIR="analysis_results/max_token_embeddings"
NUM_IMAGES=10
TARGET_LAYERS="0,4,8,16,24,31"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="logs/max_embeddings_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# All 9 combinations
declare -a COMBINATIONS=(
    "olmo-7b:vit-l-14-336:"
    "olmo-7b:dinov2-large-336:"
    "olmo-7b:siglip:"
    "llama3-8b:vit-l-14-336:"
    "llama3-8b:dinov2-large-336:"
    "llama3-8b:siglip:"
    "qwen2-7b:vit-l-14-336:_seed10"
    "qwen2-7b:dinov2-large-336:"
    "qwen2-7b:siglip:"
)

echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "Images per model: $NUM_IMAGES"
echo ""

GPU=0
PIDS=()
COMBO_NAMES=()

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r llm vision suffix <<< "$combo"

    ckpt_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision}${suffix}"
    ckpt_path="molmo_data/checkpoints/${ckpt_name}/step12000-unsharded"

    if [ ! -d "$ckpt_path" ]; then
        echo "SKIP: Checkpoint not found: $ckpt_path"
        continue
    fi

    # Adjust target layers for Qwen (28 layers, not 32)
    if [ "$llm" == "qwen2-7b" ]; then
        LAYERS="0,4,8,16,24,27"
    else
        LAYERS="$TARGET_LAYERS"
    fi

    log_file="$LOG_DIR/${llm}_${vision}.log"

    echo "GPU $GPU: $llm + $vision"
    echo "  Log: $log_file"

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python "$SCRIPT" \
        --ckpt-path "$ckpt_path" \
        --num-images $NUM_IMAGES \
        --output-dir "$OUTPUT_DIR" \
        --target-layers "$LAYERS" \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    COMBO_NAMES+=("$llm+$vision")

    GPU=$((GPU + 1))

    # Wait for model to load before launching next
    echo "  Waiting for model load..."
    for i in {1..60}; do
        sleep 3
        if grep -q "Model loaded on device" "$log_file" 2>/dev/null; then
            echo "  ✓ Model loaded"
            break
        fi
        if grep -q "Error\|Traceback" "$log_file" 2>/dev/null; then
            echo "  ✗ Error detected, check log"
            break
        fi
    done

    # If all 8 GPUs in use, wait for one to finish
    if [ $GPU -ge 8 ]; then
        echo ""
        echo "All 8 GPUs in use, waiting for a job..."
        wait -n ${PIDS[@]} 2>/dev/null || true

        # Find which finished
        NEW_PIDS=()
        NEW_NAMES=()
        for i in "${!PIDS[@]}"; do
            if kill -0 ${PIDS[$i]} 2>/dev/null; then
                NEW_PIDS+=(${PIDS[$i]})
                NEW_NAMES+=("${COMBO_NAMES[$i]}")
            else
                echo "✓ Completed: ${COMBO_NAMES[$i]}"
                GPU=$((GPU - 1))
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        COMBO_NAMES=("${NEW_NAMES[@]}")
    fi
done

echo ""
echo "Waiting for all jobs to complete..."

for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    status=$?
    if [ $status -eq 0 ]; then
        echo "✓ Completed: ${COMBO_NAMES[$i]}"
    else
        echo "✗ FAILED: ${COMBO_NAMES[$i]} (exit $status)"
    fi
done

echo ""
echo "========================================"
echo "Creating plots..."
echo "========================================"

python << 'PLOT_EOF'
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("analysis_results/max_token_embeddings")
plot_dir = Path("paper_plots/paper_figures_output/l2norm_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

models = [
    ("olmo-7b", "vit-l-14-336", ""),
    ("olmo-7b", "dinov2-large-336", ""),
    ("olmo-7b", "siglip", ""),
    ("llama3-8b", "vit-l-14-336", ""),
    ("llama3-8b", "dinov2-large-336", ""),
    ("llama3-8b", "siglip", ""),
    ("qwen2-7b", "vit-l-14-336", "_seed10"),
    ("qwen2-7b", "dinov2-large-336", ""),
    ("qwen2-7b", "siglip", ""),
]

LLM_DISPLAY = {'olmo-7b': 'OLMo-7B', 'llama3-8b': 'Llama3-8B', 'qwen2-7b': 'Qwen2-7B'}
ENC_DISPLAY = {'vit-l-14-336': 'CLIP', 'dinov2-large-336': 'DINOv2', 'siglip': 'SigLIP'}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for idx, (llm, enc, suffix) in enumerate(models):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    ckpt = f"train_mlp-only_pixmo_cap_resize_{llm}_{enc}{suffix}_step12000-unsharded"
    json_path = output_dir / f"{ckpt}_max_token.json"

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        values = np.array(data['max_embedding_values'])
        l2_norm = data['max_info']['l2_norm']
        layer = data['max_info']['layer']
        stats = data['embedding_stats']

        ax.hist(values, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Mark percentiles
        ax.axvline(x=stats['percentiles']['p1'], color='red', linestyle=':', alpha=0.7, label='p1/p99')
        ax.axvline(x=stats['percentiles']['p99'], color='red', linestyle=':', alpha=0.7)

        ax.set_title(f"{LLM_DISPLAY[llm]} + {ENC_DISPLAY[enc]}\nL2={l2_norm:.0f}, layer={layer}",
                     fontsize=11, fontweight='bold')
        ax.text(0.95, 0.95, f"mean={stats['mean']:.2f}\nstd={stats['std']:.2f}\nmax={stats['max']:.2f}",
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{LLM_DISPLAY[llm]} + {ENC_DISPLAY[enc]}", fontsize=11, fontweight='bold')

    ax.set_xlabel('Embedding Value', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('Max L2 Norm Vision Token: Distribution of Embedding Dimensions\n(Is high L2 from few large values or all values?)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(plot_dir / 'max_token_embedding_values_3x3.png', dpi=200, bbox_inches='tight')
plt.savefig(plot_dir / 'max_token_embedding_values_3x3.pdf', dpi=200, bbox_inches='tight')
print(f"Saved: {plot_dir / 'max_token_embedding_values_3x3.png'}")
plt.close()
PLOT_EOF

echo ""
echo "========================================"
echo "DONE!"
echo "Results: $OUTPUT_DIR"
echo "Plots: paper_plots/paper_figures_output/l2norm_plots/"
echo "========================================"
