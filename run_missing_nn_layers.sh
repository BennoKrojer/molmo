#!/bin/bash
#
# PARALLEL launcher: Run missing NN layers across GPUs 4-7
# Each job processes ONE model combination on ONE GPU
#

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Fast single-GPU script
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/input_embedding_nearest_neighbors_fast.py"

# Number of images (same as LLM judge evaluation)
NUM_IMAGES=300

# Available GPUs
GPUS=(4 5 6 7)
MAX_PARALLEL=${#GPUS[@]}

echo "========================================"
echo "PARALLEL Missing NN Layers (FAST)"
echo "Strategy: ${MAX_PARALLEL} independent single-GPU jobs"
echo "GPUs: ${GPUS[*]}"
echo "========================================"
echo ""

# Build list of all jobs: (checkpoint_path, layers, name)
declare -a JOBS=()

# OLMo/Llama: layers 30,31
for llm in "llama3-8b" "olmo-7b"; do
    for vision in "vit-l-14-336" "dinov2-large-336" "siglip"; do
        ckpt_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision}"
        ckpt_path="molmo_data/checkpoints/${ckpt_name}/step12000-unsharded"
        if [ -d "$ckpt_path" ]; then
            JOBS+=("${ckpt_path}:30,31:${llm}_${vision}")
        else
            echo "SKIP: Checkpoint not found: $ckpt_path"
        fi
    done
done

# Qwen: layers 26,27
for vision in "vit-l-14-336" "dinov2-large-336" "siglip"; do
    if [ "$vision" == "vit-l-14-336" ]; then
        ckpt_name="train_mlp-only_pixmo_cap_resize_qwen2-7b_${vision}_seed10"
    else
        ckpt_name="train_mlp-only_pixmo_cap_resize_qwen2-7b_${vision}"
    fi
    ckpt_path="molmo_data/checkpoints/${ckpt_name}/step12000-unsharded"
    if [ -d "$ckpt_path" ]; then
        JOBS+=("${ckpt_path}:26,27:qwen2-7b_${vision}")
    else
        echo "SKIP: Checkpoint not found: $ckpt_path"
    fi
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# Log directory
LOG_DIR="logs/missing_nn_layers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Track running jobs
PIDS=()
JOB_NAMES=()
GPU_IDX=0

for job in "${JOBS[@]}"; do
    IFS=':' read -r ckpt_path layers name <<< "$job"
    
    # Get next available GPU
    gpu=${GPUS[$GPU_IDX]}
    log_file="$LOG_DIR/${name}.log"
    
    echo "GPU $gpu: $name (layers $layers)"
    echo "  Checkpoint: $ckpt_path"
    echo "  Log: $log_file"
    
    # Launch in background
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python "$SCRIPT_PATH" \
        --ckpt-path "$ckpt_path" \
        --llm-layers "$layers" \
        --num-images $NUM_IMAGES \
        > "$log_file" 2>&1 &
    
    PIDS+=($!)
    JOB_NAMES+=("$name")
    
    GPU_IDX=$((GPU_IDX + 1))
    
    # Stagger: Wait for model to load before launching next (max 3 min timeout)
    echo "  Waiting for model to load..."
    for i in {1..36}; do
        sleep 5
        if grep -q "Model loaded" "$log_file" 2>/dev/null; then
            echo "  ✓ Model loaded! Launching next..."
            break
        fi
        if [ $i -eq 36 ]; then
            echo "  Timeout waiting for model load, continuing anyway..."
        fi
    done
    
    # If all GPUs in use, wait for one to finish
    if [ $GPU_IDX -ge $MAX_PARALLEL ]; then
        echo ""
        echo "All $MAX_PARALLEL GPUs in use, waiting for a job to complete..."
        
        # Wait for any one job to finish
        wait -n ${PIDS[@]} 2>/dev/null || true
        
        # Find which one finished and reclaim its GPU slot
        NEW_PIDS=()
        NEW_NAMES=()
        for i in "${!PIDS[@]}"; do
            if kill -0 ${PIDS[$i]} 2>/dev/null; then
                NEW_PIDS+=(${PIDS[$i]})
                NEW_NAMES+=("${JOB_NAMES[$i]}")
            else
                echo "✓ Completed: ${JOB_NAMES[$i]}"
                GPU_IDX=$((GPU_IDX - 1))
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        JOB_NAMES=("${NEW_NAMES[@]}")
    fi
done

echo ""
echo "========================================"
echo "All jobs launched! Waiting for completion..."
echo "========================================"
echo ""

# Wait for all remaining jobs
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    status=$?
    if [ $status -eq 0 ]; then
        echo "✓ Completed: ${JOB_NAMES[$i]}"
    else
        echo "✗ FAILED: ${JOB_NAMES[$i]} (exit code: $status)"
    fi
done

echo ""
echo "========================================"
echo "ALL DONE!"
echo "Check logs in: $LOG_DIR"
echo "========================================"

