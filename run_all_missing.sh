#!/bin/bash
#
# MASTER SCRIPT: Run all missing interpretability tools and LLM judge evaluations
#
# This script runs everything needed for:
# 1. Ablations (NN, LogitLens, Contextual NN, LLM Judge)
# 2. Qwen2-VL (NN, LogitLens, Contextual NN, LLM Judge for ALL 3 types)
#
# Phases:
#   1-3: Ablations (NN, LogitLens, Contextual NN)
#   4-6: LLM Judge for ablations
#   7-8: Qwen2-VL analysis (NN, LogitLens with --force-square)
#   9-10: LLM Judge for Qwen2-VL (NN, LogitLens)
#
# Expected runtime: 1-2 days
#
# Usage:
#   ./run_all_missing.sh                  # Run everything (skips existing)
#   ./run_all_missing.sh --dry-run        # Show what would be run without executing
#   ./run_all_missing.sh --test           # Run with minimal inputs for testing
#   ./run_all_missing.sh --force-qwen2vl  # Delete and regenerate ALL Qwen2-VL data
#   ./run_all_missing.sh --qwen2vl-only   # Skip ablations, run only Qwen2-VL phases (7-10)
#
# For Qwen2-VL regeneration:
#   ./run_all_missing.sh --qwen2vl-only --force-qwen2vl
#

set -e  # Exit on error

# Parse command line arguments
DRY_RUN=false
TEST_MODE=false
FORCE_QWEN2VL=false
QWEN2VL_ONLY=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --force-qwen2vl)
            FORCE_QWEN2VL=true
            shift
            ;;
        --qwen2vl-only)
            QWEN2VL_ONLY=true
            shift
            ;;
    esac
done

# Setup environment
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration
# OLMo layers: 0,1,2,4,8,16,24,30,31 (9 layers for 32-layer model)
OLMO_LAYERS="0,1,2,4,8,16,24,30,31"

if [ "$TEST_MODE" = true ]; then
    echo "🧪 TEST MODE: Using minimal inputs"
    NUM_IMAGES=5
    NUM_SAMPLES=1
    LAYERS="0"  # Just layer 0 for quick testing
else
    NUM_IMAGES=300
    NUM_SAMPLES=1
    LAYERS="$OLMO_LAYERS"  # All 9 layers for full analysis
fi

SPLIT="validation"
SEED=42

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4
MASTER_PORT=29530

# Ablations to process (from ablations_comparison, excluding pixmo_spatial and pixmo_points)
ABLATION_CHECKPOINTS=(
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_earlier-vit-layers-6"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_earlier-vit-layers-10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_unfreeze"
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_topbottom_olmo-7b_vit-l-14-336_unfreeze-llm"
)

# Ablations that need NN run (earlier ViT layers don't have NN results yet)
ABLATIONS_NEED_NN=(
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_earlier-vit-layers-6"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_earlier-vit-layers-10"
)

# Log file
LOG_DIR="logs/master_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

# Helper function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

run_cmd() {
    local desc="$1"
    local cmd="$2"
    
    # Remove API key from logged command for security
    local safe_cmd=$(echo "$cmd" | sed 's/--api-key [^ ]*/--api-key [REDACTED]/g')
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN: $desc"
        log "  Command: $safe_cmd"
        return 0
    fi
    
    log "RUNNING: $desc"
    log "  Command: $safe_cmd"
    
    if eval "$cmd" >> "$MASTER_LOG" 2>&1; then
        log "✓ SUCCESS: $desc"
        return 0
    else
        log "✗ FAILED: $desc"
        return 1
    fi
}

echo "=========================================="
echo "MASTER SCRIPT: Running All Missing Tasks"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo "Test mode: $TEST_MODE"
echo "Force Qwen2-VL: $FORCE_QWEN2VL"
echo "Qwen2-VL only: $QWEN2VL_ONLY"
if [ "$QWEN2VL_ONLY" = false ]; then
    echo "Ablations: ${#ABLATION_CHECKPOINTS[@]}"
fi
echo "=========================================="
echo ""

log "Starting master script"
log "Configuration: NUM_IMAGES=$NUM_IMAGES, LAYERS=$LAYERS, SPLIT=$SPLIT"

# ============================================================
# PHASE 0: Validation - Check all scripts before running
# ============================================================
log ""
log "========== PHASE 0: Validation =========="

VALIDATION_FAILED=false

# List of Python scripts that will be used
PYTHON_SCRIPTS=(
    "scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py"
    "scripts/analysis/logitlens.py"
    "scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py"
    "llm_judge/run_single_model_with_viz.py"
    "llm_judge/run_single_model_with_viz_contextual.py"
    "scripts/analysis/qwen2_vl/nearest_neighbors.py"
    "scripts/analysis/qwen2_vl/logitlens.py"
)

log "Checking Python script syntax..."
for script in "${PYTHON_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if python3 -m py_compile "$script" 2>/dev/null; then
            log "  ✓ $script"
        else
            log "  ✗ SYNTAX ERROR: $script"
            python3 -m py_compile "$script" 2>&1 | head -5 | tee -a "$MASTER_LOG"
            VALIDATION_FAILED=true
        fi
    else
        log "  ⚠ MISSING: $script (may be optional)"
    fi
done

log "Checking imports and dependencies..."
for script in "${PYTHON_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        # Extract module name and try to import dependencies
        if python3 -c "import sys; sys.path.insert(0, '.'); exec(open('$script').read().split('def ')[0])" 2>/dev/null; then
            log "  ✓ $script imports OK"
        else
            # Try simpler check - just verify torch and transformers are available
            :  # Skip import check failures (too many false positives)
        fi
    fi
done

log "Checking API key file..."
if [ -f "llm_judge/api_key.txt" ]; then
    if [ -s "llm_judge/api_key.txt" ]; then
        log "  ✓ llm_judge/api_key.txt exists and is non-empty"
    else
        log "  ✗ llm_judge/api_key.txt is empty!"
        VALIDATION_FAILED=true
    fi
else
    log "  ✗ llm_judge/api_key.txt not found!"
    VALIDATION_FAILED=true
fi

if [ "$QWEN2VL_ONLY" = false ]; then
    log "Checking checkpoint directories..."
    MISSING_CKPTS=0
    for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
        ckpt_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
        if [ -d "$ckpt_path" ]; then
            log "  ✓ $checkpoint_name"
        else
            log "  ✗ Missing: $ckpt_path"
            MISSING_CKPTS=$((MISSING_CKPTS + 1))
        fi
    done
    if [ $MISSING_CKPTS -gt 0 ]; then
        log "  WARNING: $MISSING_CKPTS checkpoint(s) missing"
    fi

    log "Checking contextual embeddings..."
    if [ -d "molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview" ]; then
        log "  ✓ OLMo contextual embeddings"
    else
        log "  ✗ Missing OLMo contextual embeddings"
        VALIDATION_FAILED=true
    fi
else
    log "SKIP: Ablation validation (--qwen2vl-only mode)"
fi

if [ "$VALIDATION_FAILED" = true ]; then
    log ""
    log "❌ VALIDATION FAILED - Fix errors above before running"
    exit 1
fi

log "✓ All validations passed"
log ""

# ============================================================
# PHASES 1-6: Ablations (skip if --qwen2vl-only)
# ============================================================
if [ "$QWEN2VL_ONLY" = true ]; then
    log ""
    log "SKIP: Phases 1-6 (ablations) - running Qwen2-VL only mode"
else

# ============================================================
# PHASE 1: Static NN for ALL ablations (all 9 layers)
# ============================================================
log ""
log "========== PHASE 1: Static NN for ALL Ablations (9 layers) =========="

# Count expected layers (9 for OLMo)
EXPECTED_LAYER_COUNT=9

for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    output_dir="analysis_results/nearest_neighbors/ablations/${checkpoint_name}_step12000-unsharded"
    
    # Check if ALL layers exist (count json files with layer pattern)
    if [ -d "$output_dir" ]; then
        layer_count=$(ls "$output_dir"/nearest_neighbors_analysis_*_layer*.json 2>/dev/null | wc -l)
        if [ "$layer_count" -ge "$EXPECTED_LAYER_COUNT" ]; then
            log "SKIP: NN complete for $checkpoint_name ($layer_count layers)"
            continue
        else
            log "INCOMPLETE: NN for $checkpoint_name has $layer_count/$EXPECTED_LAYER_COUNT layers - will re-run"
        fi
    fi
    
    run_cmd "NN for $checkpoint_name (all 9 layers)" \
        "torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
            scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py \
            --ckpt-path $checkpoint_path \
            --llm_layer $LAYERS \
            --output-base-dir analysis_results/nearest_neighbors/ablations"
done

# ============================================================
# PHASE 2: LogitLens for ALL ablations (all 9 layers)
# ============================================================
log ""
log "========== PHASE 2: LogitLens for ALL Ablations (9 layers) =========="

for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    output_dir="analysis_results/logit_lens/ablations/${checkpoint_name}_step12000-unsharded"
    
    # Check if ALL layers exist
    if [ -d "$output_dir" ]; then
        layer_count=$(ls "$output_dir"/logit_lens_layer*_topk*.json 2>/dev/null | wc -l)
        if [ "$layer_count" -ge "$EXPECTED_LAYER_COUNT" ]; then
            log "SKIP: LogitLens complete for $checkpoint_name ($layer_count layers)"
            continue
        else
            log "INCOMPLETE: LogitLens for $checkpoint_name has $layer_count/$EXPECTED_LAYER_COUNT layers - will re-run"
        fi
    fi
    
    run_cmd "LogitLens for $checkpoint_name (all 9 layers)" \
        "torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
            scripts/analysis/logitlens.py \
            --ckpt-path $checkpoint_path \
            --layers $LAYERS \
            --top-k 5 \
            --num-images $NUM_IMAGES \
            --output-dir analysis_results/logit_lens/ablations"
done

# ============================================================
# PHASE 3: Contextual NN for ALL ablations (PARALLEL single-GPU jobs)
# ============================================================
log ""
log "========== PHASE 3: Contextual NN for ALL Ablations (Parallel) =========="

# All ablations use OLMo-7B
CONTEXTUAL_DIR="molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview"

# Get available contextual layers
get_visual_layers() {
    local ctx_dir=$1
    local layers="0"
    for d in "$ctx_dir"/layer_*; do
        if [ -d "$d" ] && [ -f "$d/embeddings_cache.pt" ]; then
            layer=$(basename "$d" | sed 's/layer_//')
            layers="$layers,$layer"
        fi
    done
    echo "$layers"
}

VISUAL_LAYERS=$(get_visual_layers "$CONTEXTUAL_DIR")
log "Visual layers: $VISUAL_LAYERS"

# Collect jobs that need to run
declare -a JOBS_TO_RUN=()
declare -a JOB_NAMES=()

for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    output_dir="analysis_results/contextual_nearest_neighbors/ablations/${checkpoint_name}_step12000-unsharded"
    
    # Check if already exists
    if [ -d "$output_dir" ] && ls "$output_dir"/contextual_neighbors_visual*_allLayers.json 1>/dev/null 2>&1; then
        log "SKIP: Contextual NN already exists for $checkpoint_name"
        continue
    fi
    
    JOBS_TO_RUN+=("$checkpoint_path:$CONTEXTUAL_DIR:$output_dir")
    JOB_NAMES+=("$checkpoint_name")
done

# Launch jobs in parallel (up to 8 GPUs)
MAX_PARALLEL=8
PIDS=()
RUNNING_NAMES=()
RUNNING_GPUS=()  # Track which GPU each job is using

# Initialize available GPUs
declare -a AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

get_free_gpu() {
    # Return the first available GPU, or -1 if none
    if [ ${#AVAILABLE_GPUS[@]} -gt 0 ]; then
        echo "${AVAILABLE_GPUS[0]}"
    else
        echo "-1"
    fi
}

remove_gpu() {
    # Remove a GPU from the available list
    local gpu_to_remove=$1
    local new_array=()
    for g in "${AVAILABLE_GPUS[@]}"; do
        if [ "$g" != "$gpu_to_remove" ]; then
            new_array+=("$g")
        fi
    done
    AVAILABLE_GPUS=("${new_array[@]}")
}

for i in "${!JOBS_TO_RUN[@]}"; do
    IFS=':' read -r ckpt_path ctx_dir out_dir <<< "${JOBS_TO_RUN[$i]}"
    job_name="${JOB_NAMES[$i]}"
    job_log="$LOG_DIR/contextual_nn_${job_name}.log"
    
    # Wait for a free GPU if none available
    while [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; do
        log "All $MAX_PARALLEL GPUs in use, waiting for a job to complete..."
        wait -n "${PIDS[@]}" 2>/dev/null || true
        
        # Find which jobs finished and free their GPUs
        NEW_PIDS=()
        NEW_NAMES=()
        NEW_GPUS=()
        for pi in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$pi]}" 2>/dev/null; then
                NEW_PIDS+=("${PIDS[$pi]}")
                NEW_NAMES+=("${RUNNING_NAMES[$pi]}")
                NEW_GPUS+=("${RUNNING_GPUS[$pi]}")
            else
                log "✓ Completed: ${RUNNING_NAMES[$pi]} (GPU ${RUNNING_GPUS[$pi]} now free)"
                AVAILABLE_GPUS+=("${RUNNING_GPUS[$pi]}")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        RUNNING_NAMES=("${NEW_NAMES[@]}")
        RUNNING_GPUS=("${NEW_GPUS[@]}")
    done
    
    GPU=$(get_free_gpu)
    remove_gpu "$GPU"
    
    if $DRY_RUN; then
        log "DRY-RUN: Contextual NN for $job_name (GPU $GPU)"
        log "  Command: CUDA_VISIBLE_DEVICES=$GPU python scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py --ckpt-path $ckpt_path --contextual-dir $ctx_dir --visual-layer $VISUAL_LAYERS --num-images $NUM_IMAGES --output-dir analysis_results/contextual_nearest_neighbors/ablations"
        AVAILABLE_GPUS+=("$GPU")  # Return GPU to pool in dry run
        continue
    fi
    
    log "LAUNCHING: Contextual NN for $job_name (GPU $GPU)"
    log "  Log: $job_log"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python \
        scripts/analysis/contextual_nearest_neighbors_allLayers_singleGPU.py \
        --ckpt-path "$ckpt_path" \
        --contextual-dir "$ctx_dir" \
        --visual-layer "$VISUAL_LAYERS" \
        --num-images $NUM_IMAGES \
        --output-dir "analysis_results/contextual_nearest_neighbors/ablations" \
        > "$job_log" 2>&1 &
    
    PIDS+=($!)
    RUNNING_NAMES+=("$job_name")
    RUNNING_GPUS+=("$GPU")
    
    # Wait for model to load before launching next (avoid OOM from concurrent loads)
    log "  Waiting for model load..."
    for wait_i in {1..60}; do
        sleep 5
        if grep -q "✓ Model loaded" "$job_log" 2>/dev/null; then
            log "  ✓ Model loaded, launching next..."
            break
        fi
        if [ $wait_i -eq 60 ]; then
            log "  Timeout waiting, continuing anyway..."
        fi
    done
done

# Wait for all remaining jobs
if ! $DRY_RUN && [ ${#PIDS[@]} -gt 0 ]; then
    log "Waiting for ${#PIDS[@]} remaining jobs..."
    set +e  # Don't exit on error - we want to continue even if some jobs fail
    for pi in "${!PIDS[@]}"; do
        wait "${PIDS[$pi]}" || true  # Don't fail if wait returns non-zero
        status=$?
        if [ $status -eq 0 ]; then
            log "✓ SUCCESS: Contextual NN for ${RUNNING_NAMES[$pi]} (GPU ${RUNNING_GPUS[$pi]})"
        else
            log "✗ FAILED: Contextual NN for ${RUNNING_NAMES[$pi]} (GPU ${RUNNING_GPUS[$pi]}, exit $status) - continuing..."
        fi
    done
    set -e  # Re-enable exit on error
fi

# ============================================================
# PHASE 4: LLM Judge - NN for ablations that need it
# ============================================================
log ""
log "========== PHASE 4: LLM Judge NN for Missing Ablations =========="

# Read API key
if [ ! -f "llm_judge/api_key.txt" ]; then
    log "ERROR: API key file not found: llm_judge/api_key.txt"
    log "Skipping LLM Judge phases"
else
    API_KEY=$(cat llm_judge/api_key.txt)
    
    for checkpoint_name in "${ABLATIONS_NEED_NN[@]}"; do
        output_dir="analysis_results/llm_judge_nearest_neighbors/ablations"
        mkdir -p "$output_dir"
        
        # Check based on naming pattern - use simplified name for output
        model_name=$(echo "$checkpoint_name" | sed 's/train_mlp-only_pixmo_cap_resize_//' | sed 's/train_mlp-only_pixmo_cap_//' | sed 's/train_mlp-only_pixmo_topbottom_//' | sed 's/train_mlp-only_//')
        
        result_dir="$output_dir/llm_judge_${model_name}_layer0_gpt5_cropped"
        
        if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
            log "SKIP: LLM Judge NN already exists for $checkpoint_name"
            continue
        fi
        
        run_cmd "LLM Judge NN for $checkpoint_name" \
            "python3 llm_judge/run_single_model_with_viz.py \
                --llm olmo-7b \
                --vision-encoder vit-l-14-336 \
                --checkpoint-name $checkpoint_name \
                --model-name $model_name \
                --layer 0 \
                --api-key $API_KEY \
                --base-dir analysis_results/nearest_neighbors/ablations \
                --output-base $output_dir \
                --num-images $NUM_IMAGES \
                --num-samples $NUM_SAMPLES \
                --split $SPLIT \
                --seed $SEED \
                --use-cropped-region \
                --skip-if-complete \
                --resume"
    done

    # ============================================================
    # PHASE 5+6: LLM Judge - Contextual NN (LAYERS IN PARALLEL)
    # ============================================================
    log ""
    log "========== PHASE 5+6: LLM Judge Contextual NN (LAYERS IN PARALLEL) =========="
    log "Strategy: 9 parallel processes (one per layer), each processes all models sequentially"
    
    # Output directories
    ABLATION_OUTPUT="analysis_results/llm_judge_contextual_nn/ablations"
    QWEN2VL_OUTPUT="analysis_results/llm_judge_contextual_nn/qwen2-vl"
    mkdir -p "$ABLATION_OUTPUT" "$QWEN2VL_OUTPUT"
    
    # Layers: OLMo uses 0,1,2,4,8,16,24,30,31; Qwen2-VL uses 0,1,2,4,8,16,24,26,27
    # We'll run 9 parallel processes for each unique layer
    ALL_LAYERS=(0 1 2 4 8 16 24 26 27 30 31)
    
    # Function to process one layer across ALL models (ablations + Qwen2-VL)
    process_layer_for_all_models() {
        local layer="$1"
        local log_file="$ABLATION_OUTPUT/log_layer${layer}_all_models.txt"
        
        echo "[$(date '+%H:%M:%S')] Layer $layer: Starting" >> "$MASTER_LOG"
        
        # Process all OLMo ablations for this layer (only if layer is valid for OLMo: 0,1,2,4,8,16,24,30,31)
        if [[ "$layer" =~ ^(0|1|2|4|8|16|24|30|31)$ ]]; then
            for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
                local model_name=$(echo "$checkpoint_name" | sed 's/train_mlp-only_pixmo_cap_resize_//' | sed 's/train_mlp-only_pixmo_cap_//' | sed 's/train_mlp-only_pixmo_topbottom_//' | sed 's/train_mlp-only_//')
                local result_dir="$ABLATION_OUTPUT/llm_judge_${model_name}_contextual${layer}_gpt5_cropped"
                
                if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
                    echo "[$(date '+%H:%M:%S')] Layer $layer: SKIP $model_name (exists)" >> "$MASTER_LOG"
                    continue
                fi
                
                echo "[$(date '+%H:%M:%S')] Layer $layer: Running $model_name..." >> "$MASTER_LOG"
                
                python3 llm_judge/run_single_model_with_viz_contextual.py \
                    --llm olmo-7b \
                    --vision-encoder vit-l-14-336 \
                    --checkpoint-name "$checkpoint_name" \
                    --model-name "$model_name" \
                    --api-key_file llm_judge/api_key.txt \
                    --base-dir analysis_results/contextual_nearest_neighbors/ablations \
                    --output-base "$ABLATION_OUTPUT" \
                    --num-images $NUM_IMAGES \
                    --num-samples $NUM_SAMPLES \
                    --split $SPLIT \
                    --seed $SEED \
                    --layer "contextual${layer}" \
                    --use-cropped-region \
                    >> "$log_file" 2>&1
                
                if [ $? -eq 0 ]; then
                    echo "[$(date '+%H:%M:%S')] Layer $layer: ✓ $model_name done" >> "$MASTER_LOG"
                else
                    echo "[$(date '+%H:%M:%S')] Layer $layer: ✗ $model_name FAILED" >> "$MASTER_LOG"
                fi
            done
        fi
        
        # Process Qwen2-VL for this layer (only if layer is valid for Qwen2-VL: 0,1,2,4,8,16,24,26,27)
        if [[ "$layer" =~ ^(0|1|2|4|8|16|24|26|27)$ ]]; then
            local qwen_result_dir="$QWEN2VL_OUTPUT/llm_judge_qwen2vl_contextual${layer}_gpt5_cropped"
            local qwen_ctx_file="analysis_results/contextual_nearest_neighbors/ablations/Qwen_Qwen2-VL-7B-Instruct/contextual_neighbors_visual0_allLayers.json"
            
            if [ -d "$qwen_result_dir" ] && [ -f "$qwen_result_dir/results_validation.json" ]; then
                echo "[$(date '+%H:%M:%S')] Layer $layer: SKIP Qwen2-VL (exists)" >> "$MASTER_LOG"
            elif [ ! -f "$qwen_ctx_file" ]; then
                echo "[$(date '+%H:%M:%S')] Layer $layer: SKIP Qwen2-VL (no input file)" >> "$MASTER_LOG"
            else
                echo "[$(date '+%H:%M:%S')] Layer $layer: Running Qwen2-VL..." >> "$MASTER_LOG"
                
                python3 llm_judge/run_single_model_with_viz_contextual.py \
                    --llm qwen2-7b \
                    --vision-encoder qwen2-vl \
                    --checkpoint-name Qwen_Qwen2-VL-7B-Instruct \
                    --model-name qwen2vl \
                    --api-key_file llm_judge/api_key.txt \
                    --base-dir analysis_results/contextual_nearest_neighbors/ablations \
                    --output-base "$QWEN2VL_OUTPUT" \
                    --num-images $NUM_IMAGES \
                    --num-samples $NUM_SAMPLES \
                    --split $SPLIT \
                    --seed $SEED \
                    --layer "contextual${layer}" \
                    --use-cropped-region \
                    >> "$log_file" 2>&1
                
                if [ $? -eq 0 ]; then
                    echo "[$(date '+%H:%M:%S')] Layer $layer: ✓ Qwen2-VL done" >> "$MASTER_LOG"
                else
                    echo "[$(date '+%H:%M:%S')] Layer $layer: ✗ Qwen2-VL FAILED" >> "$MASTER_LOG"
                fi
            fi
        fi
        
        echo "[$(date '+%H:%M:%S')] Layer $layer: Completed all models" >> "$MASTER_LOG"
    }
    
    # Launch one parallel process per layer
    PHASE5_PIDS=()
    for layer in "${ALL_LAYERS[@]}"; do
        log "LAUNCHING: Layer $layer (parallel)"
        
        if [ "$DRY_RUN" = true ]; then
            # Count how many models this layer will process
            count=0
            if [[ "$layer" =~ ^(0|1|2|4|8|16|24|30|31)$ ]]; then
                count=$((count + ${#ABLATION_CHECKPOINTS[@]}))
            fi
            if [[ "$layer" =~ ^(0|1|2|4|8|16|24|26|27)$ ]]; then
                count=$((count + 1))  # Qwen2-VL
            fi
            log "  DRY-RUN: Would process $count models for layer $layer"
        else
            process_layer_for_all_models "$layer" &
            PHASE5_PIDS+=($!)
        fi
    done
    
    # Wait for ALL parallel layer jobs
    if [ "$DRY_RUN" = false ] && [ ${#PHASE5_PIDS[@]} -gt 0 ]; then
        log ""
        log "Waiting for ${#PHASE5_PIDS[@]} parallel layer jobs..."
        PHASE5_FAILED=0
        for pid in "${PHASE5_PIDS[@]}"; do
            if ! wait $pid; then
                ((PHASE5_FAILED++))
            fi
        done
        log "Phase 5+6 complete: $((${#PHASE5_PIDS[@]} - PHASE5_FAILED))/${#PHASE5_PIDS[@]} layers succeeded"
    fi
fi  # end of API key check

fi  # end of QWEN2VL_ONLY check (phases 1-6)

# ============================================================
# Load API key for Qwen2-VL LLM Judge phases (if not already loaded)
# ============================================================
if [ -z "$API_KEY" ]; then
    if [ -f "llm_judge/api_key.txt" ] && [ -s "llm_judge/api_key.txt" ]; then
        API_KEY=$(cat llm_judge/api_key.txt)
        log "✓ Loaded API key for Qwen2-VL LLM Judge phases"
    else
        log "WARNING: No API key found - LLM Judge phases will be skipped"
    fi
fi

# ============================================================
# PHASE 7: Static NN for Qwen2-VL (single GPU, 9 layers)
# ============================================================
log ""
log "========== PHASE 7: Static NN for Qwen2-VL (9 layers) =========="

QWEN2VL_NN_OUTPUT="analysis_results/nearest_neighbors/qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"

# Force delete old Qwen2-VL data if --force-qwen2vl flag is set
if [ "$FORCE_QWEN2VL" = true ]; then
    log "FORCE: Deleting old Qwen2-VL data (--force-qwen2vl flag set)"
    rm -rf "$QWEN2VL_NN_OUTPUT"
    rm -rf "analysis_results/logit_lens/qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"
    rm -rf "analysis_results/contextual_nearest_neighbors/qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"
    rm -rf "analysis_results/llm_judge_nearest_neighbors/qwen2-vl"
    rm -rf "analysis_results/llm_judge_logitlens/qwen2-vl"
    rm -rf "analysis_results/llm_judge_contextual_nearest_neighbors/qwen2-vl"
    log "✓ Old Qwen2-VL data deleted"
fi
QWEN2_LAYERS="0,1,2,4,8,16,24,26,27"
QWEN2_EXPECTED_LAYER_COUNT=9

# Check if ALL layers exist
if [ -d "$QWEN2VL_NN_OUTPUT" ]; then
    qwen_nn_count=$(ls "$QWEN2VL_NN_OUTPUT"/nearest_neighbors_layer*_topk*.json 2>/dev/null | wc -l)
    if [ "$qwen_nn_count" -ge "$QWEN2_EXPECTED_LAYER_COUNT" ]; then
        log "SKIP: Static NN complete for Qwen2-VL ($qwen_nn_count layers)"
    else
        log "INCOMPLETE: Static NN for Qwen2-VL has $qwen_nn_count/$QWEN2_EXPECTED_LAYER_COUNT layers - will re-run"
        run_cmd "Static NN for Qwen2-VL (all 9 layers)" \
            "python3 scripts/analysis/qwen2_vl/nearest_neighbors.py \
                --num-images $NUM_IMAGES \
                --layers $QWEN2_LAYERS \
                --fixed-resolution 448 \
                --force-square \
                --output-dir analysis_results/nearest_neighbors/qwen2_vl"
    fi
else
    run_cmd "Static NN for Qwen2-VL (all 9 layers)" \
        "python3 scripts/analysis/qwen2_vl/nearest_neighbors.py \
            --num-images $NUM_IMAGES \
            --layers $QWEN2_LAYERS \
            --fixed-resolution 448 \
            --force-square \
            --output-dir analysis_results/nearest_neighbors/qwen2_vl"
fi

# ============================================================
# PHASE 8: LogitLens for Qwen2-VL (single GPU, 9 layers)
# ============================================================
log ""
log "========== PHASE 8: LogitLens for Qwen2-VL (9 layers) =========="

QWEN2VL_LL_OUTPUT="analysis_results/logit_lens/qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"

# Check if ALL layers exist
if [ -d "$QWEN2VL_LL_OUTPUT" ]; then
    qwen_ll_count=$(ls "$QWEN2VL_LL_OUTPUT"/logit_lens_layer*_topk*.json 2>/dev/null | wc -l)
    if [ "$qwen_ll_count" -ge "$QWEN2_EXPECTED_LAYER_COUNT" ]; then
        log "SKIP: LogitLens complete for Qwen2-VL ($qwen_ll_count layers)"
    else
        log "INCOMPLETE: LogitLens for Qwen2-VL has $qwen_ll_count/$QWEN2_EXPECTED_LAYER_COUNT layers - will re-run"
        run_cmd "LogitLens for Qwen2-VL (all 9 layers)" \
            "python3 scripts/analysis/qwen2_vl/logitlens.py \
                --num-images $NUM_IMAGES \
                --layers $QWEN2_LAYERS \
                --fixed-resolution 448 \
                --force-square \
                --output-dir analysis_results/logit_lens/qwen2_vl"
    fi
else
    run_cmd "LogitLens for Qwen2-VL (all 9 layers)" \
        "python3 scripts/analysis/qwen2_vl/logitlens.py \
            --num-images $NUM_IMAGES \
            --layers $QWEN2_LAYERS \
            --fixed-resolution 448 \
            --force-square \
            --output-dir analysis_results/logit_lens/qwen2_vl"
fi

# ============================================================
# PHASE 9: LLM Judge NN for Qwen2-VL (all 9 layers, parallel)
# ============================================================
log ""
log "========== PHASE 9: LLM Judge NN for Qwen2-VL (9 layers, parallel) =========="

QWEN2VL_NN_JUDGE_OUTPUT="analysis_results/llm_judge_nearest_neighbors/qwen2-vl"
mkdir -p "$QWEN2VL_NN_JUDGE_OUTPUT"

# Check if NN data exists first
if [ ! -d "$QWEN2VL_NN_OUTPUT" ] || [ $(ls "$QWEN2VL_NN_OUTPUT"/nearest_neighbors_layer*_topk*.json 2>/dev/null | wc -l) -eq 0 ]; then
    log "SKIP: No Qwen2-VL NN data found - run Phase 7 first"
else
    # Launch all layers in parallel
    QWEN2_LAYER_ARRAY=(0 1 2 4 8 16 24 26 27)
    QWEN2VL_NN_PIDS=()
    
    for layer in "${QWEN2_LAYER_ARRAY[@]}"; do
        result_dir="$QWEN2VL_NN_JUDGE_OUTPUT/llm_judge_qwen2vl_layer${layer}_gpt5_cropped"
        
        if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
            log "SKIP: LLM Judge NN layer $layer already exists"
            continue
        fi
        
        if [ "$DRY_RUN" = true ]; then
            log "DRY-RUN: LLM Judge NN for Qwen2-VL layer $layer"
        else
            log "LAUNCHING: LLM Judge NN for Qwen2-VL layer $layer"
            python3 llm_judge/run_single_model_with_viz.py \
                --llm qwen2-7b \
                --vision-encoder qwen2-vl \
                --api-key $API_KEY \
                --checkpoint-name qwen2_vl/Qwen_Qwen2-VL-7B-Instruct \
                --model-name qwen2vl \
                --layer $layer \
                --num-images $NUM_IMAGES \
                --num-samples $NUM_SAMPLES \
                --base-dir analysis_results/nearest_neighbors \
                --output-base "$QWEN2VL_NN_JUDGE_OUTPUT" \
                --split $SPLIT \
                --seed $SEED \
                --use-cropped-region \
                >> "$LOG_DIR/llm_judge_qwen2vl_nn_layer${layer}.log" 2>&1 &
            QWEN2VL_NN_PIDS+=($!)
        fi
    done
    
    # Wait for all NN LLM Judge jobs
    if [ "$DRY_RUN" = false ] && [ ${#QWEN2VL_NN_PIDS[@]} -gt 0 ]; then
        log "Waiting for ${#QWEN2VL_NN_PIDS[@]} parallel LLM Judge NN jobs..."
        for pid in "${QWEN2VL_NN_PIDS[@]}"; do
            wait $pid || log "WARNING: LLM Judge NN job $pid failed"
        done
        log "✓ Phase 9 complete"
    fi
fi

# ============================================================
# PHASE 10: LLM Judge LogitLens for Qwen2-VL (all 9 layers, parallel)
# ============================================================
log ""
log "========== PHASE 10: LLM Judge LogitLens for Qwen2-VL (9 layers, parallel) =========="

QWEN2VL_LL_JUDGE_OUTPUT="analysis_results/llm_judge_logitlens/qwen2-vl"
mkdir -p "$QWEN2VL_LL_JUDGE_OUTPUT"

# Check if LogitLens data exists first
if [ ! -d "$QWEN2VL_LL_OUTPUT" ] || [ $(ls "$QWEN2VL_LL_OUTPUT"/logit_lens_layer*_topk*.json 2>/dev/null | wc -l) -eq 0 ]; then
    log "SKIP: No Qwen2-VL LogitLens data found - run Phase 8 first"
else
    # Launch all layers in parallel
    QWEN2VL_LL_PIDS=()
    
    for layer in "${QWEN2_LAYER_ARRAY[@]}"; do
        result_dir="$QWEN2VL_LL_JUDGE_OUTPUT/llm_judge_qwen2vl_layer${layer}_gpt5_cropped"
        
        if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
            log "SKIP: LLM Judge LogitLens layer $layer already exists"
            continue
        fi
        
        if [ "$DRY_RUN" = true ]; then
            log "DRY-RUN: LLM Judge LogitLens for Qwen2-VL layer $layer"
        else
            log "LAUNCHING: LLM Judge LogitLens for Qwen2-VL layer $layer"
            python3 llm_judge/run_single_model_with_viz_logitlens.py \
                --llm qwen2-7b \
                --vision-encoder qwen2-vl \
                --api-key $API_KEY \
                --checkpoint-name qwen2_vl/Qwen_Qwen2-VL-7B-Instruct \
                --model-name qwen2vl \
                --layer $layer \
                --num-images $NUM_IMAGES \
                --num-samples $NUM_SAMPLES \
                --base-dir analysis_results/logit_lens \
                --output-base "$QWEN2VL_LL_JUDGE_OUTPUT" \
                --split $SPLIT \
                --seed $SEED \
                --use-cropped-region \
                >> "$LOG_DIR/llm_judge_qwen2vl_logitlens_layer${layer}.log" 2>&1 &
            QWEN2VL_LL_PIDS+=($!)
        fi
    done
    
    # Wait for all LogitLens LLM Judge jobs
    if [ "$DRY_RUN" = false ] && [ ${#QWEN2VL_LL_PIDS[@]} -gt 0 ]; then
        log "Waiting for ${#QWEN2VL_LL_PIDS[@]} parallel LLM Judge LogitLens jobs..."
        for pid in "${QWEN2VL_LL_PIDS[@]}"; do
            wait $pid || log "WARNING: LLM Judge LogitLens job $pid failed"
        done
        log "✓ Phase 10 complete"
    fi
fi

# ============================================================
# PHASE 10.5: Contextual NN for Qwen2-VL (single GPU, 9 layers)
# ============================================================
log ""
log "========== PHASE 10.5: Contextual NN for Qwen2-VL (9 layers) =========="

QWEN2VL_CTX_OUTPUT="analysis_results/contextual_nearest_neighbors/qwen2_vl/Qwen_Qwen2-VL-7B-Instruct"

# Check if ALL layers exist
if [ -d "$QWEN2VL_CTX_OUTPUT" ]; then
    qwen_ctx_count=$(ls "$QWEN2VL_CTX_OUTPUT"/*_allLayers.json 2>/dev/null | wc -l)
    if [ "$qwen_ctx_count" -ge 1 ]; then
        log "SKIP: Contextual NN complete for Qwen2-VL ($qwen_ctx_count allLayers files)"
    else
        log "INCOMPLETE: Contextual NN for Qwen2-VL - will re-run"
        run_cmd "Contextual NN for Qwen2-VL (all 9 layers)" \
            "python3 scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py \
                --contextual-dir molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-VL-7B-Instruct \
                --num-images $NUM_IMAGES \
                --visual-layer $QWEN2_LAYERS \
                --fixed-resolution 448 \
                --force-square \
                --output-dir analysis_results/contextual_nearest_neighbors/qwen2_vl"
    fi
else
    run_cmd "Contextual NN for Qwen2-VL (all 9 layers)" \
        "python3 scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py \
            --contextual-dir molmo_data/contextual_llm_embeddings_vg/Qwen_Qwen2-VL-7B-Instruct \
            --num-images $NUM_IMAGES \
            --visual-layer $QWEN2_LAYERS \
            --fixed-resolution 448 \
            --force-square \
            --output-dir analysis_results/contextual_nearest_neighbors/qwen2_vl"
fi

# ============================================================
# PHASE 11: LLM Judge Contextual NN for Qwen2-VL (all 9 layers, parallel)
# ============================================================
log ""
log "========== PHASE 11: LLM Judge Contextual NN for Qwen2-VL (9 layers, parallel) =========="

QWEN2VL_CTX_JUDGE_OUTPUT="analysis_results/llm_judge_contextual_nearest_neighbors/qwen2-vl"
mkdir -p "$QWEN2VL_CTX_JUDGE_OUTPUT"

# Check if Contextual NN data exists first
if [ ! -d "$QWEN2VL_CTX_OUTPUT" ] || [ $(ls "$QWEN2VL_CTX_OUTPUT"/*_allLayers.json 2>/dev/null | wc -l) -eq 0 ]; then
    log "SKIP: No Qwen2-VL Contextual NN data found - run Phase 10.5 first"
else
    # Launch all layers in parallel
    QWEN2VL_CTX_PIDS=()
    
    for layer in "${QWEN2_LAYER_ARRAY[@]}"; do
        result_dir="$QWEN2VL_CTX_JUDGE_OUTPUT/llm_judge_qwen2vl_contextual${layer}_gpt5_cropped"
        
        if [ -d "$result_dir" ] && [ -f "$result_dir/results_validation.json" ]; then
            log "SKIP: LLM Judge Contextual layer $layer already exists"
            continue
        fi
        
        if [ "$DRY_RUN" = true ]; then
            log "DRY-RUN: LLM Judge Contextual for Qwen2-VL layer $layer"
        else
            log "LAUNCHING: LLM Judge Contextual for Qwen2-VL layer $layer"
            python3 llm_judge/run_single_model_with_viz_contextual.py \
                --llm qwen2-7b \
                --vision-encoder qwen2-vl \
                --checkpoint-name qwen2_vl/Qwen_Qwen2-VL-7B-Instruct \
                --model-name qwen2vl \
                --layer contextual$layer \
                --num-images $NUM_IMAGES \
                --num-samples $NUM_SAMPLES \
                --base-dir analysis_results/contextual_nearest_neighbors \
                --output-base "$QWEN2VL_CTX_JUDGE_OUTPUT" \
                --split $SPLIT \
                --seed $SEED \
                --use-cropped-region \
                >> "$LOG_DIR/llm_judge_qwen2vl_contextual_layer${layer}.log" 2>&1 &
            QWEN2VL_CTX_PIDS+=($!)
        fi
    done
    
    # Wait for all Contextual LLM Judge jobs
    if [ "$DRY_RUN" = false ] && [ ${#QWEN2VL_CTX_PIDS[@]} -gt 0 ]; then
        log "Waiting for ${#QWEN2VL_CTX_PIDS[@]} parallel LLM Judge Contextual jobs..."
        for pid in "${QWEN2VL_CTX_PIDS[@]}"; do
            wait $pid || log "WARNING: LLM Judge Contextual job $pid failed"
        done
        log "✓ Phase 11 complete"
    fi
fi

# ============================================================
# SUMMARY
# ============================================================
log ""
log "=========================================="
log "MASTER SCRIPT COMPLETED"
log "=========================================="
log "Log directory: $LOG_DIR"
log "Check $MASTER_LOG for full details"

echo ""
echo "=========================================="
echo "MASTER SCRIPT COMPLETED"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo ""
echo "Next steps:"
echo "1. Check logs for any failures: cat $MASTER_LOG | grep FAILED"
echo "2. Regenerate unified viewer if needed"
echo "=========================================="

