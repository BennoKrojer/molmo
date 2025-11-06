#!/bin/bash

# Parse command line arguments
DATASET=${1:-cc}  # Default to 'cc' if no argument provided

# Validate dataset argument
if [ "$DATASET" != "cc" ] && [ "$DATASET" != "vg" ]; then
    echo "Error: Invalid dataset argument. Use 'cc' or 'vg'"
    echo "Usage: $0 [cc|vg]"
    exit 1
fi

source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Running contextual NN analysis for ablations on dataset: $DATASET ($([ "$DATASET" == "cc" ] && echo "Conceptual Captions" || echo "Visual Genome"))"
echo ""

# Define ablation checkpoints
ABLATION_CHECKPOINTS=(
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10"
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11"
    "train_mlp-only_pixmo_points_resize_olmo-7b_vit-l-14-336"
)

# All ablations use olmo-7b, so we use the same contextual embeddings directory
CONTEXTUAL_SUBDIR="allenai_OLMo-7B-1024-preview"

# Contextual nearest neighbors parameters
NUM_IMAGES=300

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4
MASTER_PORT=29527

# Base script path
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/contextual_nearest_neighbors.py"

# Base path for contextual embeddings (depends on dataset)
if [ "$DATASET" == "cc" ]; then
    CONTEXTUAL_BASE="molmo_data/contextual_llm_embeddings"
    OUTPUT_BASE="analysis_results/contextual_nearest_neighbors"
else
    CONTEXTUAL_BASE="molmo_data/contextual_llm_embeddings_vg"
    OUTPUT_BASE="analysis_results/contextual_nearest_neighbors_vg"
fi

CONTEXTUAL_DIR="${CONTEXTUAL_BASE}/${CONTEXTUAL_SUBDIR}"

# Check if contextual directory exists
if [ ! -d "$CONTEXTUAL_DIR" ]; then
    echo "ERROR: Contextual directory not found: $CONTEXTUAL_DIR"
    echo "Cannot run contextual NN analysis for ablations."
    exit 1
fi

# Function to extract available layers from contextual directory
get_available_layers() {
    local contextual_dir=$1
    local layers=()
    
    # Find all layer_* directories and extract the layer numbers
    for dir in "$contextual_dir"/layer_*; do
        if [ -d "$dir" ]; then
            layer_num=$(basename "$dir" | sed 's/layer_//')
            layers+=("$layer_num")
        fi
    done
    
    # Sort layers numerically
    IFS=$'\n' layers=($(sort -n <<<"${layers[*]}"))
    unset IFS
    
    echo "${layers[@]}"
}

# Get available layers for olmo-7b
available_layers=($(get_available_layers "$CONTEXTUAL_DIR"))

if [ ${#available_layers[@]} -eq 0 ]; then
    echo "ERROR: No layer directories found in: $CONTEXTUAL_DIR"
    exit 1
fi

echo "=========================================="
echo "Contextual directory: ${CONTEXTUAL_DIR}"
echo "Available layers: ${available_layers[*]}"
echo "Output base: ${OUTPUT_BASE}/ablations/"
echo "=========================================="
echo ""

# Loop through all ablation checkpoints
for checkpoint_name in "${ABLATION_CHECKPOINTS[@]}"; do
    checkpoint_path="molmo_data/checkpoints/ablations/${checkpoint_name}/step12000-unsharded"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint_path"
        echo "Skipping ${checkpoint_name}"
        continue
    fi
    
    echo "=========================================="
    echo "Processing ablation: ${checkpoint_name}"
    echo "Checkpoint: ${checkpoint_path}"
    echo "=========================================="
    
    # Check which layers have caches available
    layers_with_cache=()
    for layer in "${available_layers[@]}"; do
        cache_file="${CONTEXTUAL_DIR}/layer_${layer}/embeddings_cache.pt"
        if [ -f "$cache_file" ]; then
            layers_with_cache+=("$layer")
        else
            echo "SKIPPED: Cache not yet built for olmo-7b layer ${layer}"
            echo "  Cache file: $cache_file"
        fi
    done
    
    if [ ${#layers_with_cache[@]} -eq 0 ]; then
        echo "WARNING: No layers with caches available for ${checkpoint_name}"
        continue
    fi
    
    # Convert array to comma-separated string
    layers_str=$(IFS=,; echo "${layers_with_cache[*]}")
    
    echo "Running analysis for contextual layers: ${layers_str} (model loaded once)..."
    
    torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
        "$SCRIPT_PATH" \
        --ckpt-path "$checkpoint_path" \
        --contextual-dir "$CONTEXTUAL_DIR" \
        --contextual-layer "$layers_str" \
        --num-images $NUM_IMAGES \
        --output-dir "${OUTPUT_BASE}/ablations"
    
    # Check if command succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for ${checkpoint_name}"
    else
        echo "SUCCESS: Completed ${checkpoint_name}"
    fi
    
    echo ""
done

echo "=========================================="
echo "All ablations processed!"
echo "=========================================="

