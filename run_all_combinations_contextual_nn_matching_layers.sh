#!/bin/bash

# VERSION THAT COMPARES MATCHING LAYERS
# This version compares visual tokens from layer N with text embeddings from layer N
# (e.g., layer 8 vision tokens vs layer 8 text embeddings)

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

echo "Running analysis for dataset: $DATASET ($([ "$DATASET" == "cc" ] && echo "Conceptual Captions" || echo "Visual Genome"))"
echo "MODE: Matching layers (visual_layer N vs contextual_layer N)"
echo ""

# Define the LLMs and vision encoders
LLMS=("llama3-8b" "qwen2-7b" "olmo-7b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

# Map LLMs to their contextual embedding directories
declare -A LLM_TO_CONTEXTUAL_DIR
LLM_TO_CONTEXTUAL_DIR["llama3-8b"]="meta-llama_Meta-Llama-3-8B"
LLM_TO_CONTEXTUAL_DIR["olmo-7b"]="allenai_OLMo-7B-1024-preview"
LLM_TO_CONTEXTUAL_DIR["qwen2-7b"]="Qwen_Qwen2-7B"

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
else
    CONTEXTUAL_BASE="molmo_data/contextual_llm_embeddings_vg"
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

# Set output directory based on dataset
if [ "$DATASET" == "cc" ]; then
    OUTPUT_DIR="analysis_results/contextual_nearest_neighbors"
else
    OUTPUT_DIR="analysis_results/contextual_nearest_neighbors_vg"
fi

echo "Contextual embeddings base: $CONTEXTUAL_BASE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Loop through all combinations
for llm in "${LLMS[@]}"; do
    # Get the contextual directory for this LLM
    contextual_subdir="${LLM_TO_CONTEXTUAL_DIR[$llm]}"
    contextual_dir="${CONTEXTUAL_BASE}/${contextual_subdir}"
    
    # Check if contextual directory exists
    if [ ! -d "$contextual_dir" ]; then
        echo "WARNING: Contextual directory not found: $contextual_dir"
        echo "Skipping all combinations for ${llm}"
        continue
    fi
    
    # Get available layers for this LLM
    available_layers=($(get_available_layers "$contextual_dir"))
    
    if [ ${#available_layers[@]} -eq 0 ]; then
        echo "WARNING: No layer directories found in: $contextual_dir"
        echo "Skipping all combinations for ${llm}"
        continue
    fi
    
    echo "=========================================="
    echo "LLM: ${llm}"
    echo "Contextual directory: ${contextual_dir}"
    echo "Available layers: ${available_layers[*]}"
    echo "=========================================="
    
    for vision_encoder in "${VISION_ENCODERS[@]}"; do
        # Special case: qwen2-7b with vit-l-14-336 uses seed10
        if [ "$llm" == "qwen2-7b" ] && [ "$vision_encoder" == "vit-l-14-336" ]; then
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}_seed10"
        else
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}"
        fi
        
        checkpoint_path="molmo_data/checkpoints/${checkpoint_name}/step12000-unsharded"
        
        # Check if checkpoint exists
        if [ ! -d "$checkpoint_path" ]; then
            echo "WARNING: Checkpoint not found: $checkpoint_path"
            echo "Skipping ${llm} + ${vision_encoder}"
            continue
        fi
        
        echo "------------------------------------------"
        echo "Processing: ${llm} + ${vision_encoder}"
        echo "Checkpoint: ${checkpoint_path}"
        echo "------------------------------------------"
        
        # Check which layers have caches available
        layers_with_cache=()
        for layer in "${available_layers[@]}"; do
            cache_file="${contextual_dir}/layer_${layer}/embeddings_cache.pt"
            if [ -f "$cache_file" ]; then
                layers_with_cache+=("$layer")
            else
                echo "SKIPPED: Cache not yet built for ${llm} layer ${layer}"
                echo "  Cache file: $cache_file"
            fi
        done
        
        if [ ${#layers_with_cache[@]} -eq 0 ]; then
            echo "WARNING: No layers with caches available for ${llm} + ${vision_encoder}"
            continue
        fi
        
        # NEW: Loop over each layer and compare matching layers
        # This compares visual_layer N with contextual_layer N
        for layer in "${layers_with_cache[@]}"; do
            echo "Running analysis for matching layers: visual${layer} vs contextual${layer}..."
            
            torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
                "$SCRIPT_PATH" \
                --ckpt-path "$checkpoint_path" \
                --contextual-dir "$contextual_dir" \
                --contextual-layer "$layer" \
                --visual-layer "$layer" \
                --num-images $NUM_IMAGES \
                --output-dir "$OUTPUT_DIR"
            
            # Check if command succeeded
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed for ${llm} + ${vision_encoder} at layer ${layer}"
            else
                echo "SUCCESS: Completed ${llm} + ${vision_encoder} at layer ${layer}"
            fi
            
            echo ""
        done
        
        echo ""
    done
    
    echo ""
done

echo "=========================================="
echo "All combinations processed!"
echo "=========================================="

