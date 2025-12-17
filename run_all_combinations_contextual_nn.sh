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

echo "Running analysis for dataset: $DATASET ($([ "$DATASET" == "cc" ] && echo "Conceptual Captions" || echo "Visual Genome"))"
echo ""

# Define the LLMs and vision encoders
# LLMS=("llama3-8b" "olmo-7b" "qwen2-7b")
LLMS=("olmo-7b" "llama3-8b" "qwen2-7b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

# Map LLMs to their contextual embedding directories
declare -A LLM_TO_CONTEXTUAL_DIR
LLM_TO_CONTEXTUAL_DIR["llama3-8b"]="meta-llama_Meta-Llama-3-8B"
LLM_TO_CONTEXTUAL_DIR["olmo-7b"]="allenai_OLMo-7B-1024-preview"
LLM_TO_CONTEXTUAL_DIR["qwen2-7b"]="Qwen_Qwen2-7B"

# Contextual nearest neighbors parameters
# Using 304 instead of 300 because 304 is divisible by 8 GPUs (304/8 = 38)
NUM_IMAGES=304

# CUDA settings - 8 GPUs for faster processing
# (The slower script loads one layer at a time, so no OOM risk)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8  # Each GPU processes fewer images = faster overall
MASTER_PORT=29527

# Base script path - using SLOWER but CORRECT version (no distributed sharding bugs)
SCRIPT_PATH="/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/contextual_nearest_neighbors_allLayers_slower.py"

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
        
        # Convert layers array to comma-separated string
        # Process all visual layers in ONE call to avoid reloading checkpoint multiple times
        # The allLayers script will automatically compare each visual layer to ALL contextual layers
        layers_str=$(IFS=,; echo "${layers_with_cache[*]}")
        
        echo "Running analysis for ALL layers (model loaded once):"
        echo "  Visual layers to process: ${layers_str}"
        echo "  Contextual layers available: ${layers_with_cache[*]}"
        echo "  → Each visual layer will be compared to ALL contextual layers"
        echo "  Using SLOWER but CORRECT version (sequential layer iteration, no sharding)"
        
        torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
            "$SCRIPT_PATH" \
            --ckpt-path "$checkpoint_path" \
            --contextual-dir "$contextual_dir" \
            --visual-layer "$layers_str" \
            --num-images $NUM_IMAGES \
            --output-dir "$OUTPUT_DIR"
        
        # Check if command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for ${llm} + ${vision_encoder}"
        else
            echo "SUCCESS: Completed ${llm} + ${vision_encoder} for all layers"
        fi
        
        echo ""
    done
    
    echo ""
done

echo "=========================================="
echo "All combinations processed!"
echo "=========================================="

