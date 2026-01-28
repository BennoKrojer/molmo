#!/bin/bash

source ../../env/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define the LLMs and vision encoders
# LLMS=("llama3-8b" "olmo-7b" "qwen2-7b")
# VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")
LLMS=("olmo-7b")
VISION_ENCODERS=("vit-l-14-336")

# Evaluation settings
SPLIT="validation"
MAX_IMAGES=300
EVAL_SCRIPT="eval_captioning_gpt-judge.py"
CAPTIONING_RESULTS_BASE="analysis_results/captioning_evaluation"
CAPTIONS_BASE_DIR="analysis_results/captions"

# Function to run evaluation on a JSON file
run_eval() {
    local json_path="$1"
    local model_name="$2"
    local stem=$(basename "$json_path")
    
    # Get absolute paths
    local project_root=$(pwd)
    local abs_json_path="${project_root}/${json_path}"
    local abs_eval_script="${project_root}/${EVAL_SCRIPT}"
    
    # Create dedicated output directory for this model
    local captioning_dir="${CAPTIONING_RESULTS_BASE}/${model_name}_step12000-unsharded"
    mkdir -p "$captioning_dir"
    
    # Output file will be in the captioning evaluation directory
    local out_file="$captioning_dir/${stem%.*}_llm_judge_${SPLIT}.json"
    
    # Check if JSON file exists
    if [ ! -f "$json_path" ]; then
        echo "SKIPPED: JSON file not found: $json_path"
        return
    fi
    
    # Check if output already exists and is complete
    if [ -f "$out_file" ]; then
        echo "SKIPPED: Output already exists: $out_file"
        echo "  (Re-run with manual deletion if you want to regenerate)"
        return
    fi
    
    echo "Running evaluation for: $json_path"
    echo "  Output directory: $captioning_dir"
    
    # Run evaluation on the captions JSON, but save output in captioning directory
    cd "$captioning_dir" || exit 1
    python3 "$abs_eval_script" \
        --results_file "$abs_json_path" \
        --split "$SPLIT" \
        --max-images $MAX_IMAGES \
        --fallback-dataset-images \
        --resume \
        --create-visualizations
    local exit_code=$?
    cd - > /dev/null || exit 1
    
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: Completed evaluation"
        echo "  Output: $out_file"
    else
        echo "ERROR: Failed evaluation for $json_path (exit code: $exit_code)"
    fi
}

echo "=========================================="
echo "Running Captioning Evaluation on All Model Combinations"
echo "Reading from: ${CAPTIONS_BASE_DIR}/"
echo "Evaluating up to ${MAX_IMAGES} images per model"
echo "=========================================="

# Loop through all combinations
for llm in "${LLMS[@]}"; do
    for vision_encoder in "${VISION_ENCODERS[@]}"; do
        # Special case: qwen2-7b with vit-l-14-336 uses seed10
        if [ "$llm" == "qwen2-7b" ] && [ "$vision_encoder" == "vit-l-14-336" ]; then
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}_seed10"
        else
            checkpoint_name="train_mlp-only_pixmo_cap_resize_${llm}_${vision_encoder}"
        fi
        
        captions_dir="${CAPTIONS_BASE_DIR}/${checkpoint_name}_step12000-unsharded"
        
        # Check if captions directory exists
        if [ ! -d "$captions_dir" ]; then
            echo "WARNING: Captions directory not found: $captions_dir"
            echo "Skipping ${llm} + ${vision_encoder}"
            continue
        fi
        
        echo ""
        echo "------------------------------------------"
        echo "Processing: ${llm} + ${vision_encoder}"
        echo "Directory: ${captions_dir}"
        echo "------------------------------------------"
        
        # Look for generated_captions.json file
        captions_json="${captions_dir}/generated_captions.json"
        
        if [ -f "$captions_json" ]; then
            echo "Found captions JSON: $(basename $captions_json)"
            run_eval "$captions_json" "$checkpoint_name"
        else
            echo "WARNING: Captions JSON not found"
            echo "  Expected: $captions_json"
            echo "  Run caption generation first"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "All combinations processed!"
echo "=========================================="
echo ""
echo "Captioning evaluation results are organized in:"
echo "  ${CAPTIONING_RESULTS_BASE}/"
echo ""
echo "For each model combination you'll find:"
echo "  - Evaluation output: generated_captions_llm_judge_${SPLIT}.json"
echo "  - Visualizations: interactive_visualizations_*/"
echo ""
echo "Input JSONs are in:"
echo "  ${CAPTIONS_BASE_DIR}/{checkpoint_name}_step12000-unsharded/generated_captions.json"
echo ""

