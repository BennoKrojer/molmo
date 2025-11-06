#!/bin/bash
#
# Monitor progress of parallel LLM judge evaluation
#

# Configuration
MODE="${1:-nn}"  # Default to nearest neighbors (nn), or use "logitlens" or "contextual"
SPLIT="${3:-validation}"  # Default to validation split

# Set expected images based on mode (can be overridden by second arg)
if [ -n "$2" ]; then
    EXPECTED_IMAGES="$2"
elif [ "$MODE" = "contextual" ]; then
    EXPECTED_IMAGES=100  # Default for contextual
else
    EXPECTED_IMAGES=300  # Default for nn/logitlens
fi

# Set output base based on mode
if [ "$MODE" = "logitlens" ]; then
    OUTPUT_BASE="analysis_results/llm_judge_logitlens"
    MODE_NAME="Logit Lens"
    LAYER_PREFIX="layer"
elif [ "$MODE" = "contextual" ]; then
    OUTPUT_BASE="analysis_results/llm_judge_contextual_nn"
    MODE_NAME="Contextual NN"
    LAYER_PREFIX="contextual"
else
    OUTPUT_BASE="analysis_results/llm_judge_nearest_neighbors"
    MODE_NAME="Nearest Neighbors"
    LAYER_PREFIX="layer"
fi

echo "=========================================="
echo "LLM Judge Progress Monitor ($MODE_NAME)"
echo "=========================================="
echo "Expected images per model-layer: $EXPECTED_IMAGES"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_BASE"
echo ""

# Model combinations
LLMS=("olmo-7b" "qwen2-7b" "llama3-8b")
VISION_ENCODERS=("vit-l-14-336" "dinov2-large-336" "siglip")

# Layers depend on mode
if [ "$MODE" = "contextual" ]; then
    # Common contextual layers (may vary by model)
    LAYERS=(1 2 4 8 16 24 26 27 30 31)
else
    # Regular layers for nn/logitlens
    LAYERS=(1 2 3 4 8 12 16 20 24 28 32)
fi

# Calculate total expected combinations
total_combinations=$((${#LLMS[@]} * ${#VISION_ENCODERS[@]} * ${#LAYERS[@]}))

while true; do
    clear
    echo "=========================================="
    echo "LLM Judge Progress Monitor ($MODE_NAME)"
    echo "$(date)"
    echo "=========================================="
    echo ""
    
    completed=0
    in_progress=0
    not_started=0
    
    # Group by model
    for llm in "${LLMS[@]}"; do
        for encoder in "${VISION_ENCODERS[@]}"; do
            # Special case for qwen2-7b + vit-l-14-336
            if [ "$llm" = "qwen2-7b" ] && [ "$encoder" = "vit-l-14-336" ]; then
                model_base="llm_judge_${llm}_${encoder}_seed10"
            else
                model_base="llm_judge_${llm}_${encoder}"
            fi
            
            echo "📊 $llm + $encoder:"
            
            # Check each layer
            layer_completed=0
            layer_in_progress=0
            layer_not_started=0
            
            for layer in "${LAYERS[@]}"; do
                # Find matching directory (handles different model suffixes like gpt5, gemini, etc)
                pattern="${OUTPUT_BASE}/${model_base}_${LAYER_PREFIX}${layer}_*"
                found_dir=""
                
                for dir in $pattern; do
                    if [ -d "$dir" ]; then
                        found_dir="$dir"
                        break
                    fi
                done
                
                if [ -n "$found_dir" ]; then
                    results_file="$found_dir/results_${SPLIT}.json"
                    
                    if [ -f "$results_file" ]; then
                        # Try different fields for total count
                        total=$(grep -o '"total": [0-9]*' "$results_file" | tail -1 | grep -o '[0-9]*')
                        if [ -z "$total" ]; then
                            # Try num_images_processed (used by contextual)
                            total=$(grep -o '"num_images_processed": [0-9]*' "$results_file" | tail -1 | grep -o '[0-9]*')
                        fi
                        if [ -z "$total" ]; then
                            # Try counting results array (fallback)
                            total=$(grep -o '"results":\s*\[' "$results_file" > /dev/null && python3 -c "import json; d=json.load(open('$results_file')); print(len(d.get('results', [])))" 2>/dev/null || echo "")
                        fi
                        
                        if [ -n "$total" ] && [ "$total" -gt 0 ]; then
                            if [ "$total" -ge "$EXPECTED_IMAGES" ]; then
                                # Try different fields for accuracy/correct
                                correct=$(grep -o '"correct": [0-9]*' "$results_file" | tail -1 | grep -o '[0-9]*')
                                if [ -z "$correct" ]; then
                                    # For contextual, count interpretable results
                                    correct=$(python3 -c "import json; d=json.load(open('$results_file')); results=d.get('results', []); print(sum(1 for r in results if r.get('interpretable', False)))" 2>/dev/null || echo "0")
                                fi
                                accuracy=$(grep -o '"accuracy": [0-9.]*' "$results_file" | tail -1 | grep -o '[0-9.]*')
                                if [ -n "$accuracy" ]; then
                                    echo "  ✓ Layer $layer: $correct/$EXPECTED_IMAGES (${accuracy}%)"
                                else
                                    echo "  ✓ Layer $layer: $correct/$EXPECTED_IMAGES"
                                fi
                                ((completed++))
                                ((layer_completed++))
                            else
                                echo "  ⏳ Layer $layer: $total/$EXPECTED_IMAGES patches"
                                ((in_progress++))
                                ((layer_in_progress++))
                            fi
                        else
                            echo "  ⏳ Layer $layer: Starting..."
                            ((in_progress++))
                            ((layer_in_progress++))
                        fi
                    else
                        echo "  ⏳ Layer $layer: Starting..."
                        ((not_started++))
                        ((layer_not_started++))
                    fi
                else
                    echo "  ⚪ Layer $layer: Not started"
                    ((not_started++))
                    ((layer_not_started++))
                fi
            done
            
            echo "    Summary: ✓$layer_completed ⏳$layer_in_progress ⚪$layer_not_started / ${#LAYERS[@]} layers"
            echo ""
        done
    done
    
    echo "=========================================="
    echo "Overall Progress: $completed/$total_combinations completed"
    echo "  ✓ Completed:    $completed"
    echo "  ⏳ In Progress:  $in_progress"
    echo "  ⚪ Not Started:  $not_started"
    echo "=========================================="
    
    if [ $completed -eq $total_combinations ]; then
        echo ""
        echo "🎉 All model-layer combinations completed!"
        break
    fi
    
    echo ""
    echo "Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done

