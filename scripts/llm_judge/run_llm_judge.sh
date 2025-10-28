#!/bin/bash

# Run LLM Judge Evaluation Script
# This script runs the LLM judge evaluation on image patches

# Default values
INPUT_JSON="test_data/test.json"
API_KEY=""
IMAGE_INDICES=""
PATCH_SIZE=28.0
BBOX_SIZE=3
NUM_SAMPLES=5
NUM_IMAGES=1
SHOW_IMAGES=false
SAVE_RESULTS="test_data/results.json"
MODEL="gpt-5"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_json)
            INPUT_JSON="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --image_indices)
            shift
            IMAGE_INDICES=""
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                IMAGE_INDICES="$IMAGE_INDICES $1"
                shift
            done
            ;;
        --patch_size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --bbox_size)
            BBOX_SIZE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --show_images)
            SHOW_IMAGES=true
            shift
            ;;
        --save_results)
            SAVE_RESULTS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --input_json <path> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --input_json <path>         Path to JSON file with image_path and patches"
            echo ""
            echo "Optional arguments:"
            echo "  --api_key <key>            OpenAI API key (default: OPENAI_API_KEY env var)"
            echo "  --image_indices <i1 i2>    Specific image indices to process (default: all)"
            echo "  --patch_size <size>        Size of each patch in pixels (default: 28.0)"
            echo "  --bbox_size <size>         Size of bounding box in patches (default: 3)"
            echo "  --num_samples <n>          Number of patches to sample per image (default: 36)"
            echo "  --num_images <n>           Number of images to process (default: all)"
            echo "  --show_images              Display images during processing"
            echo "  --save_results <path>      Path to save results JSON file"
            echo "  --model <name>             OpenAI model to use (default: gpt-5)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --input_json data.json --save_results results.json"
            echo "  $0 --input_json data.json --image_indices 0 1 2 --bbox_size 5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_JSON" ]; then
    echo "Error: --input_json is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input JSON file '$INPUT_JSON' does not exist"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "Error: OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable"
    exit 1
fi

# Build the Python command
CMD="python run_llm_judge.py --input_json \"$INPUT_JSON\""

if [ -n "$API_KEY" ]; then
    CMD="$CMD --api_key \"$API_KEY\""
fi

if [ -n "$IMAGE_INDICES" ]; then
    CMD="$CMD --image_indices$IMAGE_INDICES"
fi

CMD="$CMD --patch_size $PATCH_SIZE"
CMD="$CMD --bbox_size $BBOX_SIZE"
CMD="$CMD --num_samples $NUM_SAMPLES"

if [ -n "$NUM_IMAGES" ]; then
    CMD="$CMD --num_images $NUM_IMAGES"
fi

if [ "$SHOW_IMAGES" = true ]; then
    CMD="$CMD --show_images"
fi

if [ -n "$SAVE_RESULTS" ]; then
    CMD="$CMD --save_results \"$SAVE_RESULTS\""
fi

CMD="$CMD --model \"$MODEL\""

# Print configuration
echo "=========================================="
echo "LLM Judge Evaluation"
echo "=========================================="
echo "Input JSON: $INPUT_JSON"
echo "Model: $MODEL"
echo "Patch size: $PATCH_SIZE"
echo "Bounding box size: ${BBOX_SIZE}x${BBOX_SIZE}"
echo "Number of samples: $NUM_SAMPLES"
if [ -n "$NUM_IMAGES" ]; then
    echo "Number of images: $NUM_IMAGES"
fi
if [ -n "$IMAGE_INDICES" ]; then
    echo "Image indices:$IMAGE_INDICES"
fi
if [ -n "$SAVE_RESULTS" ]; then
    echo "Save results to: $SAVE_RESULTS"
fi
echo "Final command to be executed:"
echo "$CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD
