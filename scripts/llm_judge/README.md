# LLM Judge for Image Patch Interpretability

This tool evaluates whether nearest neighbor tokens from vision-language model's connector module are semantically meaningful for specific image regions using LLM judges.

## Files

- **`run_llm_judge.py`** - Main script to run the evaluation
- **`utils.py`** - Image processing and sampling utilities
- **`prompts.py`** - LLM prompt template for interpretability evaluation
- **`visualise.ipynb`** - Jupyter notebook for analyzing results
- **`test.json`** - Example input data file

## How to Run

```bash
python run_llm_judge.py --input_json test.json --api_key YOUR_OPENAI_API_KEY
```

### Optional Arguments

```bash
python run_llm_judge.py \
    --input_json test.json \
    --api_key YOUR_OPENAI_API_KEY \
    --image_indices 0 1 2 \           # Process only specific images
    --num_samples 36 \                # Number of patches to sample per image
    --bbox_size 3 \                   # Size of bounding box (3x3 patches)
    --save_results results.json \     # Save output to file
    --show_images                     # Display images during processing
```

## Input File Format

The input JSON file should be a list of dictionaries with the following structure:

```json
[
  {
    "image_path": "/path/to/your/image.jpg",
    "patches": [
      {
        "patch_row": 0,
        "patch_col": 0,
        "nearest_neighbors": [
          {
            "token": "example_word",
            "similarity": 0.85
          },
          {
            "token": "another_word",
            "similarity": 0.78
          }
        ]
      },
      {
        "patch_row": 0,
        "patch_col": 1,
        "nearest_neighbors": [...]
      }
    ]
  },
  {
    "image_path": "/path/to/another/image.jpg",
    "patches": [...]
  }
]
```

### Required Fields

- **`image_path`** - Full path to the image file
- **`patches`** - List of patch data with:
  - **`patch_row`**, **`patch_col`** - Patch coordinates (0-23 for 24x24 grid)
  - **`nearest_neighbors`** - List of tokens with similarity scores

## What It Does

1. Loads images and processes them to 512×512 pixels
2. Randomly samples valid patch regions (avoiding padded areas) (default: 36)
3. For each sampled region, creates a 3×3 bounding box and extracts tokens from the center patch
4. Sends the image with highlighted region + candidate tokens to GPT
5. GPT evaluates if the tokens are semantically relevant to the highlighted region
6. Returns accuracy metrics and detailed results