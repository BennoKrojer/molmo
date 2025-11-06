# Quick Start: LLM Judge Evaluation with Visualizations

## Overview

This folder contains tools to evaluate vision-language model interpretability using GPT-5 as a judge, with comprehensive visualizations for quick inspection.

## Files

### Main Scripts (2 total)

1. **`run_single_model_with_viz.py`** - Python script that runs evaluation for ONE model combination and creates visualizations
2. **`run_all_parallel.sh`** - Bash script that runs all 9 combinations (3 LLMs × 3 vision encoders) in parallel

### Other Files
- `prompts.py` - GPT prompt template
- `utils.py` - Image processing utilities
- `run_llm_judge_pixmo.py` - Original script (still works for custom runs)
- `visualise.ipynb` - Jupyter notebook for manual inspection

## Quick Usage

### Run All Model Combinations in Parallel

```bash
cd /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo
./llm_judge/run_all_parallel.sh
```

This will:
- Process 5 images per model from the **validation split**
- Sample 1 patch per image
- Run 9 combinations sequentially (total: 45 API calls)
- Save results and visualizations to `analysis_results/llm_judge_interpretability/`

**Note**: The script uses the `validation` split because the JSON files from `run_all_combinations_nn.sh` only contain data in the validation split (train split is empty).

### Run a Single Model

```bash
source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 llm_judge/run_single_model_with_viz.py \
    --llm olmo-7b \
    --vision-encoder vit-l-14-336 \
    --api-key $(cat llm_judge/api_key.txt) \
    --num-images 5 \
    --num-samples 1
```

## Output Structure

After running, you'll get:

```
analysis_results/llm_judge_interpretability/
├── llm_judge_olmo-7b_vit-l-14-336/
│   ├── results_train.json          # Evaluation results with accuracy
│   └── visualizations/              # Image files for inspection
│       ├── image_0000_patch_5_7_pass.jpg
│       ├── image_0001_patch_3_4_fail.jpg
│       └── ...
├── llm_judge_olmo-7b_dinov2-large-336/
│   └── ...
└── log_olmo-7b_vit-l-14-336.txt     # Execution logs
```

## Visualization Format

Each visualization image shows:
- **Left side**: Image with red bounding box highlighting the evaluated patch
- **Right side**:
  - Result (PASS/FAIL with color coding)
  - Candidate tokens that were evaluated
  - **Concrete words** (green) - literally visible in the patch (objects, colors, textures, text)
  - **Abstract words** (blue) - broader concepts/emotions/activities related to the patch
  - **Global words** (orange) - present elsewhere in the image but not in the patch
  - GPT's reasoning
  - Ground truth caption (for context)

## Customization

### Change number of images/patches:

Edit variables in `run_all_parallel.sh`:
```bash
NUM_IMAGES=5      # Images per model
NUM_SAMPLES=1     # Patches per image
SEED=42           # Random seed for reproducibility
```

### Run specific combinations:

Modify the arrays in `run_all_parallel.sh`:
```bash
LLMS=("olmo-7b")                    # Only one LLM
VISION_ENCODERS=("vit-l-14-336")    # Only one encoder
```

### Change prompt:

Edit `prompts.py` to modify the GPT evaluation prompt.

**Note**: The current prompt distinguishes between three types of relationships:
- **Concrete**: Literally visible in the patch (objects, colors, textures, text)
- **Abstract**: Broader concepts/emotions/activities related to the patch
- **Global**: Present elsewhere in the image but not in the highlighted patch

A patch is considered interpretable if it has any type of related words.

### Use cropped region mode:

Add `--use-cropped-region` flag to pass both the full image and a cropped region to the LLM judge. This gives the LLM more focused visual information about the specific patch.

```bash
python3 llm_judge/run_single_model_with_viz.py \
    --llm olmo-7b \
    --vision-encoder vit-l-14-336 \
    --api-key $(cat llm_judge/api_key.txt) \
    --use-cropped-region \
    --num-images 5 \
    --num-samples 1
```

**Note**: When using cropped region mode, output directories will have `_cropped` suffix (e.g., `llm_judge_olmo-7b_vit-l-14-336_cropped`).

### Set random seed for reproducibility:

Use `--seed` to ensure reproducible results across runs:

```bash
python3 llm_judge/run_single_model_with_viz.py \
    --llm olmo-7b \
    --vision-encoder vit-l-14-336 \
    --api-key $(cat llm_judge/api_key.txt) \
    --seed 42 \
    --num-images 5 \
    --num-samples 1
```

**Note**: The same seed will produce identical patch sampling and evaluation order.

## Quick Inspection Workflow

1. Run the evaluation:
   ```bash
   ./llm_judge/run_all_parallel.sh
   ```

2. Check accuracy in results JSON:
   ```bash
   grep -A 2 '"accuracy"' analysis_results/llm_judge_interpretability/*/results_train.json
   ```

3. Open visualizations to inspect:
   ```bash
   # View all images for one model
   ls analysis_results/llm_judge_interpretability/llm_judge_olmo-7b_vit-l-14-336/visualizations/
   
   # Or copy to local machine for viewing:
   scp -r user@server:path/to/visualizations/ ./local_folder/
   ```

## Logit Lens Evaluation

For evaluating logit lens results instead of nearest neighbors:

1. Run logit lens evaluation:
   ```bash
   bash llm_judge/run_all_parallel_logitlens.sh
   ```

2. Single model with logit lens:
   ```bash
   python3 llm_judge/run_single_model_with_viz_logitlens.py \
       --llm olmo-7b \
       --vision-encoder vit-l-14-336 \
       --api-key $(cat llm_judge/api_key.txt) \
       --layer layer0 \
       --use-cropped-region \
       --num-images 5 \
       --num-samples 1
   ```

**Note**: Logit lens results are saved to `analysis_results/llm_judge_logitlens/` with layer-specific naming (e.g., `llm_judge_olmo-7b_vit-l-14-336_layer0_cropped`, `llm_judge_olmo-7b_vit-l-14-336_layer12_cropped`, etc.). The script runs **models in parallel** but **layers sequentially** for each model to manage resource usage efficiently.

## Visualization: Heatmaps

Create a heatmap summarizing interpretability results across all model combinations:

```bash
python3 llm_judge/create_heatmap_nn.py
```

This generates a heatmap with:
- **Rows**: LLMs
- **Columns**: Vision encoders
- **Cells**: Interpretability percentage (color-coded)
- **Output**: `analysis_results/llm_judge_nearest_neighbors/heatmap_interpretability.pdf` (and PNG)

4. Modify prompt in `prompts.py` and re-run quickly on fewer examples:
   ```bash
   # Test on 2 images with 1 patch each
   python3 llm_judge/run_single_model_with_viz.py \
       --llm olmo-7b \
       --vision-encoder vit-l-14-336 \
       --api-key $(cat llm_judge/api_key.txt) \
       --num-images 2 \
       --num-samples 1
   ```

## Tips

- **Parallel execution**: The bash script runs all combinations simultaneously, so it's fast (~2-3 minutes for 45 API calls)
- **Logs**: Check `log_*.txt` files if something fails
- **Resume**: If a run fails, just re-run the specific model combination
- **Visualization naming**: `_pass.jpg` = interpretable, `_fail.jpg` = not interpretable

