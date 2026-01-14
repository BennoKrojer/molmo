# Human-LLM Judge Correlation Analysis

This directory contains scripts to evaluate human-LLM judge agreement on vision token interpretability.

## Problem Discovered

The original analysis revealed that the LLM judge was run on **different images** than the human study:
- **Human study**: Images from URLs (e.g., `blogger.googleusercontent.com`) 
- **Original LLM judge**: Validation set images from local storage
- This caused the low correlation (Cohen's κ = 0.168) because we were comparing different patches!

## Solution

Run the LLM judge on the **exact same patches** that humans evaluated.

## Files

### Scripts
- `run_llm_judge.sh` - Run LLM judge on NN (token-level) human study patches
- `run_llm_judge_contextual.sh` - Run LLM judge on contextual human study patches
- `run_llm_judge_on_human_study.py` - Python script that evaluates human study instances (supports both data types)
- `compute_correlations.py` - Compute correlation metrics between human and LLM judgements
- `visualize_agreement.py` - Create visualizations showing agreement/disagreement

### Data
- `interp_data_nn/data.json` - NN (token-level) human study data with 360 instances
- `interp_data_nn/results/` - Human judgement results from multiple annotators
- `interp_data_contextual/data.json` - Contextual human study data with 360 instances (sentence+token candidates)

## Usage

### Step 1: Run LLM Judge on Human Study Patches

**For NN (token-level) data:**
```bash
cd human_correlations
./run_llm_judge.sh
```

**For Contextual data:**
```bash
cd human_correlations
./run_llm_judge_contextual.sh
```

Both scripts will:
- Load 360 instances from the respective data file
- For each instance:
  - Download the image from the URL
  - Create a bounding box for the patch
  - Pass the image + 5 candidate words to GPT-5
  - Get LLM interpretability judgement
- Save results to `llm_judge_results/` or `llm_judge_results_contextual/`
- Support resume (will continue from where it left off)

For contextual data, candidates are `[sentence, token]` tuples. The script extracts the full word containing each token (e.g., `"autom"` → `"automobile"`) before passing to the LLM judge, using the same `extract_full_word_from_token()` function as `llm_judge/run_single_model_with_viz_contextual.py`.

**Note**: Each run makes ~360 API calls to GPT-5. Estimated cost: ~$10-20

### Step 2: Compute Correlations

Once the LLM judge has evaluated all instances:

**For NN data:**
```bash
python compute_correlations.py \
    --llm-results-file llm_judge_results/human_study_llm_results.json
```

**For Contextual data:**
```bash
python compute_correlations.py \
    --llm-results-file llm_judge_results_contextual/human_study_llm_results.json
```

This will output to `correlation_results.json` by default.

This computes:
- **Pearson r**: Linear correlation
- **Spearman ρ**: Rank correlation  
- **Cohen's κ**: Inter-rater agreement
- **Accuracy**: Simple agreement percentage

Per-model breakdowns for all 9 model combinations.

### Step 3: Visualize Examples

Create visualizations showing agreement/disagreement:

```bash
python visualize_agreement.py \
    --sample-type disagree \
    --num-examples 30 \
    --output-dir visualizations
```

Sample types:
- `disagree`: Show all disagreements
- `agree`: Show agreements
- `human_yes_llm_no`: Humans say interpretable, LLM says not
- `human_no_llm_yes`: LLM says interpretable, humans say not
- `all`: Show all examples

## Data Format

### NN Human Study Data (`interp_data_nn/data.json`)

Each instance contains:
```json
{
  "id": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_00139_patch0",
  "index": 139,
  "image_url": "https://...",
  "caption": "...",
  "candidates": ["word1", "word2", "word3", "word4", "word5"],
  "patch_row": 2,
  "patch_col": 14,
  "patch_type": "segmented",
  "model": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded"
}
```

### Contextual Human Study Data (`interp_data_contextual/data.json`)

Each instance contains:
```json
{
  "id": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_00139_patch0",
  "index": 139,
  "image_url": "https://...",
  "caption": "...",
  "candidates": [
    ["automobile making a left turn", "autom"],
    ["the car is red", "car"],
    ...
  ],
  "layer": 16,
  "visual_layer": 0,
  "original_token_candidates": ["word1", "word2", ...],
  "patch_row": 2,
  "patch_col": 14,
  "model": "..."
}
```

The contextual `candidates` are `[sentence, token]` tuples. The token is extracted to a full word (e.g., `"autom"` → `"automobile"`) before evaluation.

### Human Judgements (`interp_data_nn/results/`)

Each file contains judgements from one annotator:
```json
{
  "userId": "Benno",
  "results": [
    {
      "instanceId": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_00139_patch0",
      "selectedWords": [
        {"word": "Green", "relation": "concrete"},
        {"word": " colors", "relation": "abstract"}
      ],
      "noneSelected": false
    }
  ]
}
```

### LLM Judge Results (`llm_judge_results/human_study_llm_results.json`)

```json
{
  "total_instances": 360,
  "evaluated": 360,
  "results": [
    {
      "instance_id": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_00139_patch0",
      "candidates": ["word1", "word2", "word3", "word4", "word5"],
      "gpt_response": {
        "interpretable": true,
        "concrete_words": ["word1"],
        "abstract_words": ["word2"],
        "global_words": [],
        "reasoning": "..."
      }
    }
  ]
}
```

## Interpretability Definition

A patch is considered **interpretable** if:
- At least 1 of the 5 nearest neighbor tokens relates to the visual content in the patch
- This can be:
  - **Concrete**: Directly names what's visible (e.g., "cat", "building")
  - **Abstract**: Describes a concept/activity (e.g., "happiness", "running")
  - **Global**: Refers to broader image context (e.g., color names, scene types)

## Notes

- The human study used 360 instances across 9 model combinations
- Each instance shows a patch with its 5 nearest neighbor tokens
- Multiple annotators judged each instance (3 annotators: Benno, shravan, Vaibhav, Marius)
- The LLM judge uses the same prompt and criteria as humans

## Expected Results

After running the correct LLM judge on the same patches, we should see:
- Higher correlation (hopefully κ > 0.4 for "moderate" agreement)
- More meaningful agreement patterns
- Clear visualizations of what the LLM/humans agree/disagree on

