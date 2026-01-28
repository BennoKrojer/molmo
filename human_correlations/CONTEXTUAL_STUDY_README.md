# Sentence-Level (Contextual) Human Study

This guide explains how to generate data for the sentence-level interpretability study using contextual nearest neighbors.

## Overview

The token-level study (`interp_data_nn/`) shows humans 5 **individual tokens** (e.g., "cat", "green", "building").

The sentence-level study (`interp_data_contextual/`) shows humans 5 **sentences with highlighted tokens** (e.g., "A **cat** sat on the mat").

This allows us to study whether context matters for interpretability judgements.

## Generating the Data

### Step 1: Create Sentence-Level Data

```bash
cd human_correlations
python create_contextual_data_for_human_study.py
```

This will:
- Load the existing token-level study data (`interp_data_nn/data.json`)
- For each instance (image + patch):
  - Randomly select a layer from [1, 2, 4, 8, 16, 24, 30, 31]
  - Load the contextual nearest neighbors for that layer
  - Extract the top 5 sentences with highlighted tokens
- Output to `interp_data_contextual/data.json`

### Configuration

You can customize the generation:

```bash
python create_contextual_data_for_human_study.py \
    --token-data interp_data_nn/data.json \
    --contextual-dir ../analysis_results/contextual_nearest_neighbors \
    --output interp_data_contextual/data.json \
    --visual-layer 0 \
    --contextual-layers 1 2 4 8 16 24 30 31 \
    --seed 42
```

Options:
- `--token-data`: Path to token-level study data
- `--contextual-dir`: Directory with contextual nearest neighbors
- `--output`: Output file path
- `--visual-layer`: Visual layer to use (default: 0)
- `--contextual-layers`: List of contextual layers to randomly sample from
- `--seed`: Random seed for reproducible layer selection

## Data Format

The output format is similar to the token-level study, with additions:

```json
{
  "instance_id": "00139_patch0",
  "original_index": 139,
  "patch_index": 0,
  "patch_row": 2,
  "patch_col": 14,
  "image_url": "https://...",
  "caption": "...",
  "layer": 16,
  "visual_layer": 0,
  "candidates": [
    "A **cat** sat on the mat",
    "The **green** leaves rustled",
    "**Buildings** lined the street",
    "She wore a **red** dress",
    "The **sky** was blue"
  ],
  "token_candidates": ["cat", "green", "Buildings", "red", "sky"],
  "id": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded_00139_patch0",
  "model": "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded"
}
```

New fields:
- **`layer`**: The LLM layer used for this instance (randomly selected)
- **`visual_layer`**: The visual layer (0 = vision backbone output, higher = after N LLM transformer layers)
- **`sentence_candidates`**: Same as `candidates` - sentences with `**highlighted**` tokens
- **`token_candidates`**: The original token-level candidates (for reference)

## Token Highlighting Format

Tokens are highlighted using `**token**` syntax:
- `"A **cat** sat on the mat"` - the word "cat" should be highlighted
- `"The **green** leaves"` - the word "green" should be highlighted

Your human study interface should:
1. Parse the `**` markers
2. Display the highlighted word in a different color/style
3. Show the full sentence for context

## Running Human Study

Once the data is generated:

1. **Upload to annotation platform**: Use `interp_data_contextual/data.json`
2. **Instructions for annotators**:
   - "You will see an image with a red box highlighting a region"
   - "Below are 5 sentences with one word highlighted in each"
   - "Select any sentences where the highlighted word relates to the content in the red box"
   - "The highlighted word can relate in 3 ways:"
     - **Concrete**: Directly names what's visible (e.g., "cat", "building")
     - **Abstract**: Describes a concept/activity (e.g., "happiness", "running")
     - **Global**: Refers to broader image context (e.g., color, scene type)

3. **Collect results**: Save in the same format as token-level study

## Comparing Token vs Sentence Studies

After collecting both:

```python
# Analysis script (to be created)
import json

# Load both studies
with open('interp_data_nn/results/annotations.json') as f:
    token_results = json.load(f)

with open('interp_data_contextual/results/annotations.json') as f:
    sentence_results = json.load(f)

# Compare interpretability rates
# Do humans find more/fewer tokens interpretable when shown in context?
```

## Layer Selection Strategy

We randomly select layers to:
1. Get diverse representations across the LLM depth
2. Avoid biasing towards any particular layer
3. Test whether layer matters for interpretability

The default layers [1, 2, 4, 8, 16, 24, 30, 31] were chosen to:
- Sample early layers (1, 2, 4)
- Sample middle layers (8, 16)
- Sample late layers (24, 30, 31)

## Expected Differences

We hypothesize that context might:
- **Increase interpretability**: Seeing "The **cat** sat" is clearer than just "cat"
- **Decrease interpretability**: Context might make spurious matches more obvious
- **Affect relation types**: Same token might be "concrete" in one context, "abstract" in another

## File Structure

```
human_correlations/
├── create_contextual_data_for_human_study.py  # Generation script
├── CONTEXTUAL_STUDY_README.md                 # This file
├── interp_data_nn/                            # Token-level study
│   └── data.json
└── interp_data_contextual/                    # Sentence-level study (generated)
    ├── data.json
    └── results/                               # Human annotations (to be collected)
```

## Notes

- The same image patches and models are used in both studies for direct comparison
- Each instance gets a random layer assignment (reproducible with seed)
- Sentences come from real captions in the PixMo dataset
- Token highlighting position is determined by the contextual embedding metadata


