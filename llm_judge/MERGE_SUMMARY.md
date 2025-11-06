# LLM Judge Updates Merged

## Changes from `origin/judge` branch

I've successfully merged the LLM judge improvements from your colleague's branch. Here's what was updated:

### ✅ Updated Prompt (`prompts.py`)

**New Features:**
- **Clearer guidelines** distinguishing between direct and indirect relationships
- **Better examples** for text-based connections
- **New JSON format** with separate fields for `directly_related_words` and `indirectly_related_words`

**Key Improvements:**
- More precise definitions of what constitutes "direct" vs "indirect" relationships
- Better handling of text regions with specific examples
- Clearer instructions for conceptual connections

### ✅ Updated Code (`run_single_model_with_viz.py`)

**New Response Handling:**
- Parses `directly_related_words` and `indirectly_related_words` separately
- A patch is considered interpretable if it has **either** type of related words
- Updated accuracy calculation to use the new format

**Enhanced Visualizations:**
- **Green text** for directly related words (objects/attributes in the patch)
- **Orange text** for indirectly related words (conceptual/contextual connections)
- Better visual distinction between relationship types

### ✅ Updated Documentation (`QUICK_START.md`)

- Updated visualization format description
- Added explanation of the new prompt structure
- Clarified the interpretability criteria

## How to Use

The scripts work exactly the same way:

```bash
# Run all combinations in parallel
./llm_judge/run_all_parallel.sh

# Run single model
python3 llm_judge/run_single_model_with_viz.py \
    --llm olmo-7b \
    --vision-encoder vit-l-14-336 \
    --api-key $(cat llm_judge/api_key.txt) \
    --num-images 5 \
    --num-samples 1
```

## What's Different

1. **More nuanced evaluation**: GPT now distinguishes between direct and indirect relationships
2. **Better visualizations**: Color-coded words show relationship types
3. **Improved accuracy**: More precise interpretability assessment

The new prompt should give you better insights into how the vision-language models are connecting patches to tokens!

