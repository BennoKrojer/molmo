# LatentLens Similarity Data for Human Study

## Files

### `human_study_similarities_all_visual_layers.json` (Recommended)

Contains cosine similarities across **all visual layers** (0, 1, 2, 4, 8, 16, 24, 30, 31 for OLMo/LLaMA; 0, 1, 2, 4, 8, 16, 24, 26, 27 for Qwen2). This is the comprehensive version.

### `human_study_similarities_contextual.json` (Legacy)

Contains similarities for **visual_layer=0 only**. This matches what was used in the original human study (which used visual_layer=0 due to an early limitation in the data generation pipeline).

## Data Structure

```json
{
  "description": "Cosine similarities for human-annotated instances (LatentLens/contextual)",
  "total_instances": 360,
  "instances": [
    {
      "instance_id": "model_imageIdx_patchN",
      "model": "train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_step12000-unsharded",
      "image_idx": 139,
      "patch_row": 2,
      "patch_col": 14,
      "layer": 2,              // LLM layer used for contextual embeddings
      "visual_layer": 0,       // Vision layer (0=backbone output, higher=after N LLM layers)
      "image_url": "https://...",
      "caption": "Image description...",
      "neighbors": [
        {
          "rank": 1,
          "token_str": " colors",           // The token/word found as nearest neighbor
          "similarity": 0.1736,             // Cosine similarity score
          "source_caption": "the umbrella has rainbow colors",  // Caption containing this word
          "position": 4                     // Token position in caption
        },
        // ... 4 more neighbors (top-5 total)
      ]
    }
  ]
}
```

## Key Fields

- **instance_id**: Unique identifier for each annotated patch
- **layer**: The LLM layer (1, 2, 4, 8, 16, 24, 26, 27, 30, 31) from which contextual embeddings were extracted
- **similarity**: Cosine similarity between the vision token embedding and the contextual text embedding (range: typically 0.1 - 0.3)
- **neighbors**: Top-5 nearest neighbor candidates with their similarity scores

## Models Covered

9 model combinations (3 LLMs x 3 vision encoders):
- LLMs: OLMo-7B, LLaMA3-8B, Qwen2-7B
- Vision encoders: ViT-L-14-336, DINOv2-Large-336, SigLIP

## Human Annotations

The 360 instances were annotated by the paper authors. Each instance has:
- 6 random patches sampled from each of 60 images
- Annotations indicating which nearest neighbor candidates (if any) correctly describe the patch content

## Usage

```python
import json

with open('human_study_similarities_contextual.json') as f:
    data = json.load(f)

for instance in data['instances']:
    print(f"Image {instance['image_idx']}, Patch ({instance['patch_row']}, {instance['patch_col']})")
    print(f"Layer: {instance['layer']}")
    for n in instance['neighbors']:
        print(f"  {n['rank']}. '{n['token_str']}' (sim={n['similarity']:.4f})")
```

## Related Files

- `interp_data_contextual/`: Raw human annotation data
- `llm_judge_results_contextual/`: LLM judge evaluations
- `correlation_results.json`: Human-LLM agreement statistics
