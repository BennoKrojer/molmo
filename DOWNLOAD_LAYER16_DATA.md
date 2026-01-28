# Downloading Layer 16 LatentLens Data

This document explains how to download the layer 16 contextual nearest neighbor JSON files used to create `similarity_hist_combined_3x3_visual16.pdf`.

## Quick Download (All 9 Models)

```bash
pip install huggingface_hub

# Download all layer 16 JSONs (~1.5 GB total)
from huggingface_hub import hf_hub_download

models = [
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded",
]

for model in models:
    path = hf_hub_download(
        repo_id="BennoKrojer/vl_embedding_spaces",
        filename=f"contextual_nearest_neighbors/{model}/contextual_neighbors_visual16_allLayers.json",
        repo_type="dataset"
    )
    print(f"Downloaded: {path}")
```

## Download Single Model

```python
from huggingface_hub import hf_hub_download

# Example: OLMo + CLIP
path = hf_hub_download(
    repo_id="BennoKrojer/vl_embedding_spaces",
    filename="contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/contextual_neighbors_visual16_allLayers.json",
    repo_type="dataset"
)
print(f"Downloaded to: {path}")
```

## File Sizes

| Model | Size |
|-------|------|
| OLMo + CLIP | 220 MB |
| OLMo + SigLIP | 167 MB |
| OLMo + DINOv2 | 139 MB |
| Llama3 + CLIP | 174 MB |
| Llama3 + SigLIP | 177 MB |
| Llama3 + DINOv2 | 175 MB |
| Qwen2 + CLIP | 160 MB |
| Qwen2 + SigLIP | 167 MB |
| Qwen2 + DINOv2 | 163 MB |
| **Total** | **~1.5 GB** |

## JSON Structure

Each JSON file contains:
```json
{
  "results": [
    {
      "image_idx": 0,
      "chunks": [
        {
          "chunk_name": "chunk_0",
          "patches": [
            {
              "patch_idx": 0,
              "nearest_contextual_neighbors": [
                {
                  "token_str": "dog",
                  "caption": "a brown dog sitting on grass",
                  "similarity": 0.42
                },
                ...
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## HuggingFace Repo

https://huggingface.co/datasets/BennoKrojer/vl_embedding_spaces
