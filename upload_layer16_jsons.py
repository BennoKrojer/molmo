#!/usr/bin/env python3
"""
Upload layer 16 contextual_nearest_neighbors JSONs to Hugging Face repo.
These are the data files used to create similarity_hist_combined_3x3_visual16.pdf
"""

from huggingface_hub import HfApi
from pathlib import Path

# Load token from file
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

# Base directory for contextual NN results
base_dir = Path("analysis_results/contextual_nearest_neighbors")

# All 9 model combinations
model_dirs = [
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

print(f"Uploading layer 16 JSONs to {repo_name}")
print("=" * 60)

for model_dir in model_dirs:
    json_file = base_dir / model_dir / "contextual_neighbors_visual16_allLayers.json"

    if not json_file.exists():
        print(f"❌ NOT FOUND: {json_file}")
        continue

    # File size
    size_mb = json_file.stat().st_size / (1024 * 1024)

    # Path in repo: contextual_nearest_neighbors/{model_dir}/contextual_neighbors_visual16_allLayers.json
    repo_path = f"contextual_nearest_neighbors/{model_dir}/contextual_neighbors_visual16_allLayers.json"

    print(f"\n📤 Uploading: {model_dir}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Repo path: {repo_path}")

    api.upload_file(
        path_or_fileobj=str(json_file),
        path_in_repo=repo_path,
        repo_id=repo_name,
        token=token
    )

    print(f"   ✅ Done")

print("\n" + "=" * 60)
print("✅ All layer 16 JSONs uploaded!")
print(f"\nDownload from: https://huggingface.co/datasets/{repo_name}")
