#!/usr/bin/env python3
"""
Upload ALL contextual_nearest_neighbors JSONs (all layers) to HuggingFace.
9 models × 9 layers = 81 files, ~14 GB total
"""

from huggingface_hub import HfApi
from pathlib import Path

# Load token from file
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

# Base directory
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

# All 9 layers
layers = [0, 1, 2, 4, 8, 16, 24, 30, 31]

print(f"Uploading ALL layers for 9 models to {repo_name}")
print(f"Total: {len(model_dirs)} models × {len(layers)} layers = {len(model_dirs) * len(layers)} files")
print("=" * 70)

total_uploaded = 0
total_skipped = 0

for model_idx, model_dir in enumerate(model_dirs, 1):
    # Extract short name for display
    parts = model_dir.split('_')
    llm = parts[4]
    encoder = parts[5].replace('_step12000-unsharded', '').replace('_seed10', '')
    short_name = f"{llm}+{encoder}"

    print(f"\n[{model_idx}/9] {short_name}")
    print("-" * 50)

    for layer in layers:
        json_file = base_dir / model_dir / f"contextual_neighbors_visual{layer}_allLayers.json"

        if not json_file.exists():
            print(f"  Layer {layer:2d}: NOT FOUND, skipping")
            total_skipped += 1
            continue

        size_mb = json_file.stat().st_size / (1024 * 1024)
        repo_path = f"contextual_nearest_neighbors/{model_dir}/contextual_neighbors_visual{layer}_allLayers.json"

        print(f"  Layer {layer:2d}: {size_mb:6.1f} MB ... ", end="", flush=True)

        api.upload_file(
            path_or_fileobj=str(json_file),
            path_in_repo=repo_path,
            repo_id=repo_name,
            token=token
        )

        print("✓")
        total_uploaded += 1

print("\n" + "=" * 70)
print(f"✅ Done! Uploaded: {total_uploaded}, Skipped: {total_skipped}")
print(f"Repo: https://huggingface.co/{repo_name}")
