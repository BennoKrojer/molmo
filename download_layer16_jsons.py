#!/usr/bin/env python3
"""
Download all 9 layer 16 contextual nearest neighbor JSONs from HuggingFace.
Total size: ~1.5 GB

Usage:
    pip install huggingface_hub
    python download_layer16_jsons.py
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "BennoKrojer/vl_embedding_spaces"

MODELS = [
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

print(f"Downloading 9 layer 16 JSONs from {REPO_ID}")
print("=" * 60)

downloaded = []
for i, model in enumerate(MODELS, 1):
    filename = f"contextual_nearest_neighbors/{model}/contextual_neighbors_visual16_allLayers.json"
    print(f"\n[{i}/9] {model.split('_')[4]}_{model.split('_')[5].replace('_step12000-unsharded', '')}")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset"
    )
    downloaded.append(path)
    print(f"      → {path}")

print("\n" + "=" * 60)
print("✅ All 9 files downloaded!")
print(f"\nFiles are cached at: {Path(downloaded[0]).parent.parent}")
