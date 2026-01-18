#!/usr/bin/env python3
"""
Download contextual nearest neighbor JSONs from HuggingFace.

Usage:
    pip install huggingface_hub

    # Download all layers for all models (~14 GB)
    python download_layer16_jsons.py --all

    # Download specific layer (default: 16)
    python download_layer16_jsons.py --layer 16

    # Download specific layers
    python download_layer16_jsons.py --layers 0 8 16
"""

import argparse
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

ALL_LAYERS = [0, 1, 2, 4, 8, 16, 24, 30, 31]

def main():
    parser = argparse.ArgumentParser(description="Download LatentLens contextual NN data from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Download all layers (~14 GB)")
    parser.add_argument("--layer", type=int, default=16, help="Download single layer (default: 16)")
    parser.add_argument("--layers", type=int, nargs="+", help="Download specific layers")
    args = parser.parse_args()

    if args.all:
        layers = ALL_LAYERS
    elif args.layers:
        layers = args.layers
    else:
        layers = [args.layer]

    total_files = len(MODELS) * len(layers)
    print(f"Downloading {len(layers)} layer(s) × {len(MODELS)} models = {total_files} files")
    print(f"Repo: {REPO_ID}")
    print("=" * 60)

    downloaded = []
    for model in MODELS:
        parts = model.split('_')
        short_name = f"{parts[4]}+{parts[5].split('_')[0]}"
        print(f"\n{short_name}:")

        for layer in layers:
            filename = f"contextual_nearest_neighbors/{model}/contextual_neighbors_visual{layer}_allLayers.json"
            print(f"  Layer {layer:2d} ... ", end="", flush=True)

            path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            downloaded.append(path)
            print("✓")

    print("\n" + "=" * 60)
    print(f"✅ Downloaded {len(downloaded)} files!")
    print(f"Cache: {Path(downloaded[0]).parent.parent}")

if __name__ == "__main__":
    main()
