#!/usr/bin/env python3
"""
Upload hf_sharing folders to Hugging Face repo.
"""

from huggingface_hub import HfApi
import os

# Load token from file
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

# Create private repo
api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

# Folders to upload - all 9 model combinations
folders_to_upload = [
    # OLMO-7B
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded",
    # LLaMA3-8B
    "train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded",
    # Qwen2-7B
    "train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded",
    "train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded",
]

base_path = "hf_sharing"

# Check what's already uploaded
print("🔍 Checking existing uploads...")
repo_info = api.repo_info(repo_name, token=token, files_metadata=True)
existing_folders = set()
for f in repo_info.siblings:
    if f.rfilename.startswith("ckpts/"):
        parts = f.rfilename.split("/")
        if len(parts) >= 2:
            existing_folders.add(parts[1])

for folder_name in folders_to_upload:
    folder_path = os.path.join(base_path, folder_name)
    repo_path = os.path.join("ckpts", folder_name)  # Upload directly to ckpts/, not hf_sharing/ckpts/
    
    # Check if folder already exists (has at least 4 files: config.yaml, model.pt, optim.pt, train.pt)
    if folder_name in existing_folders:
        folder_files = [f for f in repo_info.siblings if f.rfilename.startswith(f"ckpts/{folder_name}/")]
        if len(folder_files) >= 4:
            print(f"\n⏭️  Skipping {folder_name} (already uploaded: {len(folder_files)} files)")
            continue
    
    print(f"\n📤 Uploading {folder_name}...")
    print(f"   Local: {folder_path}")
    print(f"   Repo:  {repo_path}")
    
    # Resolve symlink to actual path
    if os.path.islink(folder_path):
        actual_path = os.readlink(folder_path)
        if not os.path.isabs(actual_path):
            actual_path = os.path.join(os.path.dirname(folder_path), actual_path)
        folder_path = os.path.normpath(actual_path)
        print(f"   Resolved symlink to: {folder_path}")
    
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_name,
        path_in_repo=repo_path,
        token=token
    )
    
    print(f"   ✅ Uploaded {folder_name}")

print("\n✅ All folders uploaded!")

