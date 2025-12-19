#!/usr/bin/env python3
"""
Upload contextual_nearest_neighbors JSONs to Hugging Face repo.
Uploads directly to contextual_nearest_neighbors/ (not hf_sharing/contextual_nearest_neighbors/)
"""

from huggingface_hub import HfApi
import os

# Load token from file
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

# Upload contextual_nearest_neighbors folder
local_path = "hf_sharing/contextual_nearest_neighbors"
repo_path = "contextual_nearest_neighbors"  # Top level, not under hf_sharing/

print(f"📤 Uploading contextual_nearest_neighbors...")
print(f"   Local: {local_path}")
print(f"   Repo:  {repo_path}")

api.upload_folder(
    folder_path=local_path,
    repo_id=repo_name,
    path_in_repo=repo_path,
    token=token
)

print(f"✅ Uploaded contextual_nearest_neighbors")

