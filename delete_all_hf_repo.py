#!/usr/bin/env python3
"""
Delete all files from Hugging Face repo.
"""

from huggingface_hub import HfApi
import os

# Load token
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

print("Fetching repo info...")
repo_info = api.repo_info(repo_name, token=token, files_metadata=True)

all_files = [f.rfilename for f in repo_info.siblings]
total_size = sum(f.size for f in repo_info.siblings if f.size)

print(f"\n📊 Current repo state:")
print(f"  Total files: {len(all_files)}")
print(f"  Total size: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.2f} MB)")
print(f"\n⚠️  About to delete ALL {len(all_files)} files")
print("Proceeding with deletion...")

# Delete files
deleted_count = 0
failed_count = 0

for file_path in sorted(all_files):
    try:
        api.delete_file(
            path_in_repo=file_path,
            repo_id=repo_name,
            token=token
        )
        deleted_count += 1
        if deleted_count % 10 == 0:
            print(f"  Deleted {deleted_count}/{len(all_files)} files...")
    except Exception as e:
        print(f"  ❌ Failed to delete {file_path}: {e}")
        failed_count += 1

print(f"\n✅ Done!")
print(f"   Deleted: {deleted_count}")
print(f"   Failed: {failed_count}")

# Verify
print(f"\n🔍 Verifying...")
try:
    repo_info_after = api.repo_info(repo_name, token=token, files_metadata=True)
    remaining_files = len(repo_info_after.siblings)
    if remaining_files == 0:
        print("✅ Repo is now empty!")
    else:
        print(f"⚠️  {remaining_files} files still remain")
except Exception as e:
    print(f"Error verifying: {e}")

