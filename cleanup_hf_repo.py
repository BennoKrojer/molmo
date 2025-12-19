#!/usr/bin/env python3
"""
Clean up Hugging Face repo by deleting all JSON files and folders containing JSONs.
Keeps weight files (.pt, .bin).
"""

from huggingface_hub import HfApi
import os
from collections import defaultdict
from pathlib import Path

# Load token
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

print("Fetching repo info...")
repo_info = api.repo_info(repo_name, token=token, files_metadata=True)

# Categorize files
json_files = []
weight_files = []
other_files = []
all_files = {}

for f in repo_info.siblings:
    all_files[f.rfilename] = f
    if '.json' in f.rfilename:
        json_files.append(f.rfilename)
    elif '.pt' in f.rfilename or '.bin' in f.rfilename:
        weight_files.append(f.rfilename)
    else:
        other_files.append(f.rfilename)

print(f"\n📊 Current repo state:")
print(f"  JSON files: {len(json_files)}")
print(f"  Weight files: {len(weight_files)}")
print(f"  Other files: {len(other_files)}")
print(f"  Total: {len(all_files)}")

# Find directories that contain JSONs
dirs_with_jsons = defaultdict(set)
for json_file in json_files:
    if '/' in json_file:
        parts = json_file.split('/')
        # Add all parent directories
        for i in range(1, len(parts)):
            dir_path = '/'.join(parts[:i])
            dirs_with_jsons[dir_path].add(json_file)
    else:
        dirs_with_jsons[''].add(json_file)

print(f"\n📁 Directories containing JSONs: {len(dirs_with_jsons)}")

# For each directory with JSONs, check if it only contains JSONs and non-weight files
# If so, we can delete the entire directory
dirs_to_delete = set()
files_to_delete = set(json_files)  # Start with all JSON files

# Check each directory - if it only has JSONs and non-weight files, mark for deletion
for dir_path, json_files_in_dir in dirs_with_jsons.items():
    if not dir_path:  # Skip root
        continue
    
    # Get all files in this directory
    dir_files = [f for f in all_files.keys() if f.startswith(dir_path + '/')]
    
    # Check if directory only contains JSONs and non-weight files
    has_weights = any('.pt' in f or '.bin' in f for f in dir_files)
    has_non_json_non_weight = any(
        not ('.json' in f or '.pt' in f or '.bin' in f) 
        for f in dir_files
    )
    
    # If no weights and only JSONs (or JSONs + HTML/PNG which are likely visualization files), delete directory
    if not has_weights:
        # Check if it's a visualization/interactive folder (likely safe to delete)
        if 'interactive' in dir_path.lower() or 'visualization' in dir_path.lower():
            dirs_to_delete.add(dir_path)
            print(f"  Will delete directory (visualization): {dir_path}")
        # Or if it only has JSONs and maybe some small files
        elif not has_non_json_non_weight or all(
            f.endswith(('.html', '.png', '.json')) for f in dir_files
        ):
            # Check total size - if small, likely safe to delete
            total_size = sum(all_files[f].size for f in dir_files if f in all_files)
            if total_size < 100 * 1024 * 1024:  # Less than 100MB
                dirs_to_delete.add(dir_path)
                print(f"  Will delete directory (small, no weights): {dir_path}")

# Add all files in directories to delete
for dir_path in dirs_to_delete:
    dir_files = [f for f in all_files.keys() if f.startswith(dir_path + '/')]
    files_to_delete.update(dir_files)

print(f"\n🗑️  Files to delete: {len(files_to_delete)}")
print(f"   (including {len(json_files)} JSON files)")

# Show what will be kept
files_to_keep = set(all_files.keys()) - files_to_delete
print(f"\n✅ Files to keep: {len(files_to_keep)}")
for f in sorted(files_to_keep):
    size_mb = all_files[f].size / (1024**2) if f in all_files else 0
    print(f"   {f} ({size_mb:.2f} MB)")

# Confirm deletion
print(f"\n⚠️  About to delete {len(files_to_delete)} files")
print("Proceeding with deletion...")

# Delete files
print(f"\n🗑️  Deleting files...")
deleted_count = 0
failed_count = 0

for file_path in sorted(files_to_delete):
    try:
        api.delete_file(
            path_in_repo=file_path,
            repo_id=repo_name,
            token=token
        )
        deleted_count += 1
        if deleted_count % 10 == 0:
            print(f"  Deleted {deleted_count}/{len(files_to_delete)} files...")
    except Exception as e:
        print(f"  ❌ Failed to delete {file_path}: {e}")
        failed_count += 1

print(f"\n✅ Done!")
print(f"   Deleted: {deleted_count}")
print(f"   Failed: {failed_count}")

