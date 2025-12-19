#!/usr/bin/env python3
"""
Check Hugging Face repo upload status.
Can be run from anywhere - finds hf_token.txt relative to script location.
"""

from huggingface_hub import HfApi
import os
from pathlib import Path
from collections import defaultdict

# Find project root (where hf_token.txt is located)
script_dir = Path(__file__).parent
token_path = script_dir / "hf_token.txt"

if not token_path.exists():
    print(f"❌ Error: hf_token.txt not found at {token_path}")
    print("   Please run this script from the project root or ensure hf_token.txt exists.")
    exit(1)

with open(token_path, "r") as f:
    token = f.read().strip()

api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"

repo_info = api.repo_info(repo_name, token=token, files_metadata=True)

# Group files by folder
folders = defaultdict(list)
for f in repo_info.siblings:
    if '/' in f.rfilename:
        parts = f.rfilename.split('/')
        if len(parts) >= 2:
            folder = '/'.join(parts[:2])
            folders[folder].append(f)
        else:
            folders[parts[0]].append(f)
    else:
        folders['root'].append(f)

total_size = sum(f.size for f in repo_info.siblings if f.size)

print(f'📊 Hugging Face Repo Status')
print(f'Repository: {repo_name}')
print(f'Total files: {len(repo_info.siblings)}')
print(f'Total size: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.2f} MB)')
print(f'\n📁 Folders:')
for folder in sorted(folders.keys()):
    if folder != 'root':
        folder_size = sum(f.size for f in folders[folder] if f.size)
        file_count = len(folders[folder])
        # Check if folder is complete (should have at least config.yaml, model.pt, optim.pt, train.pt)
        expected_files = {'config.yaml', 'model.pt', 'optim.pt', 'train.pt'}
        actual_files = {f.rfilename.split('/')[-1] for f in folders[folder]}
        is_complete = expected_files.issubset(actual_files)
        status = '✅' if is_complete else '⏳'
        print(f'  {status} {folder}')
        print(f'      {file_count} files, {folder_size / (1024**3):.2f} GB')
        if not is_complete:
            missing = expected_files - actual_files
            print(f'      Missing: {missing}')
    else:
        if folders[folder]:
            print(f'  📁 root: {len(folders[folder])} files')

