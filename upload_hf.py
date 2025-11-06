
from huggingface_hub import HfApi, create_repo
import os

# Load token from file
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

# Create private repo
api = HfApi()
repo_name = "BennoKrojer/vl_embedding_spaces"  # change as needed

# try:
#     create_repo(repo_name, private=True, token=token)
# except:
#     pass  # repo might already exist

# Upload the entire checkpoint folder
checkpoint_path = "analysis_results/contextual_nearest_neighbors"

api.upload_folder(
    folder_path=checkpoint_path,
    repo_id=repo_name,
    path_in_repo="contextual_nearest_neighbors",
    token=token
)