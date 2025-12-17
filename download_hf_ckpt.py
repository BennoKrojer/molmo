from huggingface_hub import snapshot_download

# Load token from file
with open("hf_token.txt", "r") as f:
    token = f.read().strip()

# Download the checkpoint
checkpoint_path = snapshot_download(
    repo_id="BennoKrojer/vl_embedding_spaces",
    token=token,
    local_dir="./molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000",
    allow_patterns=["model.pt", "config.json", "train.pt", "optim.pt"]  # only these paths
)

print(f"Checkpoint downloaded to: {checkpoint_path}")

# from huggingface_hub import snapshot_download

# # Load token from file
# with open("hf_token.txt", "r") as f:
#     token = f.read().strip()

# checkpoint_path = snapshot_download(
#     repo_id="BennoKrojer/vl_embedding_spaces",
#     token=token,
#     local_dir="./json_files",
#     allow_patterns=[
#         "*/nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0.json"
#     ]
# )