source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview     --contextual-layer 8     --num-images 300

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview     --contextual-layer 16     --num-images 300

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview     --contextual-layer 24     --num-images 300

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview     --contextual-layer 8     --num-images 300

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview     --contextual-layer 16     --num-images 300

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/allenai_OLMo-7B-1024-preview     --contextual-layer 24     --num-images 300

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B     --contextual-layer 8     --num-images 300

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B --contextual-layer 16     --num-images 300

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/contextual_nearest_neighbors.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded     --contextual-dir molmo_data/contextual_llm_embeddings/Qwen_Qwen2-7B     --contextual-layer 24     --num-images 300