source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 2

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 4

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 12

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 16

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 20

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 24

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 28

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded --llm_layer 32

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 2

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 4

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 12

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 16

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 20

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 24

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 28

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --llm_layer 32

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 2

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 4

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 12

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 16

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 20

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 24

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 28

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded --llm_layer 32


# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_siglip/step12000-unsharded --force-rerun

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336/step12000-unsharded --force-rerun

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded --force-rerun

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_siglip/step12000-unsharded --force-rerun

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336/step12000-unsharded --force-rerun

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip/step12000-unsharded --force-rerun

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29525 /home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/scripts/analysis/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336/step12000-unsharded --force-rerun