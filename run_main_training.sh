source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/caption-prompt_mosaic-image.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/copy-prompt_mosaic-image.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/caption-prompt_1color-per-image.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/caption-prompt_1color-per-image_3token-only.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions-first-sentence_olmo-7b_vit-l-14-336.yaml

export NCCL_TIMEOUT=1800000
export TORCH_DISTRIBUTED_TIMEOUT=1800000

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs/rest/baseline_pixmo-points_olmo-7b_vit-l-14-336.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO torchrun --master_port=29506 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_llama3-8b_siglip.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO torchrun --master_port=29506 --nproc_per_node=4 scripts/train.py configs/baseline_pixmo-captions_llama3-8b_dinov2-large-336.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29504 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_llama3-8b_siglip.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29504 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_llama3-8b_dinov2-large-336.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs/baseline_pixmo-captions_llama3-8b_vit-l-14-336_linear.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs/baseline_pixmo-captions_olmo-7b_openvision2-l-14-336.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs/baseline_pixmo-captions_qwen2-7b_openvision2-l-14-336.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs/baseline_pixmo-captions_llama3-8b_openvision2-l-14-336.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_olmo-7b_siglip.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_olmo-7b_dinov2-large-336.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_olmo-7b_vit-l-14-336.yaml --load_path=molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/latest
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29502 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_olmoe-1b_7b_vit-l-14-336.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29504 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_qwen2-7b_dinov2-large-336.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29505 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_qwen2-7b_siglip.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions_olmo-7b_vit-l-14-336_seed10.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29502 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/baseline_pixmo-captions.yaml
# if [ "$1" = "small" ]; then
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/datasize-10_empty-prompt_1token-per-image.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/datasize-10_copy-prompt_1token-per-image.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/datasize-10_caption-prompt_1token-per-image.yaml
# elif [ "$1" = "large" ]; then
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/datasize-1000_empty-prompt_1token-per-image.yaml
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/datasize-1000_copy-prompt_1token-per-image.yaml
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/datasize-1000_caption-prompt_1token-per-image.yaml
# else
    # echo "Please specify either 'small' or 'large' as argument"
    # exit 1

# fi
