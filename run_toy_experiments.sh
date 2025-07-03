source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29503 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/caption-prompt_mosaic-image.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/copy-prompt_mosaic-image.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/caption-prompt_1color-per-image.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py configs_toy_token-img2token/caption-prompt_1color-per-image_3token-only.yaml



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
