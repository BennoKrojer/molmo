source ../../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# OLMo-7B checkpoint - analyzing specific layers
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/logitlens.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded     --layers "0,8,16,24"     --top-k 5     --num-images 300

# Llama3-8B checkpoint - analyzing specific layers
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/logitlens.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336/step12000-unsharded     --layers "0,8,16,24"     --top-k 5     --num-images 300

# Qwen2-7B checkpoint - analyzing specific layers
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/analysis/logitlens.py     --ckpt-path molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_seed10/step12000-unsharded     --layers "0,8,16,24"     --top-k 5     --num-images 300