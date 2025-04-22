export MOLMO_DATA_DIR=/mnt/research/scratch/bkroje/molmo_data
export HF_HOME=/mnt/research/scratch/bkroje/molmo_data/huggingface
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=4,5,6,7

export WANDB_PROJECT=vlm_embedding_interpretability
export WANDB_ENTITY=bennokrojer

CONFIG=$1

# torchrun --nproc-per-node=4 launch_scripts/train_captioner.py qwen2_7b --save_folder=train_mlp-only_pixmo_points --dataset=pixmo_points --points_kind=basic --wandb=null --ft_connector --global_batch_size=32 --device_train_microbatch_size=4 --device_train_batch_size=4
torchrun --nproc_per_node=4 scripts/train.py $CONFIG