"""Minimal script to iterate over 300 validation examples and print captions.
Also loads and distributes model weights across GPUs using FSDP.

Usage: torchrun --nproc_per_node=2 scripts/minimal_val_captions.py --ckpt-path <checkpoint_path>

Or with specific GPUs:
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 scripts/minimal_val_captions.py --ckpt-path <checkpoint_path>
"""
import argparse

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from olmo.model import Molmo
from olmo.config import TrainConfig, ModelConfig
from olmo.torch_util import get_local_rank, get_world_size
from olmo.data.pixmo_datasets import PixMoCap
from olmo.data import build_mm_preprocessor
from olmo.util import resource_path


def main():
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # Set CUDA device
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    
    # Parse arguments (use parse_known_args to ignore distributed launcher arguments like --local-rank)
    parser = argparse.ArgumentParser(description="Generate captions for validation images using the model")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    args, unknown = parser.parse_known_args()
    
    checkpoint_path = args.ckpt_path
    
    # Parameters
    num_val_images = 10
    
    if local_rank == 0:
        print(f"Running on {world_size} processes")
        print(f"Processing {num_val_images} validation images")
        print(f"Loading model from {checkpoint_path}")
    
    # Load model on CPU first
    # This works with both full checkpoints and stripped (MLP-only) checkpoints
    cfg = TrainConfig.load(f"{checkpoint_path}/config.yaml")
    # Override init_device to avoid meta tensors
    cfg.model.init_device = None
    
    # Initialize model
    model = Molmo(cfg.model)
    
    # Load pretrained weights (LLM + ViT) 
    model.reset_with_pretrained_weights()
    
    # Load checkpoint weights (works with both full and stripped checkpoints)
    checkpoint_weights = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    
    if local_rank == 0:
        print(f"Loaded {len(checkpoint_weights)} parameters from checkpoint")
    
    model.eval()
    
    # Create preprocessor
    model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )
    
    # Wrap model with FSDP for sharding
    if local_rank == 0:
        print("Wrapping model with FSDP for sharding...")
    
    # Get FSDP wrap policy from the model
    wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")
    
    # Wrap model in FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    
    if local_rank == 0:
        print(f"Model wrapped with FSDP on device: {device}")
    
    # Wait for all processes to finish loading model
    dist.barrier()
    
    # Distribute images across processes
    images_per_process = num_val_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    
    # Handle remainder for last process
    if local_rank == world_size - 1:
        end_idx = num_val_images
    
    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1}")
    
    # Load validation dataset
    val_dataset = PixMoCap(split="validation", mode="captions")
    
    # Process assigned images for this process
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(42 + i)
        example_data = val_dataset.get(i, rng)
        
        # Create example with prompt
        example = {
            "image": example_data["image"],
            "messages": ["Caption:"]
        }
        
        # Preprocess example
        batch = preprocessor(example, rng=rng)
        
        # Generate caption
        try:
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    images = torch.tensor(batch["images"]).unsqueeze(0).to(device)
                    image_masks = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                    image_input_idx = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                    
                    output = model.generate(
                        input_ids=input_ids,
                        images=images,
                        image_masks=image_masks,
                        image_input_idx=image_input_idx,
                        max_steps=args.max_tokens,
                        min_steps=4,
                        is_distributed=False
                    )
                    token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                    caption_text = preprocessor.tokenizer.decode(token_ids[0])
        except Exception as e:
            caption_text = f"ERROR: {e}"
        
        # Print generated caption
        print(f"[Image {i}] {caption_text}")
    
    # Wait for all processes to finish
    dist.barrier()
    
    if local_rank == 0:
        print("Done!")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

