"""Run this script with 'torchrun' to properly initialize distributed environment.

Example:
    torchrun --nproc_per_node=4 --master_port=29501 inspect_trained_ckpt.py
"""

import torch
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import time
import os
import yaml
import sys

# Force output flushing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def init_distributed():
    """Initialize distributed environment"""
    if not dist.is_initialized():
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError as e:
            print(f"failed to set multiprocessing start method: {e}")
        logger.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

        # Check if we're running with torchrun
        if "RANK" not in os.environ:
            raise RuntimeError(
                "This script must be run with 'torchrun' to properly initialize the distributed environment.\n"
                "Example: torchrun --nproc_per_node=4 --master_port=29501 inspect_trained_ckpt.py"
            )

        dist.init_process_group(backend="nccl")
        logger.info("Process group initialized")

def compare_tensors(t1, t2, name):
    """Compare two tensors and return if they are identical"""
    # Convert to regular tensors if they are sharded
    if hasattr(t1, 'local_tensor'):
        t1 = t1.local_tensor()
    if hasattr(t2, 'local_tensor'):
        t2 = t2.local_tensor()
        
    if t1.shape != t2.shape:
        return False
    if t1.dtype != t2.dtype:
        return False
    return torch.allclose(t1, t2, rtol=1e-5, atol=1e-5)

def load_pretrained_models():
    """Load pretrained vision encoder"""
    try:
        # Load config to get paths
        config_path = "/mnt/research/scratch/bkroje/molmo_data/checkpoints/train_mlp-only_pixmo_points_overlap-and-resize-c2/config.yaml"
        logger.info(f"Loading config from {config_path}")
        sys.stdout.flush()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a model instance to use its loading mechanism
        from olmo.model import Molmo
        from olmo.config import ModelConfig
        
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)
        # Set LLM to None to skip loading it
        model_config.llm = None
        model = Molmo(model_config)
        
        # Use the model's built-in loading mechanism
        logger.info("Loading pretrained weights using model's reset_with_pretrained_weights()...")
        sys.stdout.flush()
        model.reset_with_pretrained_weights()
        
        # Get only the vision state dict
        vit_state_dict = model.vision_backbone.state_dict() if model.vision_backbone else {}
        
        logger.info("Successfully loaded pretrained vision encoder")
        sys.stdout.flush()
        
        return vit_state_dict
    except Exception as e:
        logger.error(f"Error loading pretrained models: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        sys.stdout.flush()
        raise

def inspect_ckpt(ckpt_dir):
    """Inspect the contents of a checkpoint directory"""
    ckpt_dir = Path(ckpt_dir)
    rank = dist.get_rank()
    
    # First check what files we have (only rank 0)
    if rank == 0:
        logger.info(f"Contents of {ckpt_dir}:")
        for f in ckpt_dir.iterdir():
            logger.info(f"- {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        sys.stdout.flush()
    
    # Each process loads its corresponding rank file
    rank_file = ckpt_dir / f"rank{rank}.pt"
    logger.info(f"\nInspecting {rank_file}")
    logger.info(f"File size: {rank_file.stat().st_size / (1024*1024):.2f} MB")
    sys.stdout.flush()
    
    try:
        start_time = time.time()
        logger.info("Attempting to load state dict...")
        sys.stdout.flush()
        
        state_dict = torch.load(rank_file, map_location="cpu", weights_only=False)
        load_time = time.time() - start_time
        logger.info(f"Successfully loaded state dict in {load_time:.2f} seconds")
        sys.stdout.flush()
        
        # Print state dict information
        logger.info(f"Keys in state dict: {list(state_dict.keys())}")
        for k, v in state_dict.items():
            if hasattr(v, 'shape'):
                logger.info(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.info(f"{k}: type={type(v)}")
        sys.stdout.flush()
        
        # Wait for all processes to finish loading their checkpoints
        logger.info("Waiting for all processes to finish loading checkpoints...")
        sys.stdout.flush()
        dist.barrier()
        logger.info("All processes finished loading checkpoints")
        sys.stdout.flush()
        
        # Load pretrained models and compare weights (only on rank 0)
        if rank == 0:
            try:
                logger.info("\nLoading pretrained models for comparison...")
                sys.stdout.flush()
                
                vit_state_dict = load_pretrained_models()
                
                # Get model parameters from checkpoint
                model_params = state_dict['model']
                
                # Focus on one specific parameter for detailed comparison
                test_param = 'vision_backbone.image_vit.transformer.resblocks.22.feed_forward.w1.weight'
                logger.info(f"Rank {dist.get_rank()}: Starting parameter comparison for {test_param}")
                
                if test_param in model_params:
                    pretrained_name = test_param.replace('vision_backbone.', '')
                    if pretrained_name in vit_state_dict:
                        logger.info(f"\nDetailed comparison for {test_param}:")
                        
                        # Get local portion of sharded tensor
                        checkpoint_tensor = model_params[test_param].local_tensor()
                        pretrained_tensor = vit_state_dict[pretrained_name]
                        
                        # Get the slice of pretrained tensor corresponding to this rank
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                        slice_size = pretrained_tensor.shape[0] // world_size
                        start_idx = rank * slice_size
                        end_idx = (rank + 1) * slice_size
                        pretrained_slice = pretrained_tensor[start_idx:end_idx]
                        
                        logger.info(f"Rank {rank}: Checkpoint shape: {checkpoint_tensor.shape}")
                        logger.info(f"Rank {rank}: Pretrained slice shape: {pretrained_slice.shape}")
                        
                        if checkpoint_tensor.shape != pretrained_slice.shape:
                            logger.error(f"Shape mismatch! Checkpoint: {checkpoint_tensor.shape}, Pretrained slice: {pretrained_slice.shape}")
                            return
                        
                        # Print first few values from both tensors
                        checkpoint_vals = checkpoint_tensor.flatten()[:5]
                        pretrained_vals = pretrained_slice.flatten()[:5]
                        logger.info(f"Rank {rank}: First 5 values from checkpoint: {checkpoint_vals}")
                        logger.info(f"Rank {rank}: First 5 values from pretrained: {pretrained_vals}")
                        
                        # Calculate difference
                        diff = torch.abs(checkpoint_tensor - pretrained_slice)
                        max_diff = torch.max(diff)
                        mean_diff = torch.mean(diff)
                        logger.info(f"Rank {rank}: Max difference: {max_diff}")
                        logger.info(f"Rank {rank}: Mean difference: {mean_diff}")
                    else:
                        logger.warning(f"Parameter {pretrained_name} not found in pretrained model")
                else:
                    logger.warning(f"Parameter {test_param} not found in checkpoint")
                
                # Check MLP parameters (these should be different)
                logger.info("\nChecking MLP parameters...")
                sys.stdout.flush()
                
                mlp_params = [name for name in model_params.keys() if 'mlp.' in name or 'connector.' in name]
                if mlp_params:
                    logger.info(f"Found {len(mlp_params)} MLP parameters (these should differ from pretrained)")
                else:
                    logger.warning("No MLP parameters found in checkpoint")
                sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"Error during model comparison: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                sys.stdout.flush()
                raise
            
    except Exception as e:
        logger.error(f"Failed to load state dict: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        sys.stdout.flush()

if __name__ == "__main__":
    # Initialize distributed environment first
    init_distributed()
    
    ckpt_dir = "/mnt/research/scratch/bkroje/molmo_data/checkpoints/train_mlp-only_pixmo_points_overlap-and-resize-c2/step2000"
    inspect_ckpt(ckpt_dir)
    
    # Clean up distributed environment
    dist.destroy_process_group() 