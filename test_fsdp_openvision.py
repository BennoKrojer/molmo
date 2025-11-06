#!/usr/bin/env python3
"""Test OpenVision2 with FSDP wrapping to reproduce the training issue"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from open_clip import create_model
from huggingface_hub import hf_hub_download
import json
import os

def setup_distributed():
    """Setup single-process distributed for testing"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

def load_openvision2():
    """Load OpenVision2 the same way as training"""
    hf_repo = "UCSC-VLAA/openvision2-vit-large-patch14-336-vision-only"
    
    print("1. Loading config...")
    config_path = hf_hub_download(repo_id=hf_repo, filename='open_clip_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    print("2. Creating model...")
    model = create_model(
        model_name='ViT-L-14-336',
        pretrained=False,
        **model_config['model_cfg']
    )
    
    print("3. Loading weights...")
    weights_path = hf_hub_download(repo_id=hf_repo, filename='open_clip_pytorch_model.bin')
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.visual.load_state_dict(checkpoint, strict=False)
    
    vision_encoder = model.visual
    vision_encoder.output_tokens = True
    vision_encoder.eval()
    
    # Check weights are clean
    conv1_mean = vision_encoder.conv1.weight.float().mean().item()
    conv1_std = vision_encoder.conv1.weight.float().std().item()
    print(f"   After loading: conv1.weight mean={conv1_mean:.6f}, std={conv1_std:.6f}")
    
    # Check for NaN
    nan_count = 0
    for name, param in vision_encoder.named_parameters():
        if torch.isnan(param).any():
            print(f"   ❌ NaN in {name}")
            nan_count += 1
    if nan_count == 0:
        print(f"   ✓ All {sum(1 for _ in vision_encoder.parameters())} parameters are clean")
    
    # Freeze it (like training does)
    for param in vision_encoder.parameters():
        param.requires_grad = False
    print(f"   Frozen vision encoder (requires_grad=False)")
    
    return vision_encoder

def test_with_fsdp():
    print("=" * 80)
    print("Testing OpenVision2 with FSDP (replicating training)")
    print("=" * 80)
    
    setup_distributed()
    
    # Load model
    vision_encoder = load_openvision2()
    
    # Move to CUDA
    print("\n4. Moving to CUDA...")
    vision_encoder = vision_encoder.cuda()
    
    # Wrap with FSDP (this is what training does)
    print("\n5. Wrapping with FSDP...")
    try:
        fsdp_encoder = FSDP(vision_encoder)
        print("   ✓ FSDP wrapping completed")
    except Exception as e:
        print(f"   ❌ FSDP wrapping failed: {e}")
        return
    
    # Check weights after FSDP
    print("\n6. Checking weights after FSDP wrapping...")
    nan_params = []
    for name, param in fsdp_encoder.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"   ❌ NaN detected in {len(nan_params)} parameters after FSDP!")
        print(f"   First 5: {nan_params[:5]}")
        print(f"   THIS IS THE BUG!")
    else:
        print(f"   ✓ All parameters still clean after FSDP")
    
    # Try forward pass
    print("\n7. Testing forward pass...")
    x = torch.randn(1, 3, 336, 336, device='cuda')
    x = (x * 0.5) + 0.2
    
    with torch.no_grad():
        try:
            pooled, tokens = fsdp_encoder(x)
            print(f"   Output shape: {tokens.shape}")
            print(f"   Output stats: min={tokens.min():.4f}, max={tokens.max():.4f}")
            if torch.isnan(tokens).any():
                print(f"   ❌ NaN in output!")
            else:
                print(f"   ✓ Output is clean!")
        except Exception as e:
            print(f"   ❌ Forward failed: {e}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    if nan_params:
        print("FSDP wrapping corrupts OpenVision2 weights!")
        print("This is why training fails.")
    else:
        print("FSDP wrapping is OK. Issue must be elsewhere.")
    print("=" * 80)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_with_fsdp()

