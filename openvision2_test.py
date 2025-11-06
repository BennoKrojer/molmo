#!/usr/bin/env python3
"""Test OpenVision2 loading and output stability"""

import torch
from open_clip import create_model
from huggingface_hub import hf_hub_download
import json

def test_openvision2():
    print("=" * 80)
    print("Testing OpenVision2 with train vs eval mode")
    print("=" * 80)
    
    # Load model the same way as our integration
    hf_repo = "UCSC-VLAA/openvision2-vit-large-patch14-336-vision-only"
    
    print(f"\n1. Loading config from {hf_repo}...")
    config_path = hf_hub_download(repo_id=hf_repo, filename='open_clip_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"2. Creating model architecture...")
    model = create_model(
        model_name='ViT-L-14-336',
        pretrained=False,
        **model_config['model_cfg']
    )
    
    print(f"3. Loading weights...")
    weights_path = hf_hub_download(repo_id=hf_repo, filename='open_clip_pytorch_model.bin')
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.visual.load_state_dict(checkpoint, strict=False)
    
    vision_encoder = model.visual
    vision_encoder.output_tokens = True
    
    # Verify weights loaded
    conv1_mean = vision_encoder.conv1.weight.float().mean().item()
    conv1_std = vision_encoder.conv1.weight.float().std().item()
    print(f"   conv1.weight: mean={conv1_mean:.6f}, std={conv1_std:.6f}")
    
    # CHECK FOR NaN IN LOADED WEIGHTS
    print(f"\n3.5. Checking for NaN in weights...")
    nan_params = []
    for name, param in vision_encoder.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    if not nan_params:
        print(f"   ✓ All {sum(1 for _ in vision_encoder.parameters())} parameters are clean")
    else:
        print(f"   ❌ NaN found in {len(nan_params)} parameters!")
        for name in nan_params[:5]:
            print(f"      - {name}")
    
    # Check specific problematic biases from training
    print(f"\n   Checking specific biases that fail in training:")
    resblock0 = vision_encoder.transformer.resblocks[0]
    in_proj_bias = resblock0.attn.in_proj_bias
    out_proj_bias = resblock0.attn.out_proj.bias
    print(f"   - in_proj_bias: has_nan={torch.isnan(in_proj_bias).any()}, mean={in_proj_bias.mean():.6f}")
    print(f"   - out_proj_bias: has_nan={torch.isnan(out_proj_bias).any()}, mean={out_proj_bias.mean():.6f}")
    
    print(f"\n4. Creating test input (normalized images)...")
    # Simulate normalized input (ImageNet stats)
    # Shape: (B=2, C=3, H=336, W=336)
    x = torch.randn(2, 3, 336, 336)
    # Normalize to reasonable range
    x = (x * 0.5) + 0.2  # mean≈0.2, std≈0.5
    print(f"   Input: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Test 1: Training mode (BAD)
    print(f"\n5. Testing with TRAINING mode...")
    vision_encoder.train()
    with torch.no_grad():
        pooled_train, tokens_train = vision_encoder(x)
    print(f"   Output: shape={tokens_train.shape}")
    print(f"   Output stats: min={tokens_train.min():.4f}, max={tokens_train.max():.4f}, mean={tokens_train.mean():.4f}, std={tokens_train.std():.4f}")
    if torch.isnan(tokens_train).any():
        print(f"   ❌ NaN detected! Count: {torch.isnan(tokens_train).sum()}/{tokens_train.numel()}")
    if tokens_train.max() > 1000:
        print(f"   ❌ WARNING: Extremely large values detected (max={tokens_train.max():.2f})!")
    
    # Test 2: Eval mode (GOOD)
    print(f"\n6. Testing with EVAL mode...")
    vision_encoder.eval()
    with torch.no_grad():
        pooled_eval, tokens_eval = vision_encoder(x)
    print(f"   Output: shape={tokens_eval.shape}")
    print(f"   Output stats: min={tokens_eval.min():.4f}, max={tokens_eval.max():.4f}, mean={tokens_eval.mean():.4f}, std={tokens_eval.std():.4f}")
    if torch.isnan(tokens_eval).any():
        print(f"   ❌ NaN detected! Count: {torch.isnan(tokens_eval).sum()}/{tokens_eval.numel()}")
    if tokens_eval.max() > 1000:
        print(f"   ❌ WARNING: Extremely large values detected (max={tokens_eval.max():.2f})!")
    else:
        print(f"   ✓ Output looks stable!")
    
    # Compare
    print(f"\n7. Comparison:")
    print(f"   Train mode: min={tokens_train.min():.2f}, max={tokens_train.max():.2f}, std={tokens_train.std():.2f}")
    print(f"   Eval mode:  min={tokens_eval.min():.2f}, max={tokens_eval.max():.2f}, std={tokens_eval.std():.2f}")
    print(f"   Difference: {torch.abs(tokens_train - tokens_eval).mean():.4f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    if tokens_train.max() > 1000 and tokens_eval.max() < 100:
        print("✓ CONFIRMED: eval() mode fixes the instability issue!")
        print("  Train mode produces huge values, eval mode is stable.")
    elif tokens_train.max() < 100 and tokens_eval.max() < 100:
        print("? Both modes seem stable. The issue might be elsewhere.")
    else:
        print("? Unexpected results. Need further investigation.")
    print("=" * 80)

if __name__ == "__main__":
    test_openvision2()

