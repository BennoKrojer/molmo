#!/usr/bin/env python3
"""Test that eval() mode prevents the instability"""

import torch
from open_clip import create_model
from huggingface_hub import hf_hub_download
import json

def load_openvision2():
    """Load OpenVision2 the same way as our OpenVision2Transformer"""
    hf_repo = "UCSC-VLAA/openvision2-vit-large-patch14-336-vision-only"
    
    config_path = hf_hub_download(repo_id=hf_repo, filename='open_clip_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    model = create_model(
        model_name='ViT-L-14-336',
        pretrained=False,
        **model_config['model_cfg']
    )
    
    weights_path = hf_hub_download(repo_id=hf_repo, filename='open_clip_pytorch_model.bin')
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.visual.load_state_dict(checkpoint, strict=False)
    
    vision_encoder = model.visual
    vision_encoder.output_tokens = True
    
    # Set to eval mode like we do in __init__
    vision_encoder.eval()
    
    return vision_encoder

def test_train_mode_override():
    print("=" * 80)
    print("Testing: Does calling .train() on parent model break vision encoder?")
    print("=" * 80)
    
    vision_encoder = load_openvision2()
    
    # Create some input that mimics training data (normalized)
    x = torch.randn(2, 3, 336, 336)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x * std) + mean
    
    print(f"\n1. Input: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
    
    # Test 1: Normal eval mode
    print(f"\n2. Test with eval mode (should be stable):")
    vision_encoder.eval()
    with torch.no_grad():
        _, tokens1 = vision_encoder(x)
    print(f"   Output: min={tokens1.min():.4f}, max={tokens1.max():.4f}, std={tokens1.std():.4f}")
    
    # Test 2: Simulate what happens in training - parent calls .train()
    print(f"\n3. Simulating parent model.train() call...")
    vision_encoder.train()  # This is what would happen when main model calls .train()
    print(f"   Vision encoder is now in: {'training' if vision_encoder.training else 'eval'} mode")
    
    # Now test WITHOUT re-calling eval() in forward
    print(f"\n4. Forward pass WITHOUT .eval() in forward():")
    with torch.no_grad():
        _, tokens2 = vision_encoder(x)
    print(f"   Output: min={tokens2.min():.4f}, max={tokens2.max():.4f}, std={tokens2.std():.4f}")
    if tokens2.max() > 1000:
        print(f"   ❌ UNSTABLE! This is the bug!")
    
    # Test 3: Simulate our fix - call .eval() before forward
    print(f"\n5. Forward pass WITH .eval() before forward() (our fix):")
    vision_encoder.eval()  # This is what we do in OpenVision2Transformer.forward()
    print(f"   Vision encoder is now in: {'training' if vision_encoder.training else 'eval'} mode")
    with torch.no_grad():
        _, tokens3 = vision_encoder(x)
    print(f"   Output: min={tokens3.min():.4f}, max={tokens3.max():.4f}, std={tokens3.std():.4f}")
    if tokens3.max() < 100:
        print(f"   ✓ STABLE! The fix works!")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    if tokens2.max() > 1000 and tokens3.max() < 100:
        print("✓ CONFIRMED: Calling .eval() in forward() fixes the issue!")
        print("  The parent model's .train() call was breaking the vision encoder.")
    else:
        print("? The instability wasn't reproduced. Issue might be elsewhere.")
    print("=" * 80)

if __name__ == "__main__":
    test_train_mode_override()

