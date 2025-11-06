#!/usr/bin/env python3
"""Test the full preprocessing pipeline that training uses"""

import torch
import numpy as np
from PIL import Image
from open_clip import create_model
from huggingface_hub import hf_hub_download
import json

# Import our preprocessing
import sys
sys.path.insert(0, '.')
from olmo.data.model_preprocessor import openvision_resize

def load_openvision2():
    """Load OpenVision2 model"""
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
    vision_encoder.eval()
    
    return vision_encoder

def patches_to_image(patches, patch_size=14):
    """
    Reconstruct image from patches (mimics OpenVision2Transformer.forward)
    patches: (B, num_patches, patch_size * patch_size * 3)
    returns: (B, 3, H, W)
    """
    B, N, D = patches.shape
    
    # Calculate patch grid dimensions
    patch_h = patch_w = int(N ** 0.5)
    assert patch_h * patch_w == N, f"num_patches={N} is not a perfect square"
    
    # Reshape: (B, N, patch_size * patch_size * 3) -> (B, patch_h, patch_w, patch_size, patch_size, 3)
    patches = patches.reshape(B, patch_h, patch_w, patch_size, patch_size, 3)
    
    # Rearrange to form complete image
    patches = patches.permute(0, 5, 1, 3, 2, 4)  # (B, 3, patch_h, patch_size, patch_w, patch_size)
    image = patches.reshape(B, 3, patch_h * patch_size, patch_w * patch_size)
    
    return image

def test_preprocessing_pipeline():
    print("=" * 80)
    print("Testing Full Preprocessing Pipeline")
    print("=" * 80)
    
    # 1. Create a test image
    print("\n1. Creating test image (336x336)...")
    img = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
    print(f"   Raw image: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    
    # 2. Resize (using our openvision_resize function)
    print("\n2. Applying openvision_resize...")
    img_resized, _ = openvision_resize(img, (336, 336))
    print(f"   Resized: shape={img_resized.shape}, dtype={img_resized.dtype}, min={img_resized.min()}, max={img_resized.max()}")
    
    # 3. Normalize to [0, 1]
    print("\n3. Normalizing to [0, 1]...")
    img_norm = img_resized.astype(np.float32) / 255.0
    print(f"   Normalized: shape={img_norm.shape}, dtype={img_norm.dtype}, min={img_norm.min():.4f}, max={img_norm.max():.4f}")
    
    # 4. Apply ImageNet normalization (what openvision2 should use)
    print("\n4. Applying ImageNet normalization...")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_normalized = (img_norm - mean) / std
    print(f"   After normalization: min={img_normalized.min():.4f}, max={img_normalized.max():.4f}, mean={img_normalized.mean():.4f}, std={img_normalized.std():.4f}")
    
    # 5. Convert to patches (14x14)
    print("\n5. Converting to patches (14x14)...")
    patch_size = 14
    H, W, C = img_normalized.shape
    patch_h = H // patch_size
    patch_w = W // patch_size
    
    # Reshape to patches: (H, W, C) -> (patch_h, patch_size, patch_w, patch_size, C)
    img_patches = img_normalized.reshape(patch_h, patch_size, patch_w, patch_size, C)
    # Rearrange: (patch_h, patch_w, patch_size, patch_size, C)
    img_patches = img_patches.transpose(0, 2, 1, 3, 4)
    # Flatten each patch: (patch_h, patch_w, patch_size * patch_size * C)
    img_patches = img_patches.reshape(patch_h * patch_w, patch_size * patch_size * C)
    
    print(f"   Patches: shape={img_patches.shape}")
    print(f"   Patch values: min={img_patches.min():.4f}, max={img_patches.max():.4f}, mean={img_patches.mean():.4f}")
    
    # 6. Convert patches to torch and add batch dimension
    patches_tensor = torch.from_numpy(img_patches).unsqueeze(0)  # (1, num_patches, D)
    print(f"   Patches tensor: shape={patches_tensor.shape}")
    
    # 7. Reconstruct image from patches (what OpenVision2Transformer does)
    print("\n6. Reconstructing image from patches...")
    reconstructed = patches_to_image(patches_tensor, patch_size=14)
    print(f"   Reconstructed: shape={reconstructed.shape}")
    print(f"   Reconstructed values: min={reconstructed.min():.4f}, max={reconstructed.max():.4f}, mean={reconstructed.mean():.4f}, std={reconstructed.std():.4f}")
    
    # 8. Pass to OpenVision2
    print("\n7. Loading OpenVision2...")
    vision_encoder = load_openvision2()
    
    print("\n8. Passing through OpenVision2...")
    with torch.no_grad():
        pooled, tokens = vision_encoder(reconstructed)
    
    print(f"   Output: shape={tokens.shape}")
    print(f"   Output stats: min={tokens.min():.4f}, max={tokens.max():.4f}, mean={tokens.mean():.4f}, std={tokens.std():.4f}")
    
    if torch.isnan(tokens).any():
        print(f"   ❌ NaN detected! Count: {torch.isnan(tokens).sum()}/{tokens.numel()}")
    
    if tokens.max() > 1000:
        print(f"   ❌ WARNING: Extremely large values detected (max={tokens.max():.2f})!")
        print(f"   This matches the training issue!")
    else:
        print(f"   ✓ Output looks stable!")
    
    # 9. Compare with direct image input
    print("\n9. Comparing with direct image input (no patch/reconstruct)...")
    img_direct = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    print(f"   Direct input: shape={img_direct.shape}, min={img_direct.min():.4f}, max={img_direct.max():.4f}")
    
    with torch.no_grad():
        pooled_direct, tokens_direct = vision_encoder(img_direct)
    
    print(f"   Direct output: min={tokens_direct.min():.4f}, max={tokens_direct.max():.4f}, mean={tokens_direct.mean():.4f}, std={tokens_direct.std():.4f}")
    print(f"   Difference between reconstructed and direct: {torch.abs(reconstructed - img_direct).max():.6f}")
    print(f"   Output difference: {torch.abs(tokens - tokens_direct).max():.6f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    if torch.abs(reconstructed - img_direct).max() < 1e-5:
        print("✓ Reconstruction is perfect")
    else:
        print("❌ Reconstruction differs from original!")
    
    if tokens.max() > 1000:
        print("❌ Pipeline produces unstable outputs - this is the bug!")
    else:
        print("✓ Pipeline produces stable outputs")
    print("=" * 80)

if __name__ == "__main__":
    test_preprocessing_pipeline()

