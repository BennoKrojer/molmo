#!/usr/bin/env python3
"""Test OpenVision2 with real data through actual training pipeline"""

import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')

from olmo.config import ModelConfig, VisionBackboneConfig, TokenizerConfig
from olmo.data.model_preprocessor import MultiModalPreprocessor
from olmo.image_vit import OpenVision2Transformer

def create_test_config():
    """Create a minimal config for OpenVision2"""
    vision_config = VisionBackboneConfig(
        image_model_type='openvision2',
        image_model_name='UCSC-VLAA/openvision2-vit-large-patch14-336-vision-only',
        image_default_input_size=(336, 336),
        image_patch_size=14,
        image_pos_patch_size=14,
        image_emb_dim=1024,
        image_num_heads=16,
        image_num_key_value_heads=16,
        image_num_layers=24,
        image_head_dim=64,
        image_mlp_dim=4096,
        image_mlp_activations='quick_gelu',
        image_dropout_rate=0.0,
        image_num_pos=577,
        image_norm_eps=1e-05,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=0.02,
        fsdp_wrap=False,
        resize_mode='openvision',
    )
    
    tokenizer_config = TokenizerConfig(
        identifier='allenai/dolma2-tokenizer',
    )
    
    model_config = ModelConfig(
        d_model=4096,
        n_heads=32,
        n_layers=32,
        vision_backbone=vision_config,
        tokenizer=tokenizer_config,
        max_crops=12,
        crop_mode='resize',
        use_col_tokens=True,
        overlap_margins=(4, 4),
        image_pooling_h=1,
        image_pooling_w=1,
        vocab_size=100278,
        additional_vocab_size=128,
        pad_tokenizer=False,  # Not needed for vision-only test
    )
    
    return model_config

def load_sample_image():
    """Load a sample image - create a realistic one if dataset not available"""
    # Try to load from dataset, fall back to synthetic
    try:
        from datasets import load_dataset
        print("Attempting to load real image from pixmo_cap dataset...")
        dataset = load_dataset('allenai/pixmo-cap', split='train', streaming=True)
        sample = next(iter(dataset))
        image = sample['image']
        print(f"✓ Loaded real image from dataset: {image.size}, mode={image.mode}")
        return np.array(image)
    except Exception as e:
        print(f"Could not load from dataset ({e}), using synthetic image")
        # Create a realistic synthetic image (not pure random)
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        print(f"Created synthetic image: shape={img.shape}")
        return img

def test_full_pipeline():
    print("=" * 80)
    print("Testing OpenVision2 with Real Data Pipeline")
    print("=" * 80)
    
    # 1. Create config
    print("\n1. Creating model config...")
    config = create_test_config()
    
    # 2. Load image
    print("\n2. Loading sample image...")
    image = load_sample_image()
    print(f"   Image: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
    
    # 3. Create preprocessor (using same logic as in olmo/data/__init__.py)
    print("\n3. Creating MultiModalPreprocessor...")
    v_cfg = config.vision_backbone
    preprocessor = MultiModalPreprocessor(
        tokenizer=config.get_tokenizer(),
        normalize=str(v_cfg.image_model_type),
        crop_mode=config.crop_mode,
        max_crops=config.max_crops,
        overlap_margins=config.overlap_margins,
        resize=v_cfg.resize_mode,
        use_col_tokens=config.use_col_tokens,
        base_image_input_size=v_cfg.image_default_input_size,
        image_pooling_w=config.image_pooling_w,
        image_pooling_h=config.image_pooling_h,
        image_token_length_w=v_cfg.image_default_input_size[1] // v_cfg.image_patch_size,
        image_token_length_h=v_cfg.image_default_input_size[0] // v_cfg.image_patch_size,
        image_patch_size=v_cfg.image_patch_size,
    )
    print(f"   Preprocessor: normalize='{preprocessor.normalize}', resize='{preprocessor.resize}'")
    
    # 4. Preprocess image to patches
    print("\n4. Preprocessing image to patches...")
    # Returns: (patches, tokens, crop_coords, img_mask)
    patches, tokens, crop_coords, img_mask = preprocessor.image_to_patches_and_tokens(image, is_training=False)
    print(f"   Patches: shape={patches.shape}, dtype={patches.dtype}")
    print(f"   Patch values: min={patches.min():.4f}, max={patches.max():.4f}, mean={patches.mean():.4f}, std={patches.std():.4f}")
    print(f"   Tokens: shape={tokens.shape}")
    print(f"   Image mask: shape={img_mask.shape}")
    
    if np.isnan(patches).any():
        print(f"   ❌ NaN in patches! Count: {np.isnan(patches).sum()}/{patches.size}")
        return
    
    # 5. Convert to torch tensor
    print("\n5. Converting to torch tensor...")
    # patches is already (1, num_patches, D) from preprocessor in resize mode
    patches_torch = torch.from_numpy(patches)  # Shape: (1, num_patches, D)
    print(f"   Tensor: shape={patches_torch.shape}, dtype={patches_torch.dtype}")
    
    # 6. Create OpenVision2Transformer
    print("\n6. Creating OpenVision2Transformer...")
    try:
        vit = OpenVision2Transformer(config)
        print(f"   ✓ OpenVision2Transformer created")
        print(f"   num_prefix_tokens={vit.num_prefix_tokens}")
    except Exception as e:
        print(f"   ❌ Failed to create OpenVision2Transformer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. Test WITHOUT gradients (like our debug scripts)
    print("\n7. Test WITHOUT gradients (torch.no_grad)...")
    with torch.no_grad():
        try:
            # patches_torch is already (1, num_patches, D) in resize mode
            print(f"   Input: shape={patches_torch.shape}")
            output = vit.forward(patches_torch)
            features = output[0]  # Get first (and only) layer output
            print(f"   Output: shape={features.shape}")
            print(f"   Output stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")
            
            if torch.isnan(features).any():
                print(f"   ❌ NaN in output! Count: {torch.isnan(features).sum()}/{features.numel()}")
            else:
                print(f"   ✓ No NaN, output looks stable!")
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Test WITH gradients (like actual training)
    print("\n8. Test WITH gradients enabled (training mode)...")
    try:
        patches_grad = patches_torch.clone()  # (1, num_patches, D)
        patches_grad.requires_grad = True
        
        print(f"   Input: shape={patches_grad.shape}, requires_grad={patches_grad.requires_grad}")
        output = vit.forward(patches_grad)
        features = output[0]
        print(f"   Output: shape={features.shape}, requires_grad={features.requires_grad}")
        print(f"   Output stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")
        
        if torch.isnan(features).any():
            print(f"   ❌ NaN in output! Count: {torch.isnan(features).sum()}/{features.numel()}")
            print(f"   THIS IS THE BUG - gradients cause instability!")
        else:
            print(f"   ✓ No NaN even with gradients!")
            
            # Try backward pass
            print("\n9. Testing backward pass...")
            loss = features.mean()
            print(f"   Loss: {loss.item():.4f}")
            loss.backward()
            print(f"   ✓ Backward pass completed without error!")
            if patches_grad.grad is not None:
                print(f"   Gradients: min={patches_grad.grad.min():.4f}, max={patches_grad.grad.max():.4f}")
                if torch.isnan(patches_grad.grad).any():
                    print(f"   ❌ NaN in gradients!")
                else:
                    print(f"   ✓ Gradients are clean!")
    except Exception as e:
        print(f"   ❌ Failed with gradients: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_full_pipeline()

