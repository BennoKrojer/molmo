"""Check what weights are in the checkpoint and if they look trained."""
import torch

checkpoint_path = "/mnt/research/scratch/bkroje/molmo_data/checkpoints/train_mlp-only_pixmo_leftright_olmo-7b_vit-l-14-336/step12000-unsharded/model.pt"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(f"\nCheckpoint contains {len(checkpoint)} parameters")

# Check for connector weights (these should have been trained)
connector_keys = [k for k in checkpoint.keys() if "image_projector" in k or "vision_backbone" in k]
print(f"\nFound {len(connector_keys)} connector/vision weights")

# Show a sample
print("\nSample connector weights:")
for key in sorted(connector_keys)[:10]:
    tensor = checkpoint[key]
    print(f"  {key}:")
    print(f"    shape: {tensor.shape}")
    print(f"    mean: {tensor.mean().item():.6f}")
    print(f"    std: {tensor.std().item():.6f}")
    print(f"    min: {tensor.min().item():.6f}")
    print(f"    max: {tensor.max().item():.6f}")

# Check if weights look initialized or trained
# Trained weights should have reasonable statistics
print("\nChecking if connector looks trained...")
w1_key = "vision_backbone.image_projector.w1.weight"
if w1_key in checkpoint:
    w1 = checkpoint[w1_key]
    print(f"\n{w1_key}:")
    print(f"  Mean: {w1.mean().item():.6f}")
    print(f"  Std: {w1.std().item():.6f}")
    
    # Check if it's all zeros (not trained) or has reasonable values
    if w1.abs().max() < 1e-6:
        print("  ⚠️  WARNING: Weights are essentially zero!")
    elif w1.std() < 1e-4:
        print("  ⚠️  WARNING: Weights have very low variance!")
    else:
        print("  ✓ Weights look reasonable")

# Check LLM weights (should be frozen, so should match pretrained)
llm_keys = [k for k in checkpoint.keys() if "transformer.blocks" in k or "transformer.ff_out" in k]
print(f"\nFound {len(llm_keys)} LLM weights")

# Check a random LLM weight
if llm_keys:
    sample_key = llm_keys[len(llm_keys)//2]
    tensor = checkpoint[sample_key]
    print(f"\nSample LLM weight ({sample_key}):")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")

