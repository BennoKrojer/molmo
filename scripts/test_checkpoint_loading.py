"""Quick test to see if checkpoint works with correct loading."""
import torch
import numpy as np
from olmo.model import Molmo
from olmo.config import TrainConfig
from olmo.data.pixmo_datasets import PixMoPointsLeftRight
from olmo.data import build_mm_preprocessor

checkpoint_path = "/mnt/research/scratch/bkroje/molmo_data/checkpoints/train_mlp-only_pixmo_leftright_olmo-7b_vit-l-14-336/step12000-unsharded"

print("="*80)
print("Testing checkpoint loading with CORRECT method")
print("="*80)

# Load config
cfg = TrainConfig.load(f"{checkpoint_path}/config.yaml")
cfg.model.init_device = None

# Initialize model
print("\n1. Initializing model...")
model = Molmo(cfg.model)

# CRITICAL: Load pretrained weights FIRST (since LLM/ViT were frozen)
print("\n2. Loading pretrained LLM + ViT weights...")
model.reset_with_pretrained_weights()

# THEN load checkpoint (which will update the trained connector)
print("\n3. Loading checkpoint weights (will update trained connector)...")
checkpoint_weights = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")
model.load_state_dict(checkpoint_weights, strict=False)
print(f"   Loaded {len(checkpoint_weights)} parameters")

model.eval()
model = model.to("cuda:0")

# Create preprocessor
print("\n4. Creating preprocessor...")
preprocessor = build_mm_preprocessor(
    cfg.model,
    for_inference=True,
    shuffle_messages=False,
    is_training=False,
    require_image_features=True
)

# Load validation dataset
print("\n5. Loading validation dataset...")
dataset = PixMoPointsLeftRight(split="validation", kind="basic")

# Test on 3 examples
print("\n6. Testing on 3 examples...")
print("="*80)

correct = 0
total = 0

for i in range(3):
    example_data = dataset.get(i, np.random.RandomState(42))
    first_msg = example_data['message_list'][0]
    
    example = {
        "image": example_data["image"],
        "message_list": [first_msg]
    }
    
    batch = preprocessor(example, rng=np.random.RandomState(42))
    
    with torch.inference_mode():
        input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to("cuda:0")
        images = torch.tensor(batch["images"]).unsqueeze(0).to("cuda:0")
        image_masks = torch.tensor(batch.get("image_masks")).unsqueeze(0).to("cuda:0") if batch.get("image_masks") is not None else None
        image_input_idx = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to("cuda:0") if batch.get("image_input_idx") is not None else None
        
        output = model.generate(
            input_ids=input_ids,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            max_steps=30,
            min_steps=4,
            is_distributed=False
        )
        token_ids = output.token_ids[:, 0].detach().cpu().numpy()
        generated_text = preprocessor.tokenizer.decode(token_ids[0])
    
    print(f"\nExample {i}:")
    print(f"  Label: {first_msg['label']}")
    print(f"  Ground truth: {first_msg['position']}")
    print(f"  Generated: {generated_text}")
    
    # Check if correct
    if first_msg['position'] in generated_text.lower():
        print(f"  ✓ CORRECT!")
        correct += 1
    else:
        print(f"  ✗ WRONG")
    
    total += 1

print("\n" + "="*80)
print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
print("="*80)

if correct == 0:
    print("\n⚠️  Model is NOT working - all predictions wrong!")
    print("This means either:")
    print("  1. The training didn't actually work")
    print("  2. There's still a preprocessing mismatch")
elif correct == total:
    print("\n✓ Model is working perfectly!")
else:
    print(f"\n⚠️  Model is partially working ({100*correct/total:.1f}% correct)")

