"""Test a single inference to see if the model generates correctly."""
import torch
import numpy as np
from olmo.model import Molmo
from olmo.config import TrainConfig, ModelConfig
from olmo.data.pixmo_datasets import PixMoPointsLeftRight
from olmo.data import build_mm_preprocessor
from olmo.util import resource_path

checkpoint_path = "/mnt/research/scratch/bkroje/molmo_data/checkpoints/train_mlp-only_pixmo_leftright_olmo-7b_vit-l-14-336/step12000-unsharded"

print("Loading model...")
cfg = TrainConfig.load(f"{checkpoint_path}/config.yaml")
cfg.model.init_device = None

model = Molmo(cfg.model)

# Load checkpoint
print("Loading checkpoint weights...")
checkpoint_weights = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")
model.load_state_dict(checkpoint_weights, strict=False)
model.eval()
model = model.to("cuda:0")

print("Creating preprocessor...")
model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
print(f"system_prompt_kind: {model_config.system_prompt_kind}")

preprocessor = build_mm_preprocessor(
    model_config,
    for_inference=True,
    shuffle_messages=False,
    is_training=False,
    require_image_features=True
)

print("\nLoading validation dataset...")
dataset = PixMoPointsLeftRight(split="validation", kind="basic")
rng = np.random.RandomState(42)

# Get an example
example_data = dataset.get(0, rng)
first_msg = example_data['message_list'][0]

print(f"\nExample:")
print(f"  Label: {first_msg['label']}")
print(f"  Position: {first_msg['position']}")

# Create example for inference
example = {
    "image": example_data["image"],
    "message_list": [first_msg]
}

print("\nPreprocessing...")
batch = preprocessor(example, rng=rng)

print(f"Input tokens length: {len(batch['input_tokens'])}")
input_text = preprocessor.tokenizer.decode(batch['input_tokens'])
print(f"Input text (last 100 chars): {repr(input_text[-100:])}")

print("\nGenerating...")
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

print(f"\n" + "="*80)
print(f"RESULT:")
print(f"="*80)
print(f"Label: {first_msg['label']}")
print(f"Ground truth position: {first_msg['position']}")
print(f"Generated text: {generated_text}")
print(f"="*80)

# Check if it contains left or right
if "left" in generated_text.lower():
    print("✓ Generated text contains 'left'")
    if first_msg['position'] == 'left':
        print("✓✓ CORRECT!")
    else:
        print("✗ INCORRECT (should be right)")
elif "right" in generated_text.lower():
    print("✓ Generated text contains 'right'")
    if first_msg['position'] == 'right':
        print("✓✓ CORRECT!")
    else:
        print("✗ INCORRECT (should be left)")
else:
    print("✗ Generated text doesn't contain 'left' or 'right'")

