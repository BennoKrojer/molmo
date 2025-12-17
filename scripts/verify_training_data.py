"""Verify that the training data has the correct format."""
import numpy as np
from olmo.data.pixmo_datasets import PixMoPointsLeftRight
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig

# Load config
config_path = "configs/rest/baseline_pixmo-leftright_olmo-7b_vit-l-14-336.yaml"
model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

# Create preprocessor for TRAINING (not inference)
preprocessor = build_mm_preprocessor(
    model_config,
    for_inference=False,
    shuffle_messages=False,
    is_training=True,
    require_image_features=True
)

print(f"Training preprocessor:")
print(f"  system_prompt: {preprocessor.formater.system_prompt}")
print(f"  prompt_templates: {preprocessor.formater.prompt_templates}")

# Get a real example
dataset = PixMoPointsLeftRight(split="train", kind="basic")
rng = np.random.RandomState(42)
example_data = dataset.get(0, rng)

print(f"\nExample data:")
first_msg = example_data['message_list'][0]
print(f"  Label: {first_msg['label']}")
print(f"  Position: {first_msg['position']}")

# Process for training
print(f"\nProcessing for training...")
processed = preprocessor(example_data, rng=rng)

tokenizer = preprocessor.tokenizer

print(f"\nInput tokens: {len(processed['input_tokens'])}")
print(f"Target tokens: {len(processed['target_tokens'])}")

# Decode input
input_text = tokenizer.decode(processed['input_tokens'])
print(f"\nInput (last 200 chars):")
print(f"  {repr(input_text[-200:])}")

# Decode target
target_text = tokenizer.decode(processed['target_tokens'])
print(f"\nTarget (first 500 chars):")
print(f"  {repr(target_text[:500])}")

# Check if target contains the expected answer
expected_answer = f"{first_msg['label'].capitalize()} is on the {first_msg['position']} of the image."
print(f"\nExpected answer: {expected_answer}")

if "left" in target_text.lower() or "right" in target_text.lower():
    print("✓ Target contains 'left' or 'right'")
    
    if first_msg['position'] in target_text.lower():
        print(f"✓ Target contains the correct position: '{first_msg['position']}'")
    else:
        print(f"✗ WARNING: Target does NOT contain '{first_msg['position']}'")
else:
    print("✗ WARNING: Target does NOT contain 'left' or 'right'")

# Show exactly where the answer starts in the target
if first_msg['label'] in target_text:
    idx = target_text.index(first_msg['label'])
    print(f"\nAnswer starts at position {idx} in target:")
    print(f"  {repr(target_text[idx:idx+100])}")

