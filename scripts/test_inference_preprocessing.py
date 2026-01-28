"""Test that inference preprocessing matches training format."""
import numpy as np
from olmo.data.pixmo_datasets import PixMoPointsLeftRight
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig

# Load config
config_path = "configs/rest/baseline_pixmo-leftright_olmo-7b_vit-l-14-336.yaml"
model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

print(f"Config system_prompt_kind: {model_config.system_prompt_kind}")
print(f"Config prompt_type: {model_config.prompt_type}")

# Create preprocessor for inference
preprocessor_inf = build_mm_preprocessor(
    model_config,
    for_inference=True,
    shuffle_messages=False,
    is_training=False,
    require_image_features=True
)

print(f"\nInference preprocessor:")
print(f"  system_prompt: {preprocessor_inf.formater.system_prompt}")
print(f"  prompt_templates: {preprocessor_inf.formater.prompt_templates}")

# Get a real example
dataset = PixMoPointsLeftRight(split="validation", kind="basic")
rng = np.random.RandomState(42)
example_data = dataset.get(0, rng)

print(f"\nExample data:")
print(f"  Image: {example_data['image']}")
print(f"  Messages: {len(example_data['message_list'])}")
first_msg = example_data['message_list'][0]
print(f"  First message:")
print(f"    label: {first_msg['label']}")
print(f"    position: {first_msg['position']}")
print(f"    style: {first_msg['style']}")

# Process like in minimal_val_captions.py
example = {
    "image": example_data["image"],
    "message_list": [first_msg]
}

print(f"\nProcessing example...")
processed = preprocessor_inf(example, rng=rng)

print(f"\nProcessed:")
print(f"  input_tokens length: {len(processed['input_tokens'])}")

# Decode to see what's being fed to model
tokenizer = preprocessor_inf.tokenizer
input_text = tokenizer.decode(processed['input_tokens'])

print(f"\n Input text (first 500 chars):")
print(f"  {repr(input_text[:500])}")

# Check for the expected format
if "left_right" in input_text.lower():
    print(f"\n✓ Input contains 'left_right' style prefix")
else:
    print(f"\n✗ WARNING: Input does NOT contain 'left_right' style prefix!")

print(f"\nLast 100 chars of input:")
print(f"  {repr(input_text[-100:])}")

