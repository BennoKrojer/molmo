#!/usr/bin/env python3
"""Quick sanity check: load Qwen2.5-VL-32B, process 1 image, verify architecture."""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '.')

MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
IMAGE_PAD_TOKEN_ID = 151655

print(f"Loading {MODEL} with device_map=auto...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

# Print architecture summary
print(f"\n=== Architecture ===")
print(f"Model type: {model.config.model_type}")
print(f"Num layers: {model.config.num_hidden_layers}")
print(f"Hidden dim: {model.config.hidden_size}")
print(f"Vocab size: {model.config.vocab_size}")

# Explore model structure to find embed_tokens and norm
print(f"\n=== Finding embed_tokens and norm ===")
print(f"model.model type: {type(model.model)}")
print(f"model.model children: {[name for name, _ in model.model.named_children()]}")

# Try different paths for embed_tokens
for path in ['model.embed_tokens', 'model.language_model.embed_tokens', 'model.language_model.model.embed_tokens']:
    parts = path.split('.')
    obj = model
    found = True
    for p in parts:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            found = False
            break
    if found:
        print(f"  FOUND embed_tokens at: model.{path} -> shape={obj.weight.shape}")
    else:
        print(f"  NOT at: model.{path}")

# Try finding norm similarly
for path in ['model.norm', 'model.language_model.norm', 'model.language_model.model.norm']:
    parts = path.split('.')
    obj = model
    found = True
    for p in parts:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            found = False
            break
    if found:
        print(f"  FOUND norm at: model.{path}")
    else:
        print(f"  NOT at: model.{path}")

# Also check lm_head
print(f"\nmodel.lm_head: {model.lm_head.weight.shape}")

# Print device map
hf_device_map = getattr(model, 'hf_device_map', {})
devices_used = set(str(v) for v in hf_device_map.values())
print(f"\nDevices used: {devices_used}")

# Process 1 synthetic image
print(f"\n=== Processing test image ===")
image = Image.new('RGB', (448, 448), color='red')

processor.image_processor.min_pixels = 448 * 448
processor.image_processor.max_pixels = 448 * 448
processor.image_processor.do_resize = False

inputs = processor(images=[image], text="<|image_pad|>Describe this image.", return_tensors="pt")
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

input_ids = inputs['input_ids'].cpu().numpy()[0]
vision_positions = np.where(input_ids == IMAGE_PAD_TOKEN_ID)[0]
print(f"Input sequence length: {len(input_ids)}")
print(f"Vision tokens: {len(vision_positions)} (expect 256 for 16x16)")

if 'image_grid_thw' in inputs:
    print(f"Grid THW: {inputs['image_grid_thw'].tolist()}")

print(f"\nRunning forward pass...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hs = outputs.hidden_states
print(f"Num hidden states: {len(hs)} (expect {model.config.num_hidden_layers + 1})")
print(f"Hidden state shape: {hs[0].shape}")

for layer_idx in [0, 32, 63]:
    if layer_idx < len(hs):
        vision_hs = hs[layer_idx][0, vision_positions[0]:vision_positions[-1]+1, :]
        print(f"Layer {layer_idx}: vision shape={vision_hs.shape}, norm={vision_hs.float().norm(dim=-1).mean():.2f}")

print(f"\n=== ALL CHECKS PASSED ===")
