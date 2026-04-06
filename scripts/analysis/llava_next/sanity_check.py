#!/usr/bin/env python3
"""Sanity check for LLaVA-NeXT-34B: load model, process 1 image, verify architecture."""

import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '.')

from preprocessing import MODEL_NAME, IMAGE_TOKEN_ID, NUM_VISION_TOKENS, preprocess_image_llava

print(f"Loading {MODEL_NAME} with device_map=auto...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(f"\n=== Architecture ===")
print(f"Num layers: {model.config.text_config.num_hidden_layers}")
print(f"Hidden dim: {model.config.text_config.hidden_size}")
print(f"Image token ID: {model.config.image_token_index}")

# Pre-resize image to 336x336 to minimize AnyRes tile count
print(f"\n=== Processing test image (336x336) ===")
image = Image.new('RGB', (336, 336), color='red')
prompt = "<image>\nDescribe this image."

inputs = processor(images=image, text=prompt, return_tensors="pt")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")
print(f"image_sizes: {inputs['image_sizes']}")

inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Find image tokens in input_ids (HF already expanded <image> into many tokens)
input_ids = inputs['input_ids'][0].cpu().numpy()
image_positions = np.where(input_ids == IMAGE_TOKEN_ID)[0]
print(f"\n<image> token count in input_ids: {len(image_positions)}")
print(f"Input seq len: {len(input_ids)}")

# Verify contiguous range
if len(image_positions) > 0:
    start = int(image_positions[0])
    end = int(image_positions[-1]) + 1
    contiguous = (end - start) == len(image_positions)
    print(f"Vision token range: [{start}, {end}), contiguous={contiguous}")

print(f"\nRunning forward pass...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hs = outputs.hidden_states
print(f"Num hidden states: {len(hs)} (expect 61 = 60 layers + embed)")
print(f"Hidden state shape: {hs[0].shape}")
print(f"hs seq len: {hs[0].shape[1]}  vs  input_ids len: {len(input_ids)}")

# The first 576 tokens of the vision block are the base thumbnail (24x24 grid)
# The remaining tokens are the AnyRes tile + newlines
print(f"\n=== First 576 vision tokens (base thumbnail 24x24) ===")
base_start = start
base_end = start + NUM_VISION_TOKENS
print(f"Base thumbnail range: [{base_start}, {base_end})")

for layer in [0, 30, 59]:
    if layer < len(hs):
        vs = hs[layer][0, base_start:base_end, :]
        print(f"Layer {layer}: shape={vs.shape}, norm={vs.float().norm(dim=-1).mean():.2f}")

print(f"\n=== DONE — use base thumbnail (first 576 of expansion) for analysis ===")
