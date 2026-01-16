#!/usr/bin/env python3
"""
Generate captions from Qwen2-VL-7B-Instruct for PixMoCap validation images.

This script generates captions in the same JSON format as minimal_val_captions.py,
allowing direct comparison with our trained models using the same GPT-4o judge evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qwen2_vl/generate_captions.py \
        --num-images 300 \
        --output-dir analysis_results/captions/Qwen_Qwen2-VL-7B-Instruct
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Import PixMoCap dataset
from olmo.data.pixmo_datasets import PixMoCap


def generate_caption(model, processor, image, device, max_new_tokens=256):
    """Generate a caption for the given image using Qwen2-VL."""

    # Simple conversation format for Qwen2-VL
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    # Process inputs (directly passing PIL image)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate caption
    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic generation
            )

    # Decode output (remove input tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Generate captions from Qwen2-VL")
    parser.add_argument("--num-images", type=int, default=300,
                       help="Number of validation images to process")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Max tokens to generate per caption")
    parser.add_argument("--output-dir", type=str,
                       default="analysis_results/captions/Qwen_Qwen2-VL-7B-Instruct",
                       help="Output directory for captions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print("Loading Qwen2-VL-7B-Instruct...")
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Constrain image resolution to avoid OOM on large images
    # Qwen2-VL uses dynamic resolution - larger images = more tokens = more memory
    # 512x512 = 262144 pixels is a reasonable max for single GPU
    max_resolution = 512
    max_pixels = max_resolution * max_resolution
    processor.image_processor.min_pixels = 256 * 256  # Min 256x256
    processor.image_processor.max_pixels = max_pixels
    print(f"Image resolution constrained to max {max_resolution}x{max_resolution}")

    model.eval()
    print("Model loaded successfully")

    # Load validation dataset
    print("Loading PixMoCap validation dataset...")
    val_dataset = PixMoCap(split="validation", mode="captions")
    print(f"Dataset loaded with {len(val_dataset)} images")

    # Generate captions
    outputs = []
    num_images = min(args.num_images, len(val_dataset))

    print(f"\nGenerating captions for {num_images} images...")
    for i in tqdm(range(num_images)):
        rng = np.random.RandomState(42 + i)  # Same seed as minimal_val_captions.py
        example_data = val_dataset.get(i, rng)

        # Load image
        image_path = example_data["image"]
        image = Image.open(image_path).convert("RGB")

        # Generate caption
        caption = generate_caption(
            model, processor, image, device,
            max_new_tokens=args.max_tokens
        )

        outputs.append({
            "image_idx": i,
            "generated_output": caption,
            "ground_truth": example_data.get("caption", ""),
        })

    # Save results in same format as minimal_val_captions.py
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated_captions.json"

    result = {
        "checkpoint": model_name,
        "num_images": num_images,
        "max_tokens": args.max_tokens,
        "split": "validation",
        "task_type": "caption",
        "dataset": "pixmo_cap",
        "outputs": outputs
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n Saved {len(outputs)} captions to {output_file}")


if __name__ == "__main__":
    main()
