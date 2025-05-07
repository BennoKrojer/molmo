"""Run inference on a single example with a fixed prompt and image."""
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path

log = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python single_example_inference.py <checkpoint_path> <image_path> <prompt>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2]
    prompt = sys.argv[3]

    # Load model
    log.info(f"Loading model from {checkpoint_path}")
    model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    # Load image
    log.info(f"Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Create example
    example = {
        "image": image,
        "messages": {
                "messages": [prompt],
                "style": "long_caption"
        }
    }

    # Create preprocessor
    model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # Override system prompt kind to avoid length conditioning
    model_config.system_prompt_kind = "style"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    # Preprocess example
    batch = preprocessor(example, rng=np.random)

    # Run inference
    log.info("Running inference...")
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            # Clear CUDA cache
            # torch.cuda.empty_cache()
            
            # # First do a forward pass to prevent OOMs
            # model(
            #     input_ids=torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda(),
            #     images=torch.tensor(batch.get("images")).unsqueeze(0).cuda() if batch.get("images") is not None else None,
            #     image_masks=torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None,
            #     image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
            # )

            # # Clear CUDA cache again before generation
            # torch.cuda.empty_cache()

            # Then generate with reduced max_steps
            output = model.generate(
                input_ids=torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda(),
                images=torch.tensor(batch.get("images")).unsqueeze(0).cuda() if batch.get("images") is not None else None,
                image_masks=torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None,
                image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                max_steps=50,  # Reduced from 100 to 50
                is_distributed=False
            )

    # Decode and print output
    token_ids = output.token_ids[:, 0].detach().cpu().numpy()  # beam size of 1
    decoded = preprocessor.tokenizer.decode(token_ids[0])
    print("\nGenerated response:")
    print(decoded)

if __name__ == "__main__":
    main()
