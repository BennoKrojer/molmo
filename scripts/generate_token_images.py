"""Generate images for each token in the Molmo tokenizer."""
import logging
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from olmo.config import ModelConfig
from olmo.model import Molmo
from olmo.util import resource_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate images for each token in the Molmo tokenizer.')
    parser.add_argument('--output-dir', type=str, default='molmo_data/token_images',
                      help='Directory to save token images (default: molmo_data/token_images)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log.info("Loading model...")
    checkpoint_path = "/mnt/research/scratch/bkroje/checkpoints/train_mlp-only_pixmo_points_overlap-and-resize-c2/step10000"
    model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    # Get tokenizer and embedding matrix
    tokenizer = model.config.get_tokenizer()
    vocab_size = tokenizer.vocab_size()
    embedding_matrix = model.transformer.wte.embedding.detach().cpu().numpy()
    
    # Sanity check
    if vocab_size != embedding_matrix.shape[0]:
        log.error(f"Vocabulary size mismatch! Tokenizer has {vocab_size} tokens but embedding matrix has {embedding_matrix.shape[0]} rows")
        sys.exit(1)
    log.info(f"Token count matches embedding matrix size: {vocab_size}")

    # Create a font object
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except IOError:
        # Fallback to default font if Arial is not available
        font = ImageFont.load_default()

    # Generate images for each token
    log.info("Generating token images...")
    for token_id in range(vocab_size):
        # Get token text
        token_text = tokenizer.decode([token_id])
        
        # Create a new image with white background
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw the token text
        draw.text((10, 10), token_text, font=font, fill='black')
        
        # Save the image
        img.save(output_dir / f"token_{token_id}.png")
        
        if (token_id + 1) % 100 == 0:
            log.info(f"Generated {token_id + 1} token images...")

    log.info(f"Successfully generated {vocab_size} token images in {output_dir}")

if __name__ == "__main__":
    main() 