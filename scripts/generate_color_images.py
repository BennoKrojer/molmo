import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def main():
    parser = argparse.ArgumentParser(description="Generate solid color images for single-token color names.")
    parser.add_argument("--output-dir", type=str, default="molmo_data/color_images",
                        help="Directory to save color images (default: molmo_data/color_images)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer...")
    model_name = "Qwen/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    log.info("Fetching XKCD color names from matplotlib...")
    xkcd_colors = mcolors.XKCD_COLORS  # dict: {"xkcd:color name": "#hex"}

    kept = 0
    skipped = 0

    for full_name, hex_value in xkcd_colors.items():
        color_name = full_name.replace("xkcd:", "").lower()
        tokens = tokenizer.encode(color_name, add_special_tokens=False)

        if len(tokens) == 1:
            token_id = tokens[0]
            rgb = hex_to_rgb(hex_value)
            img = Image.new("RGB", (224, 224), color=rgb)
            filename = f"color_{token_id}_{color_name.replace(' ', '_')}.png"
            img.save(output_dir / filename)
            kept += 1
            log.info(f"Saved {color_name} color images")
        else:
            log.info(f"Skipping '{color_name}': tokenized as {tokens}")
            skipped += 1

    log.info(f"✅ Saved {kept} color images")
    log.info(f"❌ Skipped {skipped} due to multi-token names")

if __name__ == "__main__":
    main()
