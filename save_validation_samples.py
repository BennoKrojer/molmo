#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path
from PIL import Image
from olmo.data.pixmo_datasets import PixMoCap

dataset = PixMoCap(split="validation", mode="captions")
output_dir = Path("tmp/validation_samples")
output_dir.mkdir(parents=True, exist_ok=True)
examples = []
for i in range(300):
    ex = dataset.get(i, np.random)
    img = ex["image"] if isinstance(ex["image"], Image.Image) else Image.open(ex["image"])
    img.save(output_dir / f"image_{i:03d}.png")
    caption = ex["message_list"][0].get("text", "") if "message_list" in ex else ""
    image_url = ex.get("metadata", {}).get("image_url", "")
    examples.append({"index": i, "image_url": image_url, "caption": caption, "image_file": f"image_{i:03d}.png"})
    print(f"{i}: {image_url} | {caption[:60]}")
with open(output_dir / "examples.json", "w") as f:
    json.dump(examples, f, indent=2)
