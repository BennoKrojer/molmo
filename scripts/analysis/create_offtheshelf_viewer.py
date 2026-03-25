#!/usr/bin/env python3
"""
Create a quick HTML viewer for off-the-shelf VLM analysis results.

Generates a self-contained HTML page showing EmbeddingLens / LogitLens / LatentLens
token overlays on images for Molmo-7B-D and LLaVA-1.5-7B.

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python scripts/analysis/create_offtheshelf_viewer.py \
        --num-images 10 --output-dir analysis_results/offtheshelf_viewer
"""

import argparse
import base64
import io
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAVE_PIXMOCAP = True
except ImportError:
    HAVE_PIXMOCAP = False


MODEL_CONFIGS = {
    "molmo_7b": {
        "display_name": "Molmo-7B-D",
        "nn_dir": "analysis_results/nearest_neighbors/molmo_7b/allenai_Molmo-7B-D-0924",
        "ll_dir": "analysis_results/logit_lens/molmo_7b/allenai_Molmo-7B-D-0924",
        "ctx_dir": "analysis_results/contextual_nearest_neighbors/molmo_7b/allenai_Molmo-7B-D-0924",
        "grid_h": 12, "grid_w": 12,
        "layers": [0, 1, 2, 4, 8, 16, 24, 26, 27],
    },
    "llava_1_5": {
        "display_name": "LLaVA-1.5-7B",
        "nn_dir": "analysis_results/nearest_neighbors/llava_1_5/llava-hf_llava-1.5-7b-hf",
        "ll_dir": "analysis_results/logit_lens/llava_1_5/llava-hf_llava-1.5-7b-hf",
        "ctx_dir": "analysis_results/contextual_nearest_neighbors/llava_1_5/llava-hf_llava-1.5-7b-hf",
        "grid_h": 24, "grid_w": 24,
        "layers": [0, 1, 2, 4, 8, 16, 24, 30, 31],
    },
}


def load_results(results_dir, layer, method):
    """Load per-layer JSON results."""
    d = Path(results_dir)
    if method == "nn":
        f = d / f"nearest_neighbors_layer{layer}_topk5.json"
    elif method == "logitlens":
        f = d / f"logit_lens_layer{layer}_topk5.json"
    elif method == "contextual":
        f = d / f"contextual_neighbors_visual{layer}_allLayers.json"
    else:
        return None

    if not f.exists():
        return None
    try:
        with open(f) as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        print(f"  Warning: corrupt JSON, skipping: {f}")
        return None


def get_top1_token(patch, method):
    """Extract top-1 token string from a patch result."""
    if method == "nn":
        nns = patch.get("nearest_neighbors", [])
        return nns[0]["token"].strip() if nns else ""
    elif method == "logitlens":
        preds = patch.get("top_predictions", [])
        return preds[0]["token"].strip() if preds else ""
    elif method == "contextual":
        nns = patch.get("nearest_contextual_neighbors", [])
        return nns[0]["token_str"].strip() if nns else ""
    return ""


def render_grid_overlay(image, patches, grid_h, grid_w, method, cell_size=40):
    """Render token labels over image grid."""
    display_size = cell_size * max(grid_h, grid_w)
    img = image.copy().resize((display_size, display_size), Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    cell_h = display_size / grid_h
    cell_w = display_size / grid_w

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(8, cell_size // 5))
    except (OSError, IOError):
        font = ImageFont.load_default()

    for patch in patches:
        row = patch["patch_row"]
        col = patch["patch_col"]
        token = get_top1_token(patch, method)
        if not token:
            continue

        x = col * cell_w
        y = row * cell_h

        # Semi-transparent background
        bbox = draw.textbbox((x + 2, y + 2), token[:6], font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 160))
        draw.text((x + 2, y + 2), token[:6], fill="white", font=font)

    return img


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description="Off-the-shelf VLM viewer")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output-dir", type=str,
                       default="analysis_results/offtheshelf_viewer")
    parser.add_argument("--layers", type=str, default=None,
                       help="Override layers to show (comma-separated). Default: use model-specific layers")
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    if not HAVE_PIXMOCAP:
        print("ERROR: PixMoCap required")
        return

    dataset = PixMoCap(split=args.split, mode="captions")
    print(f"Loaded PixMoCap {args.split}")

    images = []
    captions = []
    for i in range(args.num_images):
        ex = dataset.get(i, np.random)
        img = Image.open(ex["image"]).convert("RGB")
        # Center-crop to square
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
        images.append(img)
        cap = ""
        if "message_list" in ex and len(ex["message_list"]) > 0:
            cap = ex["message_list"][0].get("text", "")
        captions.append(cap)

    print(f"Loaded {len(images)} images")

    # Build HTML
    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Off-the-shelf VLM Analysis: Molmo-7B-D & LLaVA-1.5-7B</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
h1 { color: #e94560; }
h2 { color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 4px; }
h3 { color: #16213e; background: #0f3460; color: #eee; padding: 6px 12px; border-radius: 4px; margin-top: 30px; }
.image-section { margin: 20px 0; border: 1px solid #333; padding: 15px; border-radius: 8px; background: #16213e; }
.grid-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }
.grid-cell { text-align: center; }
.grid-cell img { border: 1px solid #444; border-radius: 4px; }
.grid-cell .label { font-size: 11px; color: #aaa; margin-top: 2px; }
.caption { font-style: italic; color: #888; margin: 5px 0 10px 0; font-size: 13px; }
.method-label { font-weight: bold; color: #e94560; margin: 10px 0 5px 0; }
.nav { position: sticky; top: 0; background: #1a1a2e; padding: 10px; z-index: 100; border-bottom: 2px solid #e94560; }
.nav a { color: #e94560; margin: 0 10px; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
</style></head><body>
<div class="nav">
<strong>Off-the-shelf VLM Analysis</strong> |
"""]

    # Nav links
    for model_key, cfg in MODEL_CONFIGS.items():
        html_parts.append(f'<a href="#{model_key}">{cfg["display_name"]}</a> |')
    html_parts.append("</div>\n")

    methods_available = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        html_parts.append(f'<h2 id="{model_key}">{cfg["display_name"]}</h2>\n')
        html_parts.append(f'<p>Grid: {cfg["grid_h"]}x{cfg["grid_w"]} = {cfg["grid_h"]*cfg["grid_w"]} vision tokens</p>\n')

        layers = [int(l) for l in args.layers.split(",")] if args.layers else cfg["layers"]
        # Pick 3 representative layers for the viewer (early, mid, late)
        show_layers = [layers[0], layers[len(layers)//2], layers[-1]]

        for img_idx in range(args.num_images):
            html_parts.append(f'<div class="image-section">\n')
            html_parts.append(f'<h3>Image {img_idx}</h3>\n')
            html_parts.append(f'<p class="caption">{captions[img_idx][:120]}</p>\n')

            # Show original image
            thumb = images[img_idx].resize((200, 200), Image.LANCZOS)
            b64 = image_to_base64(thumb)
            html_parts.append(f'<img src="data:image/png;base64,{b64}" style="margin-bottom:10px;border-radius:4px;">\n')

            for method, method_label, result_dir_key in [
                ("nn", "EmbeddingLens", "nn_dir"),
                ("logitlens", "LogitLens", "ll_dir"),
                ("contextual", "LatentLens", "ctx_dir"),
            ]:
                result_dir = cfg[result_dir_key]
                html_parts.append(f'<div class="method-label">{method_label}</div>\n')
                html_parts.append('<div class="grid-row">\n')

                for layer in show_layers:
                    data = load_results(result_dir, layer, method)
                    if data is None:
                        html_parts.append(f'<div class="grid-cell"><div class="label">Layer {layer}: N/A</div></div>\n')
                        continue

                    # Find this image's results
                    results_key = "results"
                    img_results = None
                    for r in data.get(results_key, []):
                        if r.get("image_idx") == img_idx:
                            img_results = r
                            break

                    if img_results is None:
                        html_parts.append(f'<div class="grid-cell"><div class="label">Layer {layer}: no data</div></div>\n')
                        continue

                    patches = img_results.get("patches", [])
                    overlay = render_grid_overlay(
                        images[img_idx], patches, cfg["grid_h"], cfg["grid_w"],
                        method, cell_size=32
                    )
                    b64 = image_to_base64(overlay)
                    html_parts.append(f'<div class="grid-cell">')
                    html_parts.append(f'<img src="data:image/png;base64,{b64}" width="280">')
                    html_parts.append(f'<div class="label">Layer {layer}</div></div>\n')

                html_parts.append('</div>\n')

            html_parts.append('</div>\n')

    html_parts.append("</body></html>")

    out_file = output_dir / "index.html"
    with open(out_file, "w") as f:
        f.write("".join(html_parts))

    print(f"\nViewer generated: {out_file.resolve()}")
    print(f"Open in browser to inspect token-to-image alignment")


if __name__ == "__main__":
    main()
