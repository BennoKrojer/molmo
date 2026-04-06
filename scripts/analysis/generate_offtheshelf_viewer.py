#!/usr/bin/env python3
"""
Generate viewer for off-the-shelf VLMs (Molmo-7B-D, LLaVA-1.5-7B).

Reuses generate_ablation_viewers.py infrastructure with custom output directory
and a standalone index page.

Usage:
    # Quick test (5 images)
    python scripts/analysis/generate_offtheshelf_viewer.py \
        --output-dir analysis_results/offtheshelf_viewer --num-images 5

    # Full viewer
    python scripts/analysis/generate_offtheshelf_viewer.py \
        --output-dir analysis_results/offtheshelf_viewer --num-images 100
"""

import json
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Off-the-shelf model IDs (must match entries in viewer_models.json)
OFFTHESHELF_IDS = ["molmo-7b-d", "llava-1.5-7b", "qwen2.5-vl-32b", "llava-next-34b"]


def create_offtheshelf_index(output_dir: Path, model_status: list) -> None:
    """Create a standalone index.html for the off-the-shelf viewer."""

    cards_html = ""
    for entry in model_status:
        name = entry["name"]
        checkpoint = entry["checkpoint"]
        has_viewer = entry["has_viewer"]
        image_count = entry.get("image_count", 0)
        nn_layers = entry.get("nn_layers", 0)
        logit_layers = entry.get("logit_layers", 0)
        ctx_layers = entry.get("ctx_layers", 0)

        if has_viewer:
            status_color = "#e8f5e9"
            link = f'<a href="ablations/{checkpoint}/index.html" class="model-link">View Results ({image_count} images)</a>'
        else:
            status_color = "#fff3e0"
            link = '<span style="color:#999;">No data available</span>'

        cards_html += f'''
            <div class="model-card" style="background-color: {status_color};">
                <h3>{name}</h3>
                <div class="stats-row">
                    <span>EmbeddingLens: {nn_layers} layers</span>
                    <span>LogitLens: {logit_layers} layers</span>
                    <span>LatentLens: {ctx_layers} layers</span>
                </div>
                {link}
            </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Off-the-Shelf VLM Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .info {{
            background-color: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin-bottom: 30px;
        }}
        .info h3 {{ margin-top: 0; color: #1976d2; }}
        .model-card {{
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        .model-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        .model-card h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .stats-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 14px;
            color: #666;
        }}
        .model-link {{
            display: inline-block;
            padding: 8px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
        }}
        .model-link:hover {{
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Off-the-Shelf VLM Analysis</h1>
        <div class="subtitle">Interpreting visual tokens in pre-trained VLMs</div>

        <div class="info">
            <h3>About</h3>
            <p>This viewer shows EmbeddingLens, LogitLens, and LatentLens analysis for off-the-shelf
            (pre-trained, not connector-only) VLMs. These models were trained end-to-end by their
            respective teams, unlike the controlled connector-only models in the main 3x3 grid.</p>
            <ul>
                <li><strong>Molmo-7B-D:</strong> Qwen2 backbone, multi-crop ViT. Base crop: 12x12 = 144 tokens.</li>
                <li><strong>LLaVA-1.5-7B:</strong> Vicuna backbone, CLIP ViT-L/14-336. Grid: 24x24 = 576 tokens.</li>
            </ul>
        </div>

        {cards_html}
    </div>
</body>
</html>'''

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "index.html", 'w') as f:
        f.write(html)
    log.info(f"Created index: {output_dir / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description="Generate off-the-shelf VLM viewer")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for the viewer")
    parser.add_argument("--num-images", type=int, default=10,
                       help="Number of images to generate")
    parser.add_argument("--config", default=str(Path(__file__).parent / "viewer_models.json"),
                       help="Path to viewer_models.json")
    parser.add_argument("--split", default="validation",
                       help="Dataset split")
    parser.add_argument("--force", action="store_true",
                       help="Force regenerate even if exists")

    args = parser.parse_args()

    # Import heavy dependencies only after arg parsing
    from generate_ablation_viewers import (
        get_ablation_data_paths,
        load_ablation_data,
        create_ablation_model_index,
        create_image_viewer,
        validate_viewer_output,
    )
    from viewer_lib import create_preprocessor, preprocess_center_crop_square
    import numpy as np
    from PIL import Image

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    ablations = config.get("ablations", [])
    offtheshelf = [a for a in ablations if a["id"] in OFFTHESHELF_IDS]

    if not offtheshelf:
        log.error(f"No off-the-shelf models found in config. Expected IDs: {OFFTHESHELF_IDS}")
        sys.exit(1)

    log.info(f"Will generate viewers for {len(offtheshelf)} off-the-shelf model(s)")

    # Load dataset
    log.info("Loading PixMoCap dataset...")
    from olmo.data.pixmo_datasets import PixMoCap
    dataset = PixMoCap(split=args.split, mode="captions")
    log.info(f"Dataset loaded: {len(dataset)} examples")

    base_dir = Path("analysis_results")
    model_status = []

    for ablation in offtheshelf:
        abl_name = ablation["name"]
        checkpoint = ablation["checkpoint"]

        log.info(f"\n{'='*60}")
        log.info(f"Processing: {abl_name}")
        log.info(f"{'='*60}")

        # Check if already exists
        model_dir = args.output_dir / "ablations" / checkpoint
        if model_dir.exists() and not args.force:
            existing = list(model_dir.glob("image_*.html"))
            if len(existing) >= args.num_images:
                log.info(f"  Skipping (already has {len(existing)} images, use --force to regenerate)")

                # Still collect status for index
                data_paths = get_ablation_data_paths(ablation, base_dir)
                from generate_ablation_viewers import find_analysis_files
                nn_files = find_analysis_files(base_dir, "nn", data_paths.get("nn", ""))
                logit_files = find_analysis_files(base_dir, "logitlens", data_paths.get("logitlens", ""))
                ctx_files = find_analysis_files(base_dir, "contextual", data_paths.get("contextual", ""))

                model_status.append({
                    "name": abl_name,
                    "checkpoint": checkpoint,
                    "has_viewer": True,
                    "image_count": len(existing),
                    "nn_layers": len(nn_files),
                    "logit_layers": len(logit_files),
                    "ctx_layers": len(ctx_files),
                })
                continue

        # Get data paths
        data_paths = get_ablation_data_paths(ablation, base_dir)

        # Load data
        log.info("  Loading analysis data...")
        all_data = load_ablation_data(data_paths, base_dir, args.num_images, args.split)

        # Get available layers (excluding _format_* metadata keys)
        available_layers = {
            "nn": sorted([k for k in all_data["nn"].keys() if not str(k).startswith("_format")]),
            "logitlens": sorted([k for k in all_data["logitlens"].keys() if not str(k).startswith("_format")]),
            "contextual": sorted([k for k in all_data["contextual"].keys() if not str(k).startswith("_format")]),
        }

        total_layers = sum(len(v) for v in available_layers.values())
        if total_layers == 0:
            log.warning(f"  No data found for {abl_name}, skipping")
            model_status.append({
                "name": abl_name, "checkpoint": checkpoint,
                "has_viewer": False, "image_count": 0,
                "nn_layers": 0, "logit_layers": 0, "ctx_layers": 0,
            })
            continue

        log.info(f"  Found: NN={len(available_layers['nn'])}, LogitLens={len(available_layers['logitlens'])}, Contextual={len(available_layers['contextual'])}")

        # Determine preprocessing mode
        preprocessing_mode = ablation.get("preprocessing", None)

        # No Molmo checkpoint for off-the-shelf models
        preprocessor = None
        if preprocessing_mode:
            log.info(f"  Using preprocessing mode: {preprocessing_mode}")

        # Create model index
        create_ablation_model_index(args.output_dir, ablation, args.num_images, available_layers)

        # Fix breadcrumb in model index to point to our index, not the main viewer
        # (create_ablation_model_index uses ../../index.html which is correct for
        #  offtheshelf_viewer/ablations/checkpoint/index.html -> offtheshelf_viewer/index.html)

        # Create image viewers
        log.info(f"  Creating image viewers...")
        success = 0
        t0 = time.time()

        for img_idx in range(args.num_images):
            if create_image_viewer(
                args.output_dir, ablation, img_idx, all_data,
                dataset, args.split, preprocessor, preprocessing_mode
            ):
                success += 1

            if (img_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                rate = (img_idx + 1) / elapsed
                log.info(f"    Progress: {img_idx + 1}/{args.num_images} ({rate:.1f} img/s)")

        log.info(f"  Created {success}/{args.num_images} image viewers")

        # Validate first output
        first_html = args.output_dir / "ablations" / checkpoint / "image_0000.html"
        if first_html.exists():
            validate_viewer_output(
                first_html,
                len(available_layers['nn']),
                len(available_layers['logitlens']),
                len(available_layers['contextual'])
            )

        model_status.append({
            "name": abl_name,
            "checkpoint": checkpoint,
            "has_viewer": success > 0,
            "image_count": success,
            "nn_layers": len(available_layers["nn"]),
            "logit_layers": len(available_layers["logitlens"]),
            "ctx_layers": len(available_layers["contextual"]),
        })

    # Create top-level index
    create_offtheshelf_index(args.output_dir, model_status)

    log.info(f"\n{'='*60}")
    log.info("DONE!")
    log.info(f"View at: {args.output_dir.resolve()}/index.html")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
