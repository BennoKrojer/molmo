#!/usr/bin/env python3
"""
Quick visualization to compare Patchscopes vs LogitLens predictions.

Creates an HTML page with side-by-side comparisons for visual inspection.
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from olmo.data.pixmo_datasets import PixMoCap


def create_comparison_html(patchscopes_dir, logitlens_dir, output_path, num_images=2, layers=None):
    """
    Create an HTML comparison of Patchscopes vs LogitLens.
    """
    patchscopes_dir = Path(patchscopes_dir)
    logitlens_dir = Path(logitlens_dir) if logitlens_dir else None
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find available layers
    patchscopes_files = list(patchscopes_dir.glob("patchscopes_identity_layer*_topk*.json"))
    if not patchscopes_files:
        print(f"No Patchscopes files found in {patchscopes_dir}")
        return

    available_layers = sorted(set(
        int(f.stem.split("layer")[1].split("_")[0])
        for f in patchscopes_files
    ))

    if layers:
        available_layers = [l for l in available_layers if l in layers]

    print(f"Available layers: {available_layers}")

    # Load dataset for images
    dataset = PixMoCap(split="validation", mode="captions")

    # Create HTML
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Patchscopes vs LogitLens Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        h3 { color: #666; }
        .image-container {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .image-section { flex: 1; }
        .image-section img { max-width: 100%; border: 1px solid #ddd; }
        .patch-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }
        .patch-cell {
            background: #fff;
            border: 1px solid #ddd;
            padding: 8px;
            border-radius: 5px;
            font-size: 11px;
        }
        .patch-cell.corner { background: #f0f0f0; }
        .patch-header { font-weight: bold; color: #333; margin-bottom: 5px; }
        .patchscopes { color: #2196F3; }
        .logitlens { color: #4CAF50; }
        .token {
            display: inline-block;
            padding: 2px 5px;
            margin: 1px;
            background: #e0e0e0;
            border-radius: 3px;
            font-family: monospace;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .comparison-table th, .comparison-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .comparison-table th { background: #f5f5f5; }
        .method-patchscopes { background: #e3f2fd; }
        .method-logitlens { background: #e8f5e9; }
        .layer-section { margin: 20px 0; padding: 15px; background: white; border-radius: 10px; }
        .caption { font-style: italic; color: #666; margin: 10px 0; padding: 10px; background: #fafafa; }
    </style>
</head>
<body>
<h1>Patchscopes vs LogitLens Comparison</h1>
<p><strong>Patchscopes</strong> (blue): Uses identity prompt "cat->cat; 1135->1135; hello->hello; ?" with l→l patching</p>
<p><strong>LogitLens</strong> (green): Direct projection via ln_f + ff_out</p>
"""]

    # Process each layer
    for layer_idx in available_layers:
        # Load Patchscopes data
        ps_file = patchscopes_dir / f"patchscopes_identity_layer{layer_idx}_topk5.json"
        if not ps_file.exists():
            continue

        with open(ps_file) as f:
            ps_data = json.load(f)

        # Try to load LogitLens data
        ll_data = None
        if logitlens_dir:
            ll_file = logitlens_dir / f"logit_lens_layer{layer_idx}_topk5_multi-gpu.json"
            if ll_file.exists():
                with open(ll_file) as f:
                    ll_data = json.load(f)

        html_parts.append(f'<h2>Layer {layer_idx}</h2>')

        # Process each image
        for img_result in ps_data['results'][:num_images]:
            img_idx = img_result['image_idx']
            caption = img_result.get('ground_truth_caption', '')[:200]

            # Get corresponding LogitLens result
            ll_result = None
            if ll_data:
                for r in ll_data.get('results', []):
                    if r.get('image_idx') == img_idx:
                        ll_result = r
                        break

            # Get image path
            example_data = dataset.get(img_idx, np.random)
            image_path = example_data["image"]

            # Copy image to output
            img_output = output_path / f"img_{img_idx}.jpg"
            if not img_output.exists():
                img = Image.open(image_path)
                img.thumbnail((400, 400))
                img.save(img_output, quality=85)

            html_parts.append(f'''
<div class="layer-section">
<h3>Image {img_idx}</h3>
<div class="caption">Caption: {caption}...</div>
<div class="image-container">
    <div class="image-section">
        <img src="img_{img_idx}.jpg" alt="Image {img_idx}">
    </div>
    <div class="image-section">
        <h4>Patch Predictions (sample)</h4>
        <table class="comparison-table">
            <tr>
                <th>Patch</th>
                <th class="method-patchscopes">Patchscopes</th>
                <th class="method-logitlens">LogitLens</th>
            </tr>
''')

            # Show a sample of patches (corners + center)
            chunks = img_result.get('chunks', [])
            if chunks:
                patches = chunks[0].get('patches', [])
                grid_size = int(math.sqrt(len(patches)))

                # Sample patches: corners and center
                sample_indices = []
                if grid_size > 0:
                    # Corners
                    sample_indices.extend([0, grid_size-1, (grid_size-1)*grid_size, grid_size*grid_size-1])
                    # Center
                    center = grid_size // 2
                    sample_indices.append(center * grid_size + center)
                    # Some middle ones
                    sample_indices.extend([grid_size*2 + 2, grid_size*2 + grid_size-3])

                sample_indices = sorted(set(i for i in sample_indices if 0 <= i < len(patches)))[:8]

                for patch_idx in sample_indices:
                    patch = patches[patch_idx]
                    row, col = patch['patch_row'], patch['patch_col']

                    # Patchscopes predictions
                    ps_tokens = [p['token'].strip() or '⎵' for p in patch.get('top_predictions', [])[:3]]
                    ps_str = ', '.join(f'"{t}"' for t in ps_tokens)

                    # LogitLens predictions
                    ll_str = "N/A"
                    if ll_result:
                        ll_chunks = ll_result.get('chunks', [])
                        if ll_chunks:
                            ll_patches = ll_chunks[0].get('patches', [])
                            if patch_idx < len(ll_patches):
                                ll_patch = ll_patches[patch_idx]
                                ll_tokens = [p['token'].strip() or '⎵' for p in ll_patch.get('top_predictions', [])[:3]]
                                ll_str = ', '.join(f'"{t}"' for t in ll_tokens)

                    html_parts.append(f'''
            <tr>
                <td>({row},{col})</td>
                <td class="method-patchscopes">{ps_str}</td>
                <td class="method-logitlens">{ll_str}</td>
            </tr>
''')

            html_parts.append('''
        </table>
    </div>
</div>
</div>
''')

    html_parts.append('</body></html>')

    # Write HTML
    html_file = output_path / "comparison.html"
    with open(html_file, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"✓ Comparison saved to: {html_file}")
    return html_file


def main():
    parser = argparse.ArgumentParser(description="Visualize Patchscopes vs LogitLens comparison")
    parser.add_argument("--patchscopes-dir", type=str, required=True,
                       help="Directory with Patchscopes results")
    parser.add_argument("--logitlens-dir", type=str, default=None,
                       help="Directory with LogitLens results (optional)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/patchscopes_comparison",
                       help="Output directory for visualization")
    parser.add_argument("--num-images", type=int, default=2,
                       help="Number of images to visualize")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layers to show (default: all)")
    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    create_comparison_html(
        args.patchscopes_dir,
        args.logitlens_dir,
        args.output_dir,
        args.num_images,
        layers
    )


if __name__ == "__main__":
    main()
