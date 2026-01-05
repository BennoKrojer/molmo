#!/usr/bin/env python3
"""
Quick visualization to compare Patchscopes vs LogitLens vs LN-Lens predictions.

Creates an HTML page with side-by-side comparisons for visual inspection.
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from olmo.data.pixmo_datasets import PixMoCap


def create_comparison_html(patchscopes_dir, logitlens_dir, output_path, num_images=2, layers=None,
                           contextual_dir=None, num_patches=25):
    """
    Create an HTML comparison of Patchscopes vs LogitLens vs LN-Lens (contextual NN).
    """
    patchscopes_dir = Path(patchscopes_dir)
    logitlens_dir = Path(logitlens_dir) if logitlens_dir else None
    contextual_dir = Path(contextual_dir) if contextual_dir else None
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

    # Create HTML with 3-column comparison
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Patchscopes vs LogitLens vs LN-Lens Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        h3 { color: #666; }
        .method-legend {
            display: flex; gap: 30px; margin: 20px 0; padding: 15px;
            background: white; border-radius: 10px;
        }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-color { width: 20px; height: 20px; border-radius: 4px; }
        .image-container {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .image-section { flex: 0 0 400px; }
        .image-section img { max-width: 100%; border: 1px solid #ddd; }
        .table-section { flex: 1; overflow-x: auto; }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
        }
        .comparison-table th, .comparison-table td {
            border: 1px solid #ddd;
            padding: 6px 8px;
            text-align: left;
        }
        .comparison-table th { background: #f5f5f5; font-size: 11px; }
        .method-patchscopes { background: #e3f2fd; }
        .method-logitlens { background: #e8f5e9; }
        .method-lnlens { background: #fff3e0; }
        .layer-section { margin: 20px 0; padding: 15px; background: white; border-radius: 10px; }
        .caption { font-style: italic; color: #666; margin: 10px 0; padding: 10px; background: #fafafa; font-size: 12px; }
        .patch-coord { font-family: monospace; font-weight: bold; color: #333; }
        .token { font-family: monospace; }
        .highlight { background: #ffffcc; }
    </style>
</head>
<body>
<h1>Patchscopes vs LogitLens vs LN-Lens Comparison</h1>

<div class="method-legend">
    <div class="legend-item">
        <div class="legend-color" style="background: #e3f2fd;"></div>
        <strong>Patchscopes</strong>: Identity prompt patching (l→l)
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #e8f5e9;"></div>
        <strong>LogitLens</strong>: Direct ln_f + ff_out projection
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #fff3e0;"></div>
        <strong>LN-Lens</strong>: Contextual nearest neighbors
    </div>
</div>
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

        # Try to load LN-Lens (contextual NN) data
        ctx_data = None
        if contextual_dir:
            # Try different naming conventions
            ctx_patterns = [
                f"contextual_neighbors_visual{layer_idx}_allLayers.json",
                f"contextual_neighbors_layer{layer_idx}_topk5.json",
            ]
            for pattern in ctx_patterns:
                ctx_file = contextual_dir / pattern
                if ctx_file.exists():
                    with open(ctx_file) as f:
                        ctx_data = json.load(f)
                    break

        html_parts.append(f'<h2>Layer {layer_idx}</h2>')

        # Process each image
        for img_result in ps_data['results'][:num_images]:
            img_idx = img_result['image_idx']
            caption = img_result.get('ground_truth_caption', '')[:300]

            # Get corresponding LogitLens result
            ll_result = None
            if ll_data:
                for r in ll_data.get('results', []):
                    if r.get('image_idx') == img_idx:
                        ll_result = r
                        break

            # Get corresponding LN-Lens result
            ctx_result = None
            if ctx_data:
                for r in ctx_data.get('results', []):
                    if r.get('image_idx') == img_idx:
                        ctx_result = r
                        break

            # Get image path and save thumbnail
            example_data = dataset.get(img_idx, np.random)
            image_path = example_data["image"]

            img_output = output_path / f"img_{img_idx}.jpg"
            if not img_output.exists():
                img = Image.open(image_path)
                img.thumbnail((400, 400))
                img.save(img_output, quality=85)

            html_parts.append(f'''
<div class="layer-section">
<h3>Image {img_idx}</h3>
<div class="caption">{caption}...</div>
<div class="image-container">
    <div class="image-section">
        <img src="img_{img_idx}.jpg" alt="Image {img_idx}">
    </div>
    <div class="table-section">
        <table class="comparison-table">
            <tr>
                <th>Patch</th>
                <th class="method-patchscopes">Patchscopes (top-3)</th>
                <th class="method-logitlens">LogitLens (top-3)</th>
                <th class="method-lnlens">LN-Lens (top-3)</th>
            </tr>
''')

            # Get patches and sample more of them
            chunks = img_result.get('chunks', [])
            if chunks:
                patches = chunks[0].get('patches', [])
                grid_size = int(math.sqrt(len(patches)))

                # Sample patches: grid pattern + random
                sample_indices = []
                if grid_size > 0:
                    # Corners
                    sample_indices.extend([0, grid_size-1, (grid_size-1)*grid_size, grid_size*grid_size-1])
                    # Edges (midpoints)
                    mid = grid_size // 2
                    sample_indices.extend([mid, mid*grid_size, mid*grid_size + grid_size-1, (grid_size-1)*grid_size + mid])
                    # Grid pattern (every ~4 cells)
                    step = max(1, grid_size // 5)
                    for r in range(0, grid_size, step):
                        for c in range(0, grid_size, step):
                            sample_indices.append(r * grid_size + c)
                    # Center region
                    center = grid_size // 2
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            idx = (center + dr) * grid_size + (center + dc)
                            if 0 <= idx < len(patches):
                                sample_indices.append(idx)

                sample_indices = sorted(set(i for i in sample_indices if 0 <= i < len(patches)))[:num_patches]

                for patch_idx in sample_indices:
                    patch = patches[patch_idx]
                    row, col = patch['patch_row'], patch['patch_col']

                    # Patchscopes predictions
                    ps_tokens = [p['token'].replace('\n', '\\n').replace('\t', '\\t').strip() or '⎵'
                                 for p in patch.get('top_predictions', [])[:3]]
                    ps_str = ', '.join(f'<span class="token">{t[:12]}</span>' for t in ps_tokens)

                    # LogitLens predictions
                    ll_str = "N/A"
                    if ll_result:
                        ll_chunks = ll_result.get('chunks', [])
                        if ll_chunks:
                            ll_patches = ll_chunks[0].get('patches', [])
                            if patch_idx < len(ll_patches):
                                ll_patch = ll_patches[patch_idx]
                                ll_tokens = [p['token'].replace('\n', '\\n').replace('\t', '\\t').strip() or '⎵'
                                            for p in ll_patch.get('top_predictions', [])[:3]]
                                ll_str = ', '.join(f'<span class="token">{t[:12]}</span>' for t in ll_tokens)

                    # LN-Lens predictions
                    ctx_str = "N/A"
                    if ctx_result:
                        ctx_chunks = ctx_result.get('chunks', [])
                        if ctx_chunks:
                            ctx_patches = ctx_chunks[0].get('patches', [])
                            if patch_idx < len(ctx_patches):
                                ctx_patch = ctx_patches[patch_idx]
                                # Contextual NN has 'nearest_contextual_neighbors' key
                                neighbors = ctx_patch.get('nearest_contextual_neighbors',
                                           ctx_patch.get('top_predictions', []))[:3]
                                if neighbors:
                                    ctx_tokens = [n.get('token_str', n.get('token', '')).replace('\n', '\\n').replace('\t', '\\t').strip() or '⎵'
                                                 for n in neighbors]
                                    ctx_str = ', '.join(f'<span class="token">{t[:12]}</span>' for t in ctx_tokens)

                    html_parts.append(f'''
            <tr>
                <td class="patch-coord">({row},{col})</td>
                <td class="method-patchscopes">{ps_str}</td>
                <td class="method-logitlens">{ll_str}</td>
                <td class="method-lnlens">{ctx_str}</td>
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
    parser = argparse.ArgumentParser(description="Visualize Patchscopes vs LogitLens vs LN-Lens comparison")
    parser.add_argument("--patchscopes-dir", type=str, required=True,
                       help="Directory with Patchscopes results")
    parser.add_argument("--logitlens-dir", type=str, default=None,
                       help="Directory with LogitLens results (optional)")
    parser.add_argument("--contextual-dir", type=str, default=None,
                       help="Directory with LN-Lens/contextual NN results (optional)")
    parser.add_argument("--output-dir", type=str, default="analysis_results/patchscopes_comparison",
                       help="Output directory for visualization")
    parser.add_argument("--num-images", type=int, default=2,
                       help="Number of images to visualize")
    parser.add_argument("--num-patches", type=int, default=25,
                       help="Number of patches to sample per image")
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
        layers,
        args.contextual_dir,
        args.num_patches
    )


if __name__ == "__main__":
    main()
