#!/usr/bin/env python3
"""
Add border lines to EXISTING plot images without regenerating them.
This overlays thick black rectangles on the saved PNG images.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path


def add_borders_to_3x6_grid(input_path, output_path):
    """
    Add borders around each Vision|Text pair in the 3x6 histogram grid.
    Creates 3x3 border rectangles grouping each pair.
    """
    # Load existing image
    img = Image.open(input_path)
    img_array = np.array(img)

    fig, ax = plt.subplots(figsize=(img.width/100, img.height/100), dpi=100)
    ax.imshow(img_array)
    ax.axis('off')

    # Image dimensions
    h, w = img_array.shape[:2]

    # The 3x6 grid has specific regions we need to border
    # Approximate positions based on typical matplotlib layout:
    # - Title takes ~5% at top
    # - Each row is ~30% of remaining height
    # - Each Vision|Text pair is ~33% of width

    # Margins (approximate from visual inspection of the plot)
    left_margin = 0.045 * w
    right_margin = 0.99 * w
    top_margin = 0.06 * h  # Below title
    bottom_margin = 0.98 * h

    plot_width = right_margin - left_margin
    plot_height = bottom_margin - top_margin

    cell_width = plot_width / 3  # 3 model columns (each has Vision|Text)
    cell_height = plot_height / 3  # 3 LLM rows

    # Add thick black borders around each cell
    for row in range(3):
        for col in range(3):
            x = left_margin + col * cell_width
            y = top_margin + row * cell_height

            rect = patches.Rectangle(
                (x, y), cell_width, cell_height,
                linewidth=4, edgecolor='black', facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)

    # Also save PDF
    pdf_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.close()
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")
    return pdf_path


def add_borders_to_3x3_grid(input_path, output_path):
    """
    Add borders around each subplot in a 3x3 grid.
    """
    # Load existing image
    img = Image.open(input_path)
    img_array = np.array(img)

    fig, ax = plt.subplots(figsize=(img.width/100, img.height/100), dpi=100)
    ax.imshow(img_array)
    ax.axis('off')

    h, w = img_array.shape[:2]

    # Margins for 3x3 grid
    left_margin = 0.065 * w
    right_margin = 0.98 * w
    top_margin = 0.08 * h
    bottom_margin = 0.97 * h

    plot_width = right_margin - left_margin
    plot_height = bottom_margin - top_margin

    cell_width = plot_width / 3
    cell_height = plot_height / 3

    for row in range(3):
        for col in range(3):
            x = left_margin + col * cell_width
            y = top_margin + row * cell_height

            rect = patches.Rectangle(
                (x, y), cell_width, cell_height,
                linewidth=4, edgecolor='black', facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)

    pdf_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.close()
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    base = Path("paper_plots/paper_figures_output/l2norm_plots")

    print("Adding borders to L2 norm 3x6 histogram grid...")
    pdf1 = add_borders_to_3x6_grid(
        base / "l2norm_3x3_grid_log.png",
        base / "l2norm_3x3_grid_log_bordered.png"
    )

    print("\nAdding borders to max token 3x3 grid...")
    pdf2 = add_borders_to_3x3_grid(
        base / "max_token_embedding_values_3x3.png",
        base / "max_token_embedding_values_3x3_bordered.png"
    )

    print("\nDone! Copy these to paper/figures/:")
    print(f"  {pdf1} -> paper/figures/l2norm_vision_text_distributions.pdf")
    print(f"  {pdf2} -> paper/figures/max_token_embedding_distributions.pdf")
