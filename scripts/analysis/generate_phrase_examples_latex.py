#!/usr/bin/env python3
"""
Generate preprocessed images and LaTeX code for phrase annotation examples.

This script:
1. Selects 12 diverse examples from the 60 annotated ones
2. Generates preprocessed images as vision encoders see them (with red bbox)
3. Generates LaTeX table code with highlighted phrases

Vision encoder preprocessing:
- CLIP (vit-l-14-336): resize + pad to 336x336 (preserves aspect ratio)
- SigLIP: squash to 384x384
- DINOv2: squash to 336x336
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from olmo.data.pixmo_datasets import PixMoCap


# Vision encoder configs
VISION_CONFIGS = {
    'vit-l-14-336': {'size': 336, 'method': 'resize_pad', 'grid': 24, 'display': 'CLIP'},
    'siglip': {'size': 384, 'method': 'squash', 'grid': 27, 'display': 'SigLIP'},
    'dinov2-large-336': {'size': 336, 'method': 'squash', 'grid': 24, 'display': 'DINOv2'},
}

LLM_DISPLAY = {
    'llama3-8b': 'LLaMA3',
    'olmo-7b': 'OLMo',
    'qwen2-7b': 'Qwen2',
}


def resize_and_pad_pil(image: Image.Image, target_size: int) -> tuple[Image.Image, tuple[int, int, float]]:
    """
    CLIP-style preprocessing: resize preserving aspect ratio, pad to square.
    Returns (processed_image, (offset_x, offset_y, scale)).
    """
    img_w, img_h = image.size
    scale = min(target_size / img_w, target_size / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Pad to square with black background (pad_value=0 in original resize_and_pad)
    padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded.paste(resized, (offset_x, offset_y))

    return padded, (offset_x, offset_y, scale)


def squash_resize_pil(image: Image.Image, target_size: int) -> Image.Image:
    """SigLIP/DINOv2-style preprocessing: squash to square (distorts aspect ratio)."""
    return image.resize((target_size, target_size), Image.Resampling.BILINEAR)


def preprocess_and_draw_bbox(image: Image.Image, vision_encoder: str,
                              patch_row: int, patch_col: int) -> tuple[Image.Image, Image.Image]:
    """
    Apply vision encoder preprocessing and draw red bbox around the target patch.
    Returns both full image and 5x5 patch crop.

    Args:
        image: Original PIL image
        vision_encoder: One of 'vit-l-14-336', 'siglip', 'dinov2-large-336'
        patch_row: Row in the processed patch grid (top-left of 3x3 bbox)
        patch_col: Column in the processed patch grid (top-left of 3x3 bbox)

    Returns:
        (full_image, crop_image): Full preprocessed image and 5x5 patch crop, both with bbox
    """
    # Normalize vision encoder name
    if 'siglip' in vision_encoder.lower():
        vision_key = 'siglip'
    elif 'dinov2' in vision_encoder.lower():
        vision_key = 'dinov2-large-336'
    else:
        vision_key = 'vit-l-14-336'

    config = VISION_CONFIGS[vision_key]
    target_size = config['size']
    method = config['method']
    grid_size = config['grid']

    # Apply +1 offset: stored coords are top-left of 3x3, actual patch is center
    center_row = patch_row + 1
    center_col = patch_col + 1

    # Apply preprocessing
    if method == 'resize_pad':
        processed, _ = resize_and_pad_pil(image, target_size)
    else:
        processed = squash_resize_pil(image, target_size)

    # Calculate bbox position - coordinates are already in processed image's grid
    cell_size = target_size / grid_size
    x1 = int(center_col * cell_size)
    y1 = int(center_row * cell_size)
    x2 = int((center_col + 1) * cell_size)
    y2 = int((center_row + 1) * cell_size)

    # Draw red bbox (3px thick) on full image
    draw = ImageDraw.Draw(processed)
    for i in range(3):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='red')

    # Create 5x5 patch crop centered on target patch
    # Crop region: 2 patches before center, center patch, 2 patches after = 5 patches
    crop_row_start = max(0, center_row - 2)
    crop_col_start = max(0, center_col - 2)
    crop_row_end = min(grid_size, center_row + 3)
    crop_col_end = min(grid_size, center_col + 3)

    crop_x1 = int(crop_col_start * cell_size)
    crop_y1 = int(crop_row_start * cell_size)
    crop_x2 = int(crop_col_end * cell_size)
    crop_y2 = int(crop_row_end * cell_size)

    crop = processed.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return processed, crop


def find_token_bounds(phrase: str, token: str) -> tuple[str, str, str]:
    """
    Find token in phrase and expand to word boundaries.
    Returns (before, match, after) parts of the phrase.
    """
    token_clean = token.strip()
    low_phrase = phrase.lower()
    low_token = token_clean.lower()

    idx = low_phrase.find(low_token)
    if idx == -1:
        # Token not found - can't highlight
        return phrase, "", ""

    start = idx
    end = idx + len(token_clean)

    # Expand to word boundaries (include hyphens as part of words)
    while start > 0 and (phrase[start-1].isalnum() or phrase[start-1] == '-'):
        start -= 1
    while end < len(phrase) and (phrase[end].isalnum() or phrase[end] == '-'):
        end += 1

    return phrase[:start], phrase[start:end], phrase[end:]


def latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_phrase_latex(phrase: str, token: str) -> str:
    """
    Format phrase with yellow-highlighted token for LaTeX.
    Uses \colorbox{yellow!50}{\textbf{word}} syntax.
    """
    before, match, after = find_token_bounds(phrase, token)

    if not match:
        # Can't highlight - return escaped phrase
        return f'``{latex_escape(phrase)}\'\'';

    before_esc = latex_escape(before)
    match_esc = latex_escape(match)
    after_esc = latex_escape(after)

    return f'``{before_esc}\\colorbox{{yellow!50}}{{\\textbf{{{match_esc}}}}}{after_esc}\'\''


def select_diverse_examples(examples: list, n: int = 12) -> list:
    """
    Select n diverse examples covering different models and outcomes.
    """
    # Filter to examples with complete annotations
    valid = [ex for ex in examples if ex.get('better_than_no_context') in ['yes', 'neutral', 'no']
             and ex.get('better_than_random_context') in ['yes', 'neutral', 'no']]

    print(f"Found {len(valid)} examples with complete annotations")

    # Group by (llm, vision) combination
    by_model = defaultdict(list)
    for ex in valid:
        key = (ex['llm'], ex['vision'])
        by_model[key].append(ex)

    # Shuffle each group
    random.seed(42)
    for key in by_model:
        random.shuffle(by_model[key])

    # Round-robin selection to ensure diversity
    selected = []
    model_keys = list(by_model.keys())

    while len(selected) < n and any(by_model.values()):
        for key in model_keys:
            if len(selected) >= n:
                break
            if by_model[key]:
                ex = by_model[key].pop(0)
                selected.append(ex)

    return selected


def generate_latex_table(examples: list, image_dir: str) -> str:
    """
    Generate LaTeX code for the phrase examples.
    Split into multiple figures (4 examples = 2 rows per figure) to fit on pages.
    """
    lines = []
    lines.append(r"""% Auto-generated phrase example figures
% Images should be in figures/phrase_examples/

\newcommand{\phraseexample}[5]{%
  % #1: full image, #2: crop image, #3: LN-Lens phrase, #4: Random phrase, #5: model info
  \begin{minipage}[t]{0.47\textwidth}
    \raggedright
    \includegraphics[height=2.8cm]{#1}\hspace{0.3em}%
    \includegraphics[height=2.8cm]{#2}\\[0.3em]
    {\footnotesize\textbf{LN-Lens:} #3}\\[0.1em]
    {\footnotesize\textbf{Random:} #4}\\[0.1em]
    {\scriptsize\textit{#5}}
  \end{minipage}%
}
""")

    # Split into figures of 4 examples each (2 rows of 2)
    examples_per_figure = 4
    num_figures = (len(examples) + examples_per_figure - 1) // examples_per_figure

    for fig_idx in range(num_figures):
        start_idx = fig_idx * examples_per_figure
        end_idx = min(start_idx + examples_per_figure, len(examples))
        fig_examples = examples[start_idx:end_idx]

        lines.append(r"\begin{figure}[H]")
        lines.append(r"\centering")

        # Process pairs within this figure
        for pair_idx in range(0, len(fig_examples), 2):
            ex1 = fig_examples[pair_idx]
            ex2 = fig_examples[pair_idx + 1] if pair_idx + 1 < len(fig_examples) else None

            global_idx1 = start_idx + pair_idx

            # Format example 1
            img1_full = f"figures/phrase_examples/example_{global_idx1:02d}.pdf"
            img1_crop = f"figures/phrase_examples/example_{global_idx1:02d}_crop.pdf"
            phrase1 = format_phrase_latex(ex1['phrase'], ex1['token'])
            random1 = format_phrase_latex(ex1['_comparison_phrase'], ex1['token'])

            vision1 = ex1['vision']
            if 'siglip' in vision1.lower():
                vision1_display = 'SigLIP'
            elif 'dinov2' in vision1.lower():
                vision1_display = 'DINOv2'
            else:
                vision1_display = 'CLIP'

            llm1_display = LLM_DISPLAY.get(ex1['llm'], ex1['llm'])
            model1 = f"{llm1_display}+{vision1_display}, L{ex1['layer']}"

            lines.append(r"\phraseexample{" + img1_full + "}{" + img1_crop + "}{" + phrase1 + "}{" + random1 + "}{" + model1 + "}")

            if ex2:
                global_idx2 = start_idx + pair_idx + 1
                lines.append(r"\hfill")

                img2_full = f"figures/phrase_examples/example_{global_idx2:02d}.pdf"
                img2_crop = f"figures/phrase_examples/example_{global_idx2:02d}_crop.pdf"
                phrase2 = format_phrase_latex(ex2['phrase'], ex2['token'])
                random2 = format_phrase_latex(ex2['_comparison_phrase'], ex2['token'])

                vision2 = ex2['vision']
                if 'siglip' in vision2.lower():
                    vision2_display = 'SigLIP'
                elif 'dinov2' in vision2.lower():
                    vision2_display = 'DINOv2'
                else:
                    vision2_display = 'CLIP'

                llm2_display = LLM_DISPLAY.get(ex2['llm'], ex2['llm'])
                model2 = f"{llm2_display}+{vision2_display}, L{ex2['layer']}"

                lines.append(r"\phraseexample{" + img2_full + "}{" + img2_crop + "}{" + phrase2 + "}{" + random2 + "}{" + model2 + "}")

            lines.append(r"\\[1em]")

        # Caption for this figure
        example_range = f"{start_idx + 1}--{end_idx}"
        if fig_idx == 0:
            # First figure gets the full caption
            lines.append(r"\caption{")
            lines.append(r"    \textbf{Phrase annotation examples (" + example_range + r").}")
            lines.append(r"    Each panel shows a vision token's patch (red box) preprocessed as the vision encoder sees it.")
            lines.append(r"    \textbf{LN-Lens}: Top contextual nearest neighbor phrase from \vlens.")
            lines.append(r"    \textbf{Random}: A random VG phrase containing the same token.")
            lines.append(r"}")
            lines.append(r"\label{fig:phrase_examples}")
        else:
            lines.append(r"\caption{Phrase annotation examples (" + example_range + r").}")
            lines.append(r"\label{fig:phrase_examples_" + str(fig_idx + 1) + "}")

        lines.append(r"\end{figure}")
        lines.append("")

    return '\n'.join(lines)


def main():
    random.seed(42)

    base_dir = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo")

    # Load metadata
    metadata_path = base_dir / "analysis_results" / "phrase_annotation_examples" / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load dataset for images
    dataset = PixMoCap(split="validation", mode="captions")

    # Select 12 diverse examples
    selected = select_diverse_examples(metadata['examples'], n=12)
    print(f"Selected {len(selected)} examples")

    # Create output directory
    output_dir = base_dir / "paper" / "figures" / "phrase_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    for i, ex in enumerate(selected):
        img_idx = ex['img_idx']
        example_data = dataset.get(img_idx, random)
        image_path = example_data['image']

        if not Path(image_path).exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert('RGB')

        # Preprocess and draw bbox - returns full image and 5x5 crop
        full_img, crop_img = preprocess_and_draw_bbox(
            image,
            ex['vision'],
            ex['patch_row'],
            ex['patch_col']
        )

        # Save both as PDF
        full_path = output_dir / f"example_{i:02d}.pdf"
        crop_path = output_dir / f"example_{i:02d}_crop.pdf"
        full_img.save(full_path, 'PDF', resolution=150)
        crop_img.save(crop_path, 'PDF', resolution=150)
        print(f"Saved: {full_path} and crop")

    # Generate LaTeX code
    latex_code = generate_latex_table(selected, str(output_dir))

    latex_path = output_dir / "phrase_examples_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"\nLaTeX code saved to: {latex_path}")

    # Also save selected examples metadata
    selected_meta = {
        'num_examples': len(selected),
        'examples': selected
    }
    with open(output_dir / "selected_examples.json", 'w') as f:
        json.dump(selected_meta, f, indent=2)

    print(f"\nDone! Generated {len(selected)} images and LaTeX table.")
    print(f"\nTo use in appendix, add:")
    print(f"  \\input{{figures/phrase_examples/phrase_examples_table.tex}}")


if __name__ == "__main__":
    main()
