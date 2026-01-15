#!/usr/bin/env python3
"""
Generate icicle plot for interpretation types breakdown.
Shows categories (Concrete/Abstract/Global) with top words and example phrases.
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

DATA_PATH = Path("analysis_results/layer_evolution/sunburst_data.pkl")
OUTPUT_DIR = Path("analysis_results/layer_evolution")
PAPER_DIR = Path("paper/figures")

# Colors for each category
COLORS = {
    'Concrete': '#4CAF50',
    'Abstract': '#2196F3',
    'Global': '#FF9800'
}

# Layout parameters
NUM_WORDS = 6
NUM_PHRASES = 5
FIGSIZE = (16, 5)  # Wide format for full paper width


def measure_text(ax, fig, text, fontsize, fontweight='normal'):
    """Measure text width in pixels."""
    t = ax.text(0, -1000, text, fontsize=fontsize, fontweight=fontweight)
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    width_px = bbox.width
    t.remove()
    return width_px


def get_box_width_pixels(ax, x_left, x_right):
    """Convert axis coordinates to pixel width."""
    p1 = ax.transData.transform([x_left, 0])
    p2 = ax.transData.transform([x_right, 0])
    return p2[0] - p1[0]


def fit_text(ax, fig, text, box_width_px, max_fs, min_fs=3.5):
    """Fit text to box width, reducing font size or truncating as needed."""
    fontsize = max_fs
    while fontsize >= min_fs:
        width = measure_text(ax, fig, text, fontsize)
        if width <= box_width_px * 0.92:
            return text, fontsize
        fontsize -= 0.5
    # Need to truncate
    while len(text) > 4:
        text = text[:-2] + "…"
        width = measure_text(ax, fig, text, min_fs)
        if width <= box_width_px * 0.92:
            return text, min_fs
    return text[:3] + "…", min_fs


def fit_phrase_with_bold(ax, fig, phrase, word, box_width_px, max_fs, min_fs=3.5):
    """Fit phrase with bold word formatting."""
    # Work with plain text first
    plain = phrase.replace(f'*{word}*', word)

    fontsize = max_fs
    while fontsize >= min_fs:
        # Bold adds ~10% width
        width = measure_text(ax, fig, plain, fontsize) * 1.05
        if width <= box_width_px * 0.92:
            if word in plain:
                return plain.replace(word, r'$\mathbf{' + word + r'}$'), fontsize
            return plain, fontsize
        fontsize -= 0.5

    # Need to truncate
    while len(plain) > 4:
        plain = plain[:-2] + "…"
        width = measure_text(ax, fig, plain, min_fs) * 1.05
        if width <= box_width_px * 0.92:
            if word in plain and not plain.endswith(word + "…"):
                return plain.replace(word, r'$\mathbf{' + word + r'}$'), min_fs
            return plain, min_fs

    return plain[:3] + "…", min_fs


def generate_icicle_plot(data, output_path, figsize=FIGSIZE):
    """Generate icicle plot visualization."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.canvas.draw()

    categories_order = ['Concrete', 'Abstract', 'Global']
    row_height = 100 / 3

    for row_idx, cat in enumerate(categories_order):
        cat_count = data[cat]['count']
        cat_pct = cat_count / sum(data[c]['count'] for c in categories_order) * 100
        row_y = 100 - (row_idx + 1) * row_height

        # Category box
        cat_width = 8  # Narrower for wider figure
        cat_box_x = 0.3
        rect = FancyBboxPatch((cat_box_x, row_y + 0.5), cat_width - 0.3, row_height - 1,
                              boxstyle="round,pad=0.01,rounding_size=0.4",
                              facecolor=COLORS[cat], edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)

        cat_box_px = get_box_width_pixels(ax, cat_box_x, cat_box_x + cat_width - 0.3)
        cat_text, cat_fs = fit_text(ax, fig, cat, cat_box_px, 11, 8)
        ax.text(cat_width/2, row_y + row_height/2 + 3, cat_text,
                ha='center', va='center', fontsize=cat_fs, fontweight='bold', color='white')
        ax.text(cat_width/2, row_y + row_height/2 - 4, f"{cat_pct:.0f}%",
                ha='center', va='center', fontsize=8, color='white')

        # Words
        words_dict = data[cat]['words']
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1]['count'], reverse=True)[:NUM_WORDS]
        word_area_width = 100 - cat_width - 0.3
        word_col_width = word_area_width / NUM_WORDS

        for j, (word, info) in enumerate(sorted_words):
            wx = cat_width + 0.2 + j * word_col_width
            word_box_left = wx + 0.15
            word_box_right = wx + word_col_width - 0.15
            word_box_px = get_box_width_pixels(ax, word_box_left, word_box_right)

            # Lighter color for word box
            base = COLORS[cat].replace('#', '')
            r, g, b = int(base[:2], 16), int(base[2:4], 16), int(base[4:], 16)
            r2, g2, b2 = min(255, r+50), min(255, g+50), min(255, b+50)
            word_color = f'#{r2:02x}{g2:02x}{b2:02x}'

            word_box_height = 10  # Taller for shorter figure
            word_y = row_y + row_height - word_box_height - 0.3

            rect = FancyBboxPatch((word_box_left, word_y), word_box_right - word_box_left, word_box_height,
                                  boxstyle="round,pad=0.01,rounding_size=0.2",
                                  facecolor=word_color, edgecolor='white', linewidth=0.8)
            ax.add_patch(rect)

            word_text, word_fs = fit_text(ax, fig, word, word_box_px, 9, 6)
            ax.text((word_box_left + word_box_right) / 2, word_y + word_box_height/2 + 1.5,
                    word_text, ha='center', va='center', fontsize=word_fs, fontweight='bold', color='#333')
            ax.text((word_box_left + word_box_right) / 2, word_y + word_box_height/2 - 2.5,
                    f"({info['count']})", ha='center', va='center', fontsize=6, color='#555')

            # Phrases
            phrases = info.get('phrases', {})
            phrase_area_height = word_y - row_y - 0.3
            phrase_height = phrase_area_height / NUM_PHRASES
            sorted_phrases = list(phrases.keys())[:NUM_PHRASES] if phrases else []

            for k in range(NUM_PHRASES):
                py = word_y - 0.2 - (k + 1) * phrase_height + phrase_height
                phrase_box_bottom = py - phrase_height + 0.15

                # Even lighter color for phrases
                r3, g3, b3 = min(255, r+80), min(255, g+80), min(255, b+80)
                phrase_color = f'#{r3:02x}{g3:02x}{b3:02x}'

                rect = FancyBboxPatch((word_box_left, phrase_box_bottom),
                                      word_box_right - word_box_left, phrase_height - 0.15,
                                      boxstyle="round,pad=0.005,rounding_size=0.08",
                                      facecolor=phrase_color, edgecolor='white', linewidth=0.3)
                ax.add_patch(rect)

                if k < len(sorted_phrases):
                    phrase = sorted_phrases[k]
                    phrase_text, phrase_fs = fit_phrase_with_bold(ax, fig, phrase, word, word_box_px, 6.5, 4)
                    ax.text((word_box_left + word_box_right) / 2, (py + phrase_box_bottom) / 2,
                            phrase_text, ha='center', va='center', fontsize=phrase_fs, color='#444')

    # Save outputs
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path.with_suffix('.pdf')


def main():
    # Load data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Generate plot
    output_path = OUTPUT_DIR / "icicle_final"
    pdf_path = generate_icicle_plot(data, output_path)

    # Copy to paper directory
    import shutil
    paper_path = PAPER_DIR / "fig_icicle_interpretation_types.pdf"
    shutil.copy(pdf_path, paper_path)

    print(f"Generated: {pdf_path}")
    print(f"Copied to: {paper_path}")


if __name__ == "__main__":
    main()
