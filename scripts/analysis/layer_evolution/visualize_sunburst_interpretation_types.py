#!/usr/bin/env python3
"""
Sunburst chart for interpretation types showing:
- Inner ring: 3 interpretation types (Concrete, Abstract, Global) - proportional to counts
- Middle ring: Most frequent words per type - proportional to counts
- Outer ring: Example phrases with preceding context ("...the black")

Uses matplotlib nested pie charts with radial text orientation.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_radial_rotation(angle_deg):
    """Get rotation for radial outward text, flipped 180° on left side."""
    angle_deg = angle_deg % 360
    # On left side (90° to 270°), flip 180° so text reads outward
    if 90 < angle_deg < 270:
        return angle_deg + 180
    else:
        return angle_deg


def create_sunburst(data, output_path, num_words=8, num_phrases_per_word=3):
    """
    Create a 3-ring sunburst chart.

    Args:
        data: dict with keys 'Concrete', 'Abstract', 'Global'
              Each value has 'count' and 'words' dict
        output_path: Where to save the figure
        num_words: Number of top words per category
        num_phrases_per_word: Number of example phrases per word
    """

    # Colors for the 3 categories
    colors = {
        'Concrete': '#4CAF50',  # Green
        'Abstract': '#2196F3',  # Blue
        'Global': '#FF9800'     # Orange
    }

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal')

    # Get total counts for proportions
    total_all = sum(data[cat]['count'] for cat in ['Concrete', 'Abstract', 'Global'])

    # Build data for all 3 rings
    # Ring 1: Categories
    cat_sizes = []
    cat_colors = []
    cat_labels = []

    # Ring 2: Words (proportional to word counts)
    word_sizes = []
    word_colors = []
    word_labels = []
    word_angles = []  # For text placement

    # Ring 3: Phrases
    phrase_sizes = []
    phrase_colors = []
    phrase_labels = []
    phrase_angles = []

    current_angle = 0  # Track cumulative angle

    for cat in ['Concrete', 'Abstract', 'Global']:
        cat_count = data[cat]['count']
        cat_prop = cat_count / total_all
        cat_angle_span = cat_prop * 360

        cat_sizes.append(cat_prop)
        cat_colors.append(colors[cat])
        cat_labels.append(f"{cat}\n({cat_count:,})")

        # Get top words for this category, sorted by count
        words_dict = data[cat]['words']
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1]['count'], reverse=True)[:num_words]

        # Calculate total count of selected words for proportional sizing
        total_word_count = sum(w[1]['count'] for w in sorted_words)

        # Track angle within this category
        cat_start_angle = current_angle

        for word, word_info in sorted_words:
            word_count = word_info['count']
            # Word proportion relative to category's word total
            word_prop_in_cat = word_count / total_word_count
            # Actual size is proportion of full circle
            word_size = cat_prop * word_prop_in_cat
            word_angle_span = word_size * 360

            word_sizes.append(word_size)
            word_colors.append(colors[cat])
            word_labels.append(f"{word}\n({word_count})")

            # Calculate midpoint angle for this word wedge
            word_mid_angle = current_angle + (word_angle_span / 2)
            word_angles.append(word_mid_angle)

            # Get example phrases (captions) - prefer preceding context format
            raw_captions = word_info.get('captions', [])

            # Filter and format captions to show preceding context
            formatted_captions = []
            for cap in raw_captions:
                # Check if caption ends with the word (preceding context)
                cap_lower = cap.lower().rstrip('.')
                word_lower = word.lower()
                if cap_lower.endswith(word_lower):
                    # Already in "...context word" format
                    formatted_captions.append(cap)
                elif cap.startswith('...') and word_lower in cap_lower:
                    # Has ... prefix, good format
                    formatted_captions.append(cap)

            # If no preceding-context captions, use any captions we have
            if not formatted_captions:
                formatted_captions = [f"...{word}" for _ in range(num_phrases_per_word)]
            elif len(formatted_captions) < num_phrases_per_word:
                # Pad with the word itself
                formatted_captions.extend([f"...{word}"] * (num_phrases_per_word - len(formatted_captions)))

            captions = formatted_captions[:num_phrases_per_word]

            num_phrases = len(captions)
            phrase_angle_each = word_angle_span / num_phrases

            for i, caption in enumerate(captions):
                phrase_sizes.append(word_size / num_phrases)
                phrase_colors.append(colors[cat])
                phrase_labels.append(caption)

                # Phrase midpoint angle
                phrase_mid = current_angle + (i + 0.5) * phrase_angle_each
                phrase_angles.append(phrase_mid)

            current_angle += word_angle_span

    # Draw rings from inside out
    # Ring 1: Categories (innermost)
    ring1_width = 0.25
    ring1_radius = 0.3
    wedges1, _ = ax.pie(
        cat_sizes,
        radius=ring1_radius,
        colors=cat_colors,
        wedgeprops=dict(width=ring1_width, edgecolor='white', linewidth=2),
        startangle=90,
        counterclock=False
    )

    # Ring 2: Words (middle)
    ring2_width = 0.25
    ring2_radius = ring1_radius + ring2_width
    wedges2, _ = ax.pie(
        word_sizes,
        radius=ring2_radius,
        colors=[c for c in word_colors],  # Slightly lighter
        wedgeprops=dict(width=ring2_width, edgecolor='white', linewidth=1),
        startangle=90,
        counterclock=False
    )

    # Ring 3: Phrases (outermost) - lighter colors
    ring3_width = 0.45
    ring3_radius = ring2_radius + ring3_width

    # Create lighter versions of colors for outer ring
    def lighten_color(hex_color, factor=0.6):
        """Make a color lighter by mixing with white."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    phrase_colors_light = [lighten_color(c, 0.5) for c in phrase_colors]

    wedges3, _ = ax.pie(
        phrase_sizes,
        radius=ring3_radius,
        colors=phrase_colors_light,
        wedgeprops=dict(width=ring3_width, edgecolor='white', linewidth=0.5),
        startangle=90,
        counterclock=False
    )

    # Add text labels for Ring 1 (categories) - centered
    angle_so_far = 90  # Start at top (90 degrees)
    for i, (cat, size) in enumerate(zip(['Concrete', 'Abstract', 'Global'], cat_sizes)):
        angle_span = size * 360
        mid_angle = angle_so_far - angle_span / 2
        mid_rad = np.radians(mid_angle)

        r = ring1_radius - ring1_width / 2
        x = r * np.cos(mid_rad)
        y = r * np.sin(mid_rad)

        ax.text(x, y, cat, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

        angle_so_far -= angle_span

    # Add text labels for Ring 2 (words) - radial outward
    angle_so_far = 90
    for i, (label, size) in enumerate(zip(word_labels, word_sizes)):
        angle_span = size * 360
        mid_angle = angle_so_far - angle_span / 2
        mid_rad = np.radians(mid_angle)

        r = ring1_radius + ring2_width / 2
        x = r * np.cos(mid_rad)
        y = r * np.sin(mid_rad)

        # Radial rotation with flip on left side
        rotation = get_radial_rotation(mid_angle - 90)  # -90 to make text radial

        # Extract just the word (not the count)
        word_only = label.split('\n')[0]

        ax.text(x, y, word_only, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                rotation=rotation, rotation_mode='anchor')

        angle_so_far -= angle_span

    # Add text labels for Ring 3 (phrases) - radial outward
    angle_so_far = 90
    for i, (label, size) in enumerate(zip(phrase_labels, phrase_sizes)):
        angle_span = size * 360
        mid_angle = angle_so_far - angle_span / 2
        mid_rad = np.radians(mid_angle)

        r = ring2_radius + ring3_width / 2
        x = r * np.cos(mid_rad)
        y = r * np.sin(mid_rad)

        # Radial rotation with flip on left side
        rotation = get_radial_rotation(mid_angle - 90)

        # Truncate label if too long
        max_chars = 25
        if len(label) > max_chars:
            label = label[:max_chars-3] + "..."

        # Adaptive font size based on wedge size
        fontsize = max(5, min(8, int(angle_span * 0.8)))

        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, color='#333333',
                rotation=rotation, rotation_mode='anchor')

        angle_so_far -= angle_span

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Concrete'], label=f"Concrete ({data['Concrete']['count']:,})"),
        Patch(facecolor=colors['Abstract'], label=f"Abstract ({data['Abstract']['count']:,})"),
        Patch(facecolor=colors['Global'], label=f"Global ({data['Global']['count']:,})")
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    plt.title('Interpretation Types: Words and Context Phrases', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(str(output_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved sunburst to {output_path}")


def main():
    base_dir = Path(__file__).parent.parent.parent.parent
    data_path = base_dir / 'analysis_results/layer_evolution/sunburst_data.pkl'
    output_path = base_dir / 'analysis_results/layer_evolution/sunburst_interpretation_types.pdf'

    # Also save to paper figures
    paper_output = base_dir / 'paper/figures/fig_sunburst_interpretation_types.pdf'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    create_sunburst(data, output_path, num_words=8, num_phrases_per_word=3)

    # Copy to paper figures
    import shutil
    shutil.copy(output_path, paper_output)
    shutil.copy(str(output_path).replace('.pdf', '.png'),
                str(paper_output).replace('.pdf', '.png'))
    print(f"Copied to {paper_output}")


if __name__ == '__main__':
    main()
