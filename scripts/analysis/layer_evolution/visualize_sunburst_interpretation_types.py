#!/usr/bin/env python3
"""
Sunburst chart for interpretation types showing:
- Inner ring: 3 interpretation types (Concrete, Abstract, Global) - proportional to counts
- Middle ring: Most frequent words per type - proportional to counts
- Outer ring: Example phrases with preceding context ("...the black *word*")

Uses Plotly sunburst with insidetextorientation='radial' for automatic text orientation.

Usage:
    python visualize_sunburst_interpretation_types.py
    python visualize_sunburst_interpretation_types.py --data sunburst_data_layer0.pkl --suffix "_layer0" --title "Layer 0"
"""

import argparse
import pickle
import re
from pathlib import Path

import plotly.graph_objects as go


def format_context(caption, word, max_words=3):
    """
    Format caption to show PRECEDING context only (autoregressive LLM sees only past).
    Format: "...word1 word2 word3 <b>target</b>" (always max 3 preceding words)
    """
    cap_lower = caption.lower()
    word_lower = word.lower()

    # Find word position
    match = re.search(r'\b' + re.escape(word_lower) + r'\b', cap_lower)
    if not match:
        return None

    start_idx = match.start()
    preceding = caption[:start_idx].strip()

    # Always show up to 3 preceding words - never truncate further
    if preceding:
        words_before = preceding.split()[-max_words:]  # Take last 3 words
        context = " ".join(words_before)
        if len(preceding.split()) > max_words:
            return f"...{context} <b>{word}</b>"
        else:
            return f"{context} <b>{word}</b>"

    # No preceding context - just the bold word
    return f"<b>{word}</b>"


def create_sunburst(data, output_path, num_words=5, num_phrases_per_word=2, title=None):
    """
    Create a 3-ring sunburst chart using Plotly.
    Plotly's insidetextorientation='radial' handles text orientation automatically.

    - 5 words per category
    - "Others" capped at 20% visually, labeled with actual %
    - 2 phrases per word
    """
    if title is None:
        title = "Interpretation Types: Words and Context Phrases"
    # Colors for the 3 categories
    colors = {
        'Concrete': '#4CAF50',  # Green
        'Abstract': '#2196F3',  # Blue
        'Global': '#FF9800'     # Orange
    }

    # Build sunburst data: ids, labels, parents, values
    ids = []
    labels = []
    parents = []
    values = []
    marker_colors = []
    font_sizes = []  # Progressive font sizes by depth level

    # Font sizes: inner ring big, middle medium, outer small
    FONT_SIZE_CATEGORY = 22  # Inner ring (Concrete, Abstract, Global)
    FONT_SIZE_WORD = 13      # Middle ring (words)
    FONT_SIZE_PHRASE = 10    # Outer ring (phrases)

    # First pass: calculate visual values for each category
    cat_visual_values = {}
    cat_data = {}
    for cat in ['Concrete', 'Abstract', 'Global']:
        words_dict = data[cat]['words']
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1]['count'], reverse=True)[:num_words]
        shown_count = sum(w[1]['count'] for w in sorted_words)
        other_count = data[cat]['count'] - shown_count
        other_pct = other_count / data[cat]['count'] * 100
        # Others visual capped at 20% => 25% of shown
        others_visual = min(other_count, shown_count * 0.25)
        cat_visual = shown_count + others_visual
        cat_visual_values[cat] = cat_visual
        cat_data[cat] = {
            'sorted_words': sorted_words,
            'other_count': other_count,
            'other_pct': other_pct,
            'others_visual': others_visual,
            'shown_count': shown_count
        }

    # Level 1: Categories with adjusted visual values
    for cat in ['Concrete', 'Abstract', 'Global']:
        cat_count = data[cat]['count']
        ids.append(cat)
        labels.append(f"{cat}<br>({cat_count:,})")
        parents.append("")
        values.append(cat_visual_values[cat])  # Use visual value, not total
        marker_colors.append(colors[cat])
        font_sizes.append(FONT_SIZE_CATEGORY)

    # Level 2: Words + Others
    for cat in ['Concrete', 'Abstract', 'Global']:
        cd = cat_data[cat]

        # Add "Others" segment
        if cd['other_count'] > 0:
            other_id = f"{cat}-other"
            ids.append(other_id)
            labels.append(f"Others ({cd['other_pct']:.0f}%)")
            parents.append(cat)
            values.append(cd['others_visual'])
            # Lighter color
            base_color = colors[cat]
            lighter = base_color.replace('#', '')
            r, g, b = int(lighter[:2], 16), int(lighter[2:4], 16), int(lighter[4:], 16)
            r, g, b = min(255, r+60), min(255, g+60), min(255, b+60)
            marker_colors.append(f'#{r:02x}{g:02x}{b:02x}')
            font_sizes.append(FONT_SIZE_WORD)  # Same as words level

        for word, word_info in cd['sorted_words']:
            word_count = word_info['count']
            word_id = f"{cat}-{word}"

            ids.append(word_id)
            labels.append(word)
            parents.append(cat)
            values.append(word_count)  # Proportional to count
            marker_colors.append(colors[cat])
            font_sizes.append(FONT_SIZE_WORD)

            # Level 3: Phrases with REAL counts from data + Others
            phrases_dict = word_info.get('phrases', {})

            if phrases_dict:
                # Sort ALL phrases by count, take top N for display
                all_sorted = sorted(phrases_dict.items(), key=lambda x: x[1], reverse=True)

                # Calculate totals for Others percentage (using all phrases)
                total_all = sum(count for _, count in all_sorted)

                # Determine how many phrases we can show given word_count
                # Each phrase needs at least 1 unit, and we might need 1 for Others
                max_phrases = min(3, word_count)  # Can't show more phrases than word_count
                top_n = all_sorted[:max_phrases]

                top_n_total = sum(count for _, count in top_n)
                others_count = total_all - top_n_total
                others_pct = (others_count / total_all * 100) if total_all > 0 else 0

                # Decide if we show Others (only if we have budget and it's meaningful)
                show_others = others_count > 0 and others_pct > 0.5 and word_count > len(top_n)

                # Distribute word_count among phrases (and Others if shown)
                num_segments = len(top_n) + (1 if show_others else 0)

                # For small word_counts, distribute evenly; for larger, use proportions
                if word_count <= num_segments:
                    # Just give 1 to each (may not show Others)
                    phrase_values = [1] * min(len(top_n), word_count)
                    if word_count > len(top_n) and show_others:
                        phrase_values.append(1)
                else:
                    # Calculate proportional values
                    if show_others:
                        others_visual = max(1, min(int(others_count / total_all * word_count), int(word_count * 0.10)))
                        shown_visual = word_count - others_visual
                    else:
                        shown_visual = word_count
                        others_visual = 0

                    phrase_values = []
                    for phrase_text, phrase_count in top_n:
                        scaled = int(phrase_count / top_n_total * shown_visual) if top_n_total > 0 else shown_visual // len(top_n)
                        phrase_values.append(max(1, scaled))

                    if show_others:
                        phrase_values.append(others_visual)

                    # Adjust to ensure exact sum
                    diff = word_count - sum(phrase_values)
                    if diff != 0:
                        # Add/subtract from largest value
                        max_idx = phrase_values.index(max(phrase_values))
                        phrase_values[max_idx] += diff

                # Add top phrases
                for i, (phrase_text, phrase_count) in enumerate(top_n[:len(phrase_values) if not show_others else len(phrase_values)-1]):
                    phrase_id = f"{word_id}-phrase{i}"
                    label = phrase_text.replace(f'*{word}*', f'<b>{word}</b>')

                    ids.append(phrase_id)
                    labels.append(label)
                    parents.append(word_id)
                    values.append(phrase_values[i])
                    marker_colors.append(colors[cat])
                    font_sizes.append(FONT_SIZE_PHRASE)

                # Add "Others" segment if showing
                if show_others and len(phrase_values) > len(top_n):
                    phrase_id = f"{word_id}-others"
                    ids.append(phrase_id)
                    labels.append(f"Others ({others_pct:.0f}%)")
                    parents.append(word_id)
                    values.append(phrase_values[-1])
                    # Lighter color for Others
                    base_color = colors[cat]
                    lighter = base_color.replace('#', '')
                    r, g, b = int(lighter[:2], 16), int(lighter[2:4], 16), int(lighter[4:], 16)
                    r, g, b = min(255, r+80), min(255, g+80), min(255, b+80)
                    marker_colors.append(f'#{r:02x}{g:02x}{b:02x}')
                    font_sizes.append(FONT_SIZE_PHRASE)
            else:
                # Fallback - single phrase with just the word
                phrase_id = f"{word_id}-phrase0"
                ids.append(phrase_id)
                labels.append(f"<b>{word}</b>")
                parents.append(word_id)
                values.append(word_count)
                marker_colors.append(colors[cat])
                font_sizes.append(FONT_SIZE_PHRASE)

    # Validate data before creating sunburst (Plotly fails silently with bad data)
    for i, (id_, parent, val) in enumerate(zip(ids, parents, values)):
        if val <= 0:
            raise ValueError(f"Invalid value {val} for id '{id_}' (parent='{parent}'). Values must be positive.")

    # Verify parent-child sums for branchvalues="total"
    for i, (id_, parent, val) in enumerate(zip(ids, parents, values)):
        if parent == '':  # Root nodes
            children_sum = sum(v for p, v in zip(parents, values) if p == id_)
            if abs(val - children_sum) > 0.01:
                raise ValueError(f"Category '{id_}' value {val} != children sum {children_sum}")

    # Create sunburst with progressive font sizes
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",  # Children must sum to parent - no gaps
        marker=dict(
            colors=marker_colors,
            line=dict(width=0.5, color='white')
        ),
        insidetextorientation='radial',
        textfont=dict(size=font_sizes),  # Progressive sizes per segment
        maxdepth=3,
        textinfo='label',
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18),
            x=0.5,
            y=0.98,
        ),
        width=1000,
        height=1000,
        margin=dict(t=40, l=0, r=0, b=0),
        uniformtext=dict(minsize=6, mode='hide'),  # Hide if can't fit at min size
    )

    # Save as PDF and PNG
    fig.write_image(str(output_path), format='pdf')
    fig.write_image(str(output_path).replace('.pdf', '.png'), format='png', scale=2)

    print(f"Saved sunburst to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate sunburst visualization')
    parser.add_argument('--data', default='sunburst_data.pkl',
                        help='Data filename (in analysis_results/layer_evolution/)')
    parser.add_argument('--suffix', default='',
                        help='Suffix for output filename (e.g., "_layer0")')
    parser.add_argument('--title', default=None,
                        help='Chart title (e.g., "Layer 0")')
    parser.add_argument('--no-paper-copy', action='store_true',
                        help='Skip copying to paper figures')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent.parent
    data_path = base_dir / f'analysis_results/layer_evolution/{args.data}'
    output_path = base_dir / f'analysis_results/layer_evolution/sunburst_interpretation_types{args.suffix}.pdf'

    # Also save to paper figures
    paper_output = base_dir / f'paper/figures/fig_sunburst_interpretation_types{args.suffix}.pdf'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    create_sunburst(data, output_path, title=args.title)

    # Copy to paper figures
    if not args.no_paper_copy:
        import shutil
        shutil.copy(output_path, paper_output)
        shutil.copy(str(output_path).replace('.pdf', '.png'),
                    str(paper_output).replace('.pdf', '.png'))
        print(f"Copied to {paper_output}")


if __name__ == '__main__':
    main()
