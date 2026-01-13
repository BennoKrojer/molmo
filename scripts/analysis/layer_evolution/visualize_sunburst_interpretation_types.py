#!/usr/bin/env python3
"""
Sunburst chart for interpretation types showing:
- Inner ring: 3 interpretation types (Concrete, Abstract, Global) - proportional to counts
- Middle ring: Most frequent words per type - proportional to counts
- Outer ring: Example phrases with preceding context ("...the black *word*")

Uses Plotly sunburst with insidetextorientation='radial' for automatic text orientation.
"""

import pickle
import re
from pathlib import Path

import plotly.graph_objects as go


def format_preceding_context(caption, word, max_words=4):
    """
    Extract preceding context and format as '...context *word*'.
    Returns None if word not found or no preceding context.
    """
    cap_lower = caption.lower()
    word_lower = word.lower()

    # Find word position
    match = re.search(r'\b' + re.escape(word_lower) + r'\b', cap_lower)
    if not match:
        return None

    start_idx = match.start()
    preceding = caption[:start_idx].strip()

    if not preceding:
        # Word at start - no preceding context
        return None

    # Take last N words
    words_before = preceding.split()[-max_words:]
    if not words_before:
        return None

    context = " ".join(words_before)
    # Add ellipsis if we truncated
    if len(preceding.split()) > max_words:
        return f"...{context} *{word}*"
    else:
        return f"{context} *{word}*"


def create_sunburst(data, output_path, num_words=8, num_phrases_per_word=3):
    """
    Create a 3-ring sunburst chart using Plotly.
    Plotly's insidetextorientation='radial' handles text orientation automatically.
    """
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

    # Root (empty center)
    ids.append("root")
    labels.append("")
    parents.append("")
    values.append(0)
    marker_colors.append("white")

    # Level 1: Categories
    for cat in ['Concrete', 'Abstract', 'Global']:
        cat_count = data[cat]['count']
        ids.append(cat)
        labels.append(f"{cat}<br>({cat_count:,})")
        parents.append("root")
        values.append(cat_count)  # Proportional to count
        marker_colors.append(colors[cat])

    # Level 2: Words (proportional to word counts)
    for cat in ['Concrete', 'Abstract', 'Global']:
        words_dict = data[cat]['words']
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1]['count'], reverse=True)[:num_words]

        for word, word_info in sorted_words:
            word_count = word_info['count']
            word_id = f"{cat}-{word}"

            ids.append(word_id)
            labels.append(word)
            parents.append(cat)
            values.append(word_count)  # Proportional to count
            marker_colors.append(colors[cat])

            # Level 3: Phrases with preceding context
            raw_captions = word_info.get('captions', [])

            # Format captions as preceding context
            formatted = []
            for cap in raw_captions:
                ctx = format_preceding_context(cap, word)
                if ctx:
                    formatted.append(ctx)

            # If no good preceding context, try using raw captions that have ...
            if not formatted:
                for cap in raw_captions:
                    if cap.startswith('...') or '...' in cap:
                        formatted.append(cap)

            # Fallback
            if not formatted:
                formatted = [f"*{word}*"] * num_phrases_per_word

            # Take up to num_phrases_per_word
            phrases = formatted[:num_phrases_per_word]

            # Add phrase entries (split word count among phrases)
            phrase_value = word_count // len(phrases) if phrases else word_count
            for i, phrase in enumerate(phrases):
                phrase_id = f"{word_id}-phrase{i}"
                ids.append(phrase_id)
                labels.append(phrase)
                parents.append(word_id)
                values.append(phrase_value)
                # Lighter color for phrases
                marker_colors.append(colors[cat])

    # Create sunburst
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=marker_colors),
        insidetextorientation='radial',  # Automatic radial text orientation!
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title=dict(
            text="Interpretation Types: Words and Context Phrases",
            font=dict(size=18)
        ),
        width=1200,
        height=1200,
        margin=dict(t=50, l=10, r=10, b=10)
    )

    # Save as PDF and PNG
    fig.write_image(str(output_path), format='pdf')
    fig.write_image(str(output_path).replace('.pdf', '.png'), format='png', scale=2)

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
