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


def format_context(caption, word, max_words=4):
    """
    Format caption to show context around word.
    Prefers preceding context ("...context *word*").
    Falls back to following context ("*word* context...") if word is at start.
    """
    cap_lower = caption.lower()
    word_lower = word.lower()

    # Find word position
    match = re.search(r'\b' + re.escape(word_lower) + r'\b', cap_lower)
    if not match:
        return None

    start_idx = match.start()
    end_idx = match.end()

    preceding = caption[:start_idx].strip()
    following = caption[end_idx:].strip().rstrip('.')

    # Prefer preceding context
    if preceding:
        words_before = preceding.split()[-max_words:]
        context = " ".join(words_before)
        if len(preceding.split()) > max_words:
            return f"...{context} *{word}*"
        else:
            return f"{context} *{word}*"

    # Fall back to following context
    if following:
        words_after = following.split()[:max_words]
        context = " ".join(words_after)
        if len(following.split()) > max_words:
            return f"*{word}* {context}..."
        else:
            return f"*{word}* {context}"

    # Just the word if no context
    return f"*{word}*"


def create_sunburst(data, output_path, num_words=3, num_phrases_per_word=2):
    """
    Create a 3-ring sunburst chart using Plotly.
    Plotly's insidetextorientation='radial' handles text orientation automatically.

    Defaults chosen for readability:
    - 3 words per category to ensure text fits in all segments
    - 2 phrases per word to keep outer ring readable
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

    # Level 1: Categories (no root - categories are top level with parent="")
    for cat in ['Concrete', 'Abstract', 'Global']:
        cat_count = data[cat]['count']
        ids.append(cat)
        labels.append(f"{cat}<br>({cat_count:,})")
        parents.append("")  # Top level
        values.append(cat_count)
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
                ctx = format_context(cap, word)
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
                # Truncate long phrases for readability
                if len(phrase) > 25:
                    phrase = phrase[:22] + "..."
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
        # Don't use branchvalues="total" - let Plotly handle sizing
        marker=dict(
            colors=marker_colors,
            line=dict(width=1, color='white')
        ),
        insidetextorientation='radial',
        textfont=dict(size=12),
        maxdepth=3,
        textinfo='label',  # Show labels only
    ))

    fig.update_layout(
        title=dict(
            text="Interpretation Types: Words and Context Phrases",
            font=dict(size=20),
            x=0.5,
        ),
        width=1800,  # Even larger
        height=1800,
        margin=dict(t=80, l=30, r=30, b=30),
        uniformtext=dict(minsize=5),  # Allow smaller text
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

    create_sunburst(data, output_path)  # Use defaults: 3 words, 2 phrases

    # Copy to paper figures
    import shutil
    shutil.copy(output_path, paper_output)
    shutil.copy(str(output_path).replace('.pdf', '.png'),
                str(paper_output).replace('.pdf', '.png'))
    print(f"Copied to {paper_output}")


if __name__ == '__main__':
    main()
