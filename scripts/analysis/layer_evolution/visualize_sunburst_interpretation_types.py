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


def create_sunburst(data, output_path, num_words=5, num_phrases_per_word=2):
    """
    Create a 3-ring sunburst chart using Plotly.
    Plotly's insidetextorientation='radial' handles text orientation automatically.

    - 5 words per category
    - "Others" capped at 20% visually, labeled with actual %
    - 2 phrases per word
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

        for word, word_info in cd['sorted_words']:
            word_count = word_info['count']
            word_id = f"{cat}-{word}"

            ids.append(word_id)
            labels.append(word)
            parents.append(cat)
            values.append(word_count)  # Proportional to count
            marker_colors.append(colors[cat])

            # Level 3: Phrases with REAL counts from data
            phrases_dict = word_info.get('phrases', {})

            if phrases_dict:
                # Sort by count and take top 3
                sorted_phrases = sorted(phrases_dict.items(), key=lambda x: x[1], reverse=True)[:3]

                # Calculate total for these top phrases to ensure they sum to word_count
                phrase_total = sum(count for _, count in sorted_phrases)

                for i, (phrase_text, phrase_count) in enumerate(sorted_phrases):
                    phrase_id = f"{word_id}-phrase{i}"

                    # Convert *word* format to <b>word</b> for HTML bold
                    label = phrase_text.replace(f'*{word}*', f'<b>{word}</b>')

                    ids.append(phrase_id)
                    labels.append(label)
                    parents.append(word_id)
                    # Scale phrase counts to sum to word_count (for branchvalues="total")
                    scaled_value = int(phrase_count / phrase_total * word_count) if phrase_total > 0 else word_count // 3
                    values.append(scaled_value)
                    marker_colors.append(colors[cat])

                # Add remainder to first phrase to ensure exact sum
                phrase_sum = sum(values[-len(sorted_phrases):])
                if phrase_sum < word_count:
                    values[-len(sorted_phrases)] += (word_count - phrase_sum)
            else:
                # Fallback - single phrase with just the word
                phrase_id = f"{word_id}-phrase0"
                ids.append(phrase_id)
                labels.append(f"<b>{word}</b>")
                parents.append(word_id)
                values.append(word_count)
                marker_colors.append(colors[cat])

    # Create sunburst
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
        textfont=dict(size=12),
        maxdepth=3,
        textinfo='label',
    ))

    fig.update_layout(
        title=dict(
            text="Interpretation Types: Words and Context Phrases",
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
