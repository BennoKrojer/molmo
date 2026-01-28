#!/usr/bin/env python3
"""
Generate LaTeX tables for interpretation type breakdown.
Follows academic paper style conventions.
"""

import pickle
from pathlib import Path
import re

# Configuration
DATA_DIR = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/analysis_results/layer_evolution")
OUTPUT_FILE = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/paper/figures/interpretation_type_tables.tex")

# Model configurations
MODELS = [
    ("olmo-7b_vit-l-14-336", "OLMo-7B + ViT-L", "olmo_vitl"),
    ("olmo-7b_siglip", "OLMo-7B + SigLIP", "olmo_siglip"),
    ("olmo-7b_dinov2-large-336", "OLMo-7B + DINOv2", "olmo_dinov2"),
    ("llama3-8b_vit-l-14-336", "LLaMA3-8B + ViT-L", "llama_vitl"),
    ("llama3-8b_siglip", "LLaMA3-8B + SigLIP", "llama_siglip"),
    ("llama3-8b_dinov2-large-336", "LLaMA3-8B + DINOv2", "llama_dinov2"),
    ("qwen2-7b_vit-l-14-336_seed10", "Qwen2-7B + ViT-L", "qwen2_vitl"),
    ("qwen2-7b_siglip", "Qwen2-7B + SigLIP", "qwen2_siglip"),
    ("qwen2-7b_dinov2-large-336", "Qwen2-7B + DINOv2", "qwen2_dinov2"),
    ("qwen2vl", "Qwen2-VL-7B", "qwen2vl"),
]

VARIANTS = [
    ("", "All layers", "all"),
    ("_early", "Early layers (0, 1, 2)", "early"),
    ("_late", "Late layers", "late"),
]


def escape_latex(text):
    """Escape special LaTeX characters."""
    if text is None or text == "-":
        return "---"

    # First remove any existing backslash sequences (corrupt data)
    text = text.replace('\\\\', '').replace('\\', '')

    # Replace problematic Unicode characters
    replacements = {
        '—': '---',
        '–': '--',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        'é': "e",  # Simplified - just use plain e
        'ç': "c",  # Simplified - just use plain c
        'ñ': "n",
        'ü': "u",
        'ö': "o",
        'ä': "a",
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def format_phrase(phrase, word, max_len=35):
    """Format a phrase with bold word, truncating if needed."""
    if phrase is None or phrase == "-" or not phrase.strip():
        return "---"

    # Clean the phrase
    phrase = phrase.strip()

    # Remove existing bold markers (from *word* format)
    phrase = phrase.replace(f'*{word}*', word)

    # Track if we need ellipsis at start/end
    needs_start_ellipsis = False
    needs_end_ellipsis = False

    # Truncate if too long (work with plain text first)
    if len(phrase) > max_len:
        # Find word position
        word_pos = phrase.lower().find(word.lower())
        if word_pos == -1:
            # Word not in phrase, just truncate
            phrase = phrase[:max_len-3]
            needs_end_ellipsis = True
        else:
            # Keep context around the word
            word_end = word_pos + len(word)

            # Calculate how much we can show
            if word_pos > max_len // 2:
                # Start with ellipsis
                start = max(0, word_pos - 10)
                phrase = phrase[start:]
                needs_start_ellipsis = True
                word_pos = word_pos - start  # Adjust position

            if len(phrase) > max_len - 3:
                phrase = phrase[:max_len-3]
                needs_end_ellipsis = True

    # Now escape LaTeX characters
    phrase_escaped = escape_latex(phrase)

    # Add bold for the word (case-insensitive replacement)
    word_escaped = escape_latex(word)

    # Find and bold the word
    pattern = re.compile(re.escape(word_escaped), re.IGNORECASE)
    phrase_with_bold = pattern.sub(r'\\textbf{' + word_escaped + r'}', phrase_escaped, count=1)

    # Add ellipsis (after escaping, so they don't get escaped)
    if needs_start_ellipsis:
        phrase_with_bold = "..." + phrase_with_bold
    if needs_end_ellipsis:
        phrase_with_bold = phrase_with_bold + "..."

    return phrase_with_bold


def format_word_entry(word, count):
    """Format a word with its count."""
    word_escaped = escape_latex(word)
    return f"{word_escaped} ({count})"


def load_sunburst_data(model_key, variant_suffix):
    """Load sunburst data from pickle file."""
    pkl_path = DATA_DIR / f"sunburst_data_{model_key}{variant_suffix}.pkl"

    if not pkl_path.exists():
        print(f"Warning: {pkl_path} not found")
        return None

    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def generate_table(model_key, model_display, label_key, variant_suffix, variant_display, variant_label):
    """Generate a single LaTeX table."""
    data = load_sunburst_data(model_key, variant_suffix)
    if data is None:
        return ""

    # Calculate total count for percentages
    total_count = sum(data.get(cat, {}).get('count', 0) for cat in ['Concrete', 'Abstract', 'Global'])
    if total_count == 0:
        return ""

    # Extract category data
    categories = {}
    for cat_name in ['Concrete', 'Abstract', 'Global']:
        cat_data = data.get(cat_name, {})
        if cat_data and cat_data.get('count', 0) > 0:
            # Calculate percentage
            pct = round(100 * cat_data['count'] / total_count)

            # Get words dict and convert to sorted list
            words_dict = cat_data.get('words', {})
            words_list = []
            for word, word_info in words_dict.items():
                # Get best phrase (highest count)
                phrases_dict = word_info.get('phrases', {})
                if phrases_dict:
                    best_phrase = max(phrases_dict.keys(), key=lambda p: phrases_dict[p])
                else:
                    best_phrase = "---"
                words_list.append({
                    'word': word,
                    'count': word_info.get('count', 0),
                    'phrase': best_phrase
                })

            # Sort by count and take top 10
            words_list.sort(key=lambda x: x['count'], reverse=True)
            words_list = words_list[:10]

            categories[cat_name] = {
                'percentage': pct,
                'words': words_list
            }

    if not categories:
        return ""

    # Build table
    lines = []

    # Table header (matching paper style: no vertical bars, small font)
    lines.append(f"% {model_display} - {variant_display}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    # Use fixed-width columns to prevent overflow
    lines.append(r"\begin{tabular}{@{}l l p{4.5cm}@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{Word (count)} & \textbf{Example Phrase} \\")
    lines.append(r"\midrule")

    # Add rows for each category
    for cat_idx, (cat_name, cat_data) in enumerate(categories.items()):
        pct = cat_data['percentage']
        words = cat_data['words']

        if not words:
            continue

        # First row of category shows category name
        for word_idx, word_data in enumerate(words):
            word = word_data.get('word', '')
            count = word_data.get('count', 0)
            phrase = word_data.get('phrase', '---')

            word_entry = format_word_entry(word, count)
            phrase_entry = format_phrase(phrase, word)

            if word_idx == 0:
                cat_entry = f"\\textbf{{{cat_name}}} ({pct}\\%)"
            else:
                cat_entry = ""

            lines.append(f"{cat_entry} & {word_entry} & {phrase_entry} \\\\")

        # Add midrule between categories (except after last)
        if cat_idx < len(categories) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Caption at end (paper style)
    late_layers = "31, 30, 24" if "qwen" not in model_key.lower() else "27, 26, 24"
    if variant_label == "all":
        caption = f"Interpretation types for \\textbf{{{model_display}}} across all layers. Top 10 words per category with occurrence counts and example Visual Genome phrases (target word in bold)."
    elif variant_label == "early":
        caption = f"Interpretation types for \\textbf{{{model_display}}} at early layers (0, 1, 2)."
    else:
        caption = f"Interpretation types for \\textbf{{{model_display}}} at late layers ({late_layers})."

    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:interp_{label_key}_{variant_label}}}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate all interpretation type tables."""
    all_tables = []

    all_tables.append("% Auto-generated interpretation type tables")
    all_tables.append("% Style: matches paper conventions (no vertical bars, caption at end)")
    all_tables.append("% Generated by generate_interpretation_tables_v2.py")
    all_tables.append("")

    for model_key, model_display, label_key in MODELS:
        for variant_suffix, variant_display, variant_label in VARIANTS:
            table = generate_table(model_key, model_display, label_key,
                                   variant_suffix, variant_display, variant_label)
            if table:
                all_tables.append(table)

    # Write output
    output = "\n".join(all_tables)
    OUTPUT_FILE.write_text(output)
    print(f"Generated tables: {OUTPUT_FILE}")

    # Count tables
    table_count = output.count(r"\begin{table}")
    print(f"Total tables: {table_count}")

    # Verify brace balance
    open_braces = output.count('{')
    close_braces = output.count('}')
    print(f"Brace balance: {open_braces} open, {close_braces} close, diff={open_braces-close_braces}")


if __name__ == "__main__":
    main()
