#!/usr/bin/env python3
"""
Generate LaTeX tables for interpretation type breakdown - V3.
Proper academic design:
- Full page width
- Multiple phrase examples per word
- Proper spacing
- Grouped by model (all/early/late in one table)
"""

import pickle
from pathlib import Path
import re

DATA_DIR = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/analysis_results/layer_evolution")
OUTPUT_FILE = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/paper/figures/interpretation_type_tables.tex")

# Model configurations
MODELS = [
    ("olmo-7b_vit-l-14-336", "OLMo-7B + ViT-L", "olmo_vitl", "31, 30, 24"),
    ("olmo-7b_siglip", "OLMo-7B + SigLIP", "olmo_siglip", "31, 30, 24"),
    ("olmo-7b_dinov2-large-336", "OLMo-7B + DINOv2", "olmo_dinov2", "31, 30, 24"),
    ("llama3-8b_vit-l-14-336", "LLaMA3-8B + ViT-L", "llama_vitl", "31, 30, 24"),
    ("llama3-8b_siglip", "LLaMA3-8B + SigLIP", "llama_siglip", "31, 30, 24"),
    ("llama3-8b_dinov2-large-336", "LLaMA3-8B + DINOv2", "llama_dinov2", "31, 30, 24"),
    ("qwen2-7b_vit-l-14-336_seed10", "Qwen2-7B + ViT-L", "qwen2_vitl", "27, 26, 24"),
    ("qwen2-7b_siglip", "Qwen2-7B + SigLIP", "qwen2_siglip", "27, 26, 24"),
    ("qwen2-7b_dinov2-large-336", "Qwen2-7B + DINOv2", "qwen2_dinov2", "27, 26, 24"),
    ("qwen2vl", "Qwen2-VL-7B", "qwen2vl", "27, 26, 24"),
]

NUM_WORDS = 8  # Top N words per category
NUM_PHRASES = 2  # Number of example phrases per word


def escape_latex(text):
    """Escape special LaTeX characters."""
    if text is None or text == "-":
        return "---"

    # Remove corrupt backslash sequences
    text = text.replace('\\\\', '').replace('\\', '')

    replacements = {
        '—': '---', '–': '--',
        ''': "'", ''': "'", '"': '"', '"': '"',
        ''': "'", ''': "'",
        'é': "e", 'ç': "c", 'ñ': "n", 'ü': "u", 'ö': "o", 'ä': "a",
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_phrase(phrase, word, max_len=50):
    """Format a phrase with bold word, truncating intelligently."""
    if phrase is None or phrase == "-" or not phrase.strip():
        return None

    phrase = phrase.strip().replace(f'*{word}*', word)

    needs_start = False
    needs_end = False

    if len(phrase) > max_len:
        word_pos = phrase.lower().find(word.lower())
        if word_pos == -1:
            phrase = phrase[:max_len-3]
            needs_end = True
        else:
            if word_pos > max_len // 2:
                start = max(0, word_pos - 15)
                phrase = phrase[start:]
                needs_start = True
            if len(phrase) > max_len - 3:
                phrase = phrase[:max_len-3]
                needs_end = True

    phrase_escaped = escape_latex(phrase)
    word_escaped = escape_latex(word)

    pattern = re.compile(re.escape(word_escaped), re.IGNORECASE)
    result = pattern.sub(r'\\textbf{' + word_escaped + r'}', phrase_escaped, count=1)

    if needs_start:
        result = "..." + result
    if needs_end:
        result = result + "..."

    return result


def load_data(model_key, variant_suffix):
    """Load sunburst data from pickle file."""
    pkl_path = DATA_DIR / f"sunburst_data_{model_key}{variant_suffix}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_category_data(data):
    """Extract structured category data from raw pickle data."""
    total = sum(data.get(cat, {}).get('count', 0) for cat in ['Concrete', 'Abstract', 'Global'])
    if total == 0:
        return None

    categories = {}
    for cat_name in ['Concrete', 'Abstract', 'Global']:
        cat_data = data.get(cat_name, {})
        if not cat_data or cat_data.get('count', 0) == 0:
            continue

        pct = round(100 * cat_data['count'] / total)
        words_dict = cat_data.get('words', {})

        words_list = []
        for word, word_info in words_dict.items():
            phrases_dict = word_info.get('phrases', {})
            # Get top N phrases by count
            sorted_phrases = sorted(phrases_dict.items(), key=lambda x: x[1], reverse=True)
            top_phrases = [p[0] for p in sorted_phrases[:NUM_PHRASES]]

            words_list.append({
                'word': word,
                'count': word_info.get('count', 0),
                'phrases': top_phrases
            })

        words_list.sort(key=lambda x: x['count'], reverse=True)
        categories[cat_name] = {'percentage': pct, 'words': words_list[:NUM_WORDS]}

    return categories


def generate_model_table(model_key, model_display, label_key, late_layers):
    """Generate a combined table for one model (all/early/late side by side)."""

    # Load all three variants
    data_all = load_data(model_key, "")
    data_early = load_data(model_key, "_early")
    data_late = load_data(model_key, "_late")

    if not all([data_all, data_early, data_late]):
        return ""

    cat_all = extract_category_data(data_all)
    cat_early = extract_category_data(data_early)
    cat_late = extract_category_data(data_late)

    if not all([cat_all, cat_early, cat_late]):
        return ""

    lines = []
    lines.append(f"% {model_display}")
    lines.append(r"\begin{table*}[!htbp]")  # Better placement control
    lines.append(r"\centering")
    lines.append(r"\footnotesize")  # Consistent small font
    lines.append(f"\\caption{{\\textbf{{{model_display}}} interpretation type breakdown. "
                 f"Top {NUM_WORDS} words per category with occurrence counts and example Visual Genome phrases. "
                 f"Early = layers 0,1,2; Late = layers {late_layers}.}}")
    lines.append(f"\\label{{tab:interp_{label_key}}}")
    lines.append(r"\vspace{1mm}")

    # Full-width table - cleaner header structure
    lines.append(r"\setlength{\tabcolsep}{3pt}")  # Tighter column spacing
    lines.append(r"\begin{tabular}{@{}p{1.3cm} p{1.6cm} p{4.0cm} | p{1.6cm} p{1.6cm} p{4.0cm}@{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{\textit{All Layers}} & \multicolumn{3}{c}{\textit{Early vs Late Comparison}} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(l){4-6}")
    lines.append(r"\textbf{Type} & \textbf{Word (\#)} & \textbf{Example Phrases} & \textbf{Early} & \textbf{Late} & \textbf{Late Example} \\")
    lines.append(r"\midrule")

    # Generate rows for each category
    for cat_name in ['Concrete', 'Abstract', 'Global']:
        if cat_name not in cat_all:
            continue

        all_data = cat_all[cat_name]
        early_data = cat_early.get(cat_name, {'percentage': 0, 'words': []})
        late_data = cat_late.get(cat_name, {'percentage': 0, 'words': []})

        # First row shows category name and percentages
        first_word_all = all_data['words'][0] if all_data['words'] else None
        first_word_early = early_data['words'][0] if early_data['words'] else None
        first_word_late = late_data['words'][0] if late_data['words'] else None

        if first_word_all:
            phrases_str = format_phrases(first_word_all)
            word_all = f"{escape_latex(first_word_all['word'])} ({first_word_all['count']})"
        else:
            phrases_str = "---"
            word_all = "---"

        early_str = f"{escape_latex(first_word_early['word'])} ({first_word_early['count']})" if first_word_early else "---"
        late_str = f"{escape_latex(first_word_late['word'])} ({first_word_late['count']})" if first_word_late else "---"
        late_phrases = format_phrases(first_word_late) if first_word_late else "---"

        cat_label = f"\\textbf{{{cat_name}}} ({all_data['percentage']}\\%)"
        lines.append(f"{cat_label} & {word_all} & {phrases_str} & {early_str} & {late_str} & {late_phrases} \\\\")

        # Remaining words (skip first, already shown)
        max_rows = max(len(all_data['words']), len(early_data['words']), len(late_data['words'])) - 1
        for i in range(1, min(max_rows + 1, NUM_WORDS)):
            word_all_i = all_data['words'][i] if i < len(all_data['words']) else None
            word_early_i = early_data['words'][i] if i < len(early_data['words']) else None
            word_late_i = late_data['words'][i] if i < len(late_data['words']) else None

            col1 = ""  # Empty category column for continuation rows
            col2 = f"{escape_latex(word_all_i['word'])} ({word_all_i['count']})" if word_all_i else ""
            col3 = format_phrases(word_all_i) if word_all_i else ""
            col4 = f"{escape_latex(word_early_i['word'])} ({word_early_i['count']})" if word_early_i else ""
            col5 = f"{escape_latex(word_late_i['word'])} ({word_late_i['count']})" if word_late_i else ""
            col6 = format_phrases(word_late_i) if word_late_i else ""

            lines.append(f" & {col2} & {col3} & {col4} & {col5} & {col6} \\\\")

        # Add visual separator between categories (except last)
        if cat_name != 'Global':
            lines.append(r"\addlinespace[3pt]")
            lines.append(r"\midrule")
            lines.append(r"\addlinespace[2pt]")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append(r"\vspace{5mm}")  # Space before next table
    lines.append("")

    return "\n".join(lines)


def format_phrases(word_data):
    """Format multiple phrases for a word."""
    if not word_data or not word_data.get('phrases'):
        return "---"

    formatted = []
    for phrase in word_data['phrases'][:NUM_PHRASES]:
        fp = format_phrase(phrase, word_data['word'])
        if fp:
            formatted.append(fp)

    if not formatted:
        return "---"

    # Join with semicolon for multiple phrases
    return "; ".join(formatted)


def main():
    """Generate all interpretation type tables."""
    lines = []
    lines.append("% Auto-generated interpretation type tables - V3")
    lines.append("% Full-width tables with multiple phrase examples")
    lines.append("% Each model has one table combining all/early/late")
    lines.append("")
    lines.append(r"\clearpage")  # Start on fresh page
    lines.append("")

    for model_key, model_display, label_key, late_layers in MODELS:
        table = generate_model_table(model_key, model_display, label_key, late_layers)
        if table:
            lines.append(table)

    output = "\n".join(lines)
    OUTPUT_FILE.write_text(output)
    print(f"Generated: {OUTPUT_FILE}")

    # Validation
    table_count = output.count(r'\begin{table*}')
    open_b = output.count('{')
    close_b = output.count('}')
    print(f"Tables: {table_count}")
    print(f"Braces: {open_b}/{close_b} (diff={open_b-close_b})")


if __name__ == "__main__":
    main()
