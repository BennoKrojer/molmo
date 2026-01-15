#!/usr/bin/env python3
"""
Generate LaTeX tables for interpretation type breakdown - V4.
Cleaner design focusing on readability:
- Main table: All layers with full phrase examples
- Summary row showing early vs late percentage shift
"""

import pickle
from pathlib import Path
import re

DATA_DIR = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/analysis_results/layer_evolution")
OUTPUT_FILE = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo/paper/figures/interpretation_type_tables.tex")

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

NUM_WORDS = 10  # Top N words per category
NUM_PHRASES = 3  # Number of example phrases per word


def escape_latex(text):
    """Escape special LaTeX characters and clean corrupted data."""
    if text is None or text == "-" or not str(text).strip():
        return None

    text = str(text).strip()

    # Remove corrupt patterns
    text = text.replace('\\\\', '').replace('\\', '')
    text = re.sub(r'^["\']+', '', text)  # Remove leading quotes
    text = re.sub(r'---+', '-', text)  # Fix multiple dashes in words

    # Skip clearly corrupted entries
    if len(text) < 2 or text.startswith('-') or not text[0].isalpha():
        return None

    replacements = {
        '—': '--', '–': '-',
        ''': "'", ''': "'", '"': '"', '"': '"', ''': "'", ''': "'",
        'é': 'e', 'ç': 'c', 'ñ': 'n', 'ü': 'u', 'ö': 'o', 'ä': 'a',
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_phrase(phrase, word, max_len=45):
    """Format a phrase with bold word."""
    if not phrase or phrase == "-":
        return None

    phrase = str(phrase).strip()
    phrase = phrase.replace(f'*{word}*', word)

    # Skip if phrase is just the word or very short
    if phrase.lower() == word.lower() or len(phrase) < len(word) + 3:
        return None

    needs_start = False
    needs_end = False

    if len(phrase) > max_len:
        word_pos = phrase.lower().find(word.lower())
        if word_pos == -1:
            phrase = phrase[:max_len-3]
            needs_end = True
        else:
            if word_pos > max_len // 2:
                start = max(0, word_pos - 12)
                phrase = phrase[start:]
                needs_start = True
            if len(phrase) > max_len - 3:
                phrase = phrase[:max_len-3]
                needs_end = True

    phrase_escaped = escape_latex(phrase)
    if not phrase_escaped:
        return None

    word_escaped = escape_latex(word)
    if not word_escaped:
        return None

    pattern = re.compile(re.escape(word_escaped), re.IGNORECASE)
    result = pattern.sub(r'\\textbf{' + word_escaped + r'}', phrase_escaped, count=1)

    if needs_start:
        result = "..." + result
    if needs_end:
        result = result + "..."

    return result


def load_data(model_key, variant_suffix=""):
    """Load sunburst data from pickle file."""
    pkl_path = DATA_DIR / f"sunburst_data_{model_key}{variant_suffix}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_percentages(data):
    """Extract category percentages from data."""
    total = sum(data.get(cat, {}).get('count', 0) for cat in ['Concrete', 'Abstract', 'Global'])
    if total == 0:
        return {}
    return {
        cat: round(100 * data.get(cat, {}).get('count', 0) / total)
        for cat in ['Concrete', 'Abstract', 'Global']
    }


def extract_words_with_phrases(data, num_words=NUM_WORDS, num_phrases=NUM_PHRASES):
    """Extract top words with their best phrases."""
    total = sum(data.get(cat, {}).get('count', 0) for cat in ['Concrete', 'Abstract', 'Global'])
    if total == 0:
        return {}

    result = {}
    for cat_name in ['Concrete', 'Abstract', 'Global']:
        cat_data = data.get(cat_name, {})
        if not cat_data:
            continue

        pct = round(100 * cat_data.get('count', 0) / total)
        words_dict = cat_data.get('words', {})

        words_list = []
        for word, word_info in words_dict.items():
            clean_word = escape_latex(word)
            if not clean_word:
                continue

            phrases_dict = word_info.get('phrases', {})
            # Get top phrases, filtering out bad ones
            sorted_phrases = sorted(phrases_dict.items(), key=lambda x: x[1], reverse=True)
            good_phrases = []
            for phrase_text, _ in sorted_phrases:
                formatted = format_phrase(phrase_text, word)
                if formatted and formatted not in good_phrases:
                    good_phrases.append(formatted)
                if len(good_phrases) >= num_phrases:
                    break

            if good_phrases:  # Only include words that have valid phrases
                words_list.append({
                    'word': clean_word,
                    'count': word_info.get('count', 0),
                    'phrases': good_phrases
                })

        words_list.sort(key=lambda x: x['count'], reverse=True)
        result[cat_name] = {
            'percentage': pct,
            'words': words_list[:num_words]
        }

    return result


def generate_model_table(model_key, model_display, label_key, late_layers):
    """Generate a clean table for one model."""

    data_all = load_data(model_key, "")
    data_early = load_data(model_key, "_early")
    data_late = load_data(model_key, "_late")

    if not data_all:
        return ""

    cat_all = extract_words_with_phrases(data_all)
    pct_early = extract_percentages(data_early) if data_early else {}
    pct_late = extract_percentages(data_late) if data_late else {}

    if not cat_all:
        return ""

    lines = []
    lines.append(f"% {model_display}")
    lines.append(r"\begin{table*}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")

    # Clean column structure: Type | Word (Count) | Example Phrases
    lines.append(r"\begin{tabular}{@{} p{2.2cm} p{2.0cm} p{11.5cm} @{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{Word (n)} & \textbf{Top Visual Genome Phrases} \\")
    lines.append(r"\midrule")

    for cat_idx, cat_name in enumerate(['Concrete', 'Abstract', 'Global']):
        if cat_name not in cat_all:
            continue

        cat_data = cat_all[cat_name]
        pct = cat_data['percentage']
        words = cat_data['words']

        # Layer shift info
        early_pct = pct_early.get(cat_name, pct)
        late_pct = pct_late.get(cat_name, pct)

        for word_idx, word_data in enumerate(words):
            word = word_data['word']
            count = word_data['count']
            phrases = word_data['phrases']

            # Format phrases with semicolon separator
            phrases_str = "; ".join(phrases) if phrases else "---"

            # First row shows category with percentage and layer shift (always show for consistency)
            if word_idx == 0:
                shift = f" ({early_pct}\\%$\\rightarrow${late_pct}\\%)"
                cat_cell = f"\\textbf{{{cat_name}}} ({pct}\\%){shift}"
            else:
                cat_cell = ""

            lines.append(f"{cat_cell} & {word} ({count}) & {phrases_str} \\\\")

        # Add spacing between categories
        if cat_idx < 2:
            lines.append(r"\addlinespace[4pt]")
            lines.append(r"\midrule")
            lines.append(r"\addlinespace[2pt]")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Caption with layer info
    lines.append(f"\\caption{{\\textbf{{{model_display}}} interpretation type breakdown. "
                 f"Top {NUM_WORDS} words per category with occurrence counts and most frequent Visual Genome phrases. "
                 f"Percentages in parentheses show early$\\rightarrow$late layer shift (layers 0-2 vs {late_layers}).}}")
    lines.append(f"\\label{{tab:interp_{label_key}}}")
    lines.append(r"\end{table*}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate all interpretation type tables."""
    lines = []
    lines.append("% Auto-generated interpretation type tables - V4")
    lines.append("% Clean design: Category | Word | Example Phrases")
    lines.append("% Layer shift shown in category header")
    lines.append("")
    lines.append(r"\clearpage")
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
