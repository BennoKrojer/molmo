#!/usr/bin/env python3
"""
Generate sunburst data from LLM judge contextual results.

Extracts words categorized as Concrete/Abstract/Global and their
preceding context phrases from captions.
"""

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path


def find_word_in_text(word, text):
    """Find the position of a word in text (case-insensitive)."""
    pattern = r'\b' + re.escape(word) + r'\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def extract_preceding_context(caption, word, max_words=3):
    """
    Extract preceding context for a word in a caption.
    Returns format like "...the black *tent*" or just "*tent*" if at start.
    Uses max 3 preceding words (autoregressive LLM sees only past).
    """
    pos = find_word_in_text(word, caption)
    if pos is None:
        return None

    start_idx, end_idx = pos
    preceding_text = caption[:start_idx].strip()

    if not preceding_text:
        # Word is at the start
        return f"*{word}*"

    # Split into words and take last N (max 3 for autoregressive context)
    words = preceding_text.split()
    if len(words) > max_words:
        context = " ".join(words[-max_words:])
        return f"...{context} *{word}*"
    else:
        return f"{preceding_text} *{word}*"


def load_contextual_nn_results(base_dir, model_name, layer):
    """Load contextual NN results to get caption data."""
    nn_dir = base_dir / "contextual_nearest_neighbors"

    # Try different naming patterns
    patterns = [
        f"train_mlp-only_pixmo_cap_resize_{model_name}_step12000-unsharded",
        f"{model_name}",
    ]

    for pattern in patterns:
        model_dir = nn_dir / pattern
        if model_dir.exists():
            # Look for layer file
            layer_file = model_dir / f"contextual_neighbors_visual{layer}_allLayers.json"
            if layer_file.exists():
                with open(layer_file) as f:
                    return json.load(f)

    return None


def extract_captions_for_words(nn_data, words_set):
    """
    Extract captions containing specific words from NN data.
    OPTIMIZED: O(captions) instead of O(words × captions)
    Returns dict: word -> list of captions
    """
    word_captions = defaultdict(list)

    if not nn_data or 'results' not in nn_data:
        return word_captions

    # Convert words_set to lowercase for matching
    words_lower = {w.lower() for w in words_set}

    for result in nn_data['results']:
        chunks = result.get('chunks', [])
        for chunk in chunks:
            patches = chunk.get('patches', [])
            for patch in patches:
                neighbors = patch.get('nearest_contextual_neighbors', [])
                for neighbor in neighbors:
                    caption = neighbor.get('caption', '')
                    if not caption:
                        continue

                    # Tokenize caption once, check all words at once
                    caption_lower = caption.lower()
                    caption_words = set(re.findall(r'\b\w+\b', caption_lower))

                    # Find intersection with target words
                    matches = caption_words & words_lower
                    for word in matches:
                        word_captions[word].append(caption)

    return word_captions


def main():
    base_dir = Path(__file__).parent.parent.parent.parent
    results_dir = base_dir / "analysis_results/llm_judge_contextual_nn"
    output_path = base_dir / "analysis_results/layer_evolution/sunburst_data.pkl"

    # Categories to collect
    categories = {
        'Concrete': defaultdict(lambda: {'count': 0, 'captions': []}),
        'Abstract': defaultdict(lambda: {'count': 0, 'captions': []}),
        'Global': defaultdict(lambda: {'count': 0, 'captions': []})
    }

    category_totals = {'Concrete': 0, 'Abstract': 0, 'Global': 0}

    # Process all LLM judge results (excluding ablations)
    result_dirs = list(results_dir.glob("llm_judge_*_contextual*_gpt5_cropped"))
    result_dirs = [d for d in result_dirs if '/ablations/' not in str(d) and d.is_dir()]

    print(f"Found {len(result_dirs)} LLM judge result directories", flush=True)

    all_word_captions = defaultdict(list)
    nn_data_cache = {}  # Cache NN data per model to avoid reloading

    # First pass: collect all words and count by category
    all_words_global = set()
    for result_dir in result_dirs:
        result_file = result_dir / "results_validation.json"
        if not result_file.exists():
            continue

        # Parse model name and layer from directory
        dir_name = result_dir.name
        parts = dir_name.replace("llm_judge_", "").replace("_gpt5_cropped", "")
        layer_match = re.search(r'_contextual(\d+)$', parts)
        if layer_match:
            layer = int(layer_match.group(1))
            model_name = parts[:layer_match.start()]
        else:
            continue

        print(f"Processing {model_name} layer {layer}...", flush=True)

        with open(result_file) as f:
            data = json.load(f)

        results = data.get('results', [])

        # Collect all words
        for result in results:
            gpt = result.get('gpt_response', {})
            for key in ['concrete_words', 'abstract_words', 'global_words']:
                words = gpt.get(key, [])
                if words:
                    all_words_global.update(w.lower() for w in words)

        # Load contextual NN data ONCE per model (use layer 0 or first available)
        if model_name not in nn_data_cache:
            # Try layer 0 first, then any available layer
            for try_layer in [0, 1, 2, 4, 8, 16, 24, 30, 31]:
                nn_data = load_contextual_nn_results(base_dir / "analysis_results", model_name, try_layer)
                if nn_data:
                    print(f"  Loaded NN data for {model_name} (layer {try_layer})", flush=True)
                    nn_data_cache[model_name] = nn_data
                    break
            else:
                nn_data_cache[model_name] = None

        # Count words by category
        for result in results:
            gpt = result.get('gpt_response', {})

            for word in gpt.get('concrete_words', []):
                word_lower = word.lower()
                categories['Concrete'][word_lower]['count'] += 1
                category_totals['Concrete'] += 1

            for word in gpt.get('abstract_words', []):
                word_lower = word.lower()
                categories['Abstract'][word_lower]['count'] += 1
                category_totals['Abstract'] += 1

            for word in gpt.get('global_words', []):
                word_lower = word.lower()
                categories['Global'][word_lower]['count'] += 1
                category_totals['Global'] += 1

    # Extract captions from cached NN data
    print(f"\nExtracting captions from {len(nn_data_cache)} cached models...", flush=True)
    for model_name, nn_data in nn_data_cache.items():
        if nn_data:
            word_caps = extract_captions_for_words(nn_data, all_words_global)
            for word, caps in word_caps.items():
                all_word_captions[word].extend(caps)
            print(f"  {model_name}: extracted captions for {len(word_caps)} words", flush=True)

    # Now COUNT phrase occurrences for each word (not just list examples)
    print("\nCounting phrase context occurrences...", flush=True)
    for cat_name, words_dict in categories.items():
        for word, info in words_dict.items():
            captions = all_word_captions.get(word, [])

            # Count occurrences of each unique phrase context
            phrase_counts = defaultdict(int)
            for caption in captions:
                context = extract_preceding_context(caption, word)
                if context and len(context) < 60:  # Reasonable length
                    phrase_counts[context] += 1

            # Sort by count and store as dict {phrase: count}
            sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
            info['phrases'] = {phrase: count for phrase, count in sorted_phrases[:10]}  # Top 10 by count

            # Also keep total phrase count for verification
            info['phrase_total'] = sum(phrase_counts.values())

    # Build final data structure
    data = {}
    for cat_name in ['Concrete', 'Abstract', 'Global']:
        # Sort words by count
        sorted_words = sorted(
            categories[cat_name].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        data[cat_name] = {
            'count': category_totals[cat_name],
            'words': {word: info for word, info in sorted_words[:50]}  # Top 50 words
        }

        print(f"\n{cat_name}: {category_totals[cat_name]} total, {len(sorted_words)} unique words")
        print(f"  Top 5 words: {[w for w, _ in sorted_words[:5]]}")
        # Show phrase counts for top word
        if sorted_words:
            top_word, top_info = sorted_words[0]
            phrases = top_info.get('phrases', {})
            print(f"  '{top_word}' top phrases: {list(phrases.items())[:3]}")

    # Save pickle
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved sunburst data to {output_path}")


if __name__ == '__main__':
    main()
