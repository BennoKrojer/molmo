#!/usr/bin/env python3
"""
Dynamic LatentLens Study: Evolutionary search for better phrase descriptions.

Instead of searching a fixed corpus, we:
1. Start with current LatentLens top phrases (from VG corpus)
2. Use GPT-4o to generate variations/elaborations
3. Compute contextual embeddings of generated phrases
4. Score by cosine similarity to visual token
5. Keep top-3, iterate

This is a small-scale study with 5 examples to validate the concept.
"""

import argparse
import json
import os
import gc
import time
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


def load_llm_judge_results(results_path):
    """Load LLM judge results and filter to interpretable patches."""
    with open(results_path) as f:
        data = json.load(f)

    interpretable = []
    for result in data["results"]:
        if result.get("interpretable", False):
            interpretable.append(result)

    return interpretable, data


def load_contextual_nn_results(nn_path, image_idx, patch_row, patch_col):
    """Load the LatentLens results for a specific patch."""
    with open(nn_path) as f:
        data = json.load(f)

    for img_result in data["results"]:
        if img_result["image_idx"] != image_idx:
            continue

        for chunk in img_result["chunks"]:
            for patch in chunk["patches"]:
                if patch["patch_row"] == patch_row and patch["patch_col"] == patch_col:
                    return patch["nearest_contextual_neighbors"], img_result["ground_truth_caption"]

    return None, None


def extract_visual_token_embedding(model, preprocessor, dataset, image_idx, patch_row, patch_col,
                                    visual_layer, device, use_n_token_only):
    """Extract the visual token embedding for a specific patch."""
    example_data = dataset.get(image_idx, np.random)
    prompt = "Describe this image in detail."
    example = {"image": example_data["image"], "messages": [prompt]}
    batch = preprocessor(example, rng=np.random)

    # Move to device
    images = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
    image_masks = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
    image_input_idx = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None

    # Get grid dimensions
    patches_per_chunk = image_input_idx.shape[2]
    grid_size = int(math.sqrt(patches_per_chunk))
    patch_idx = patch_row * grid_size + patch_col

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            if visual_layer == 0:
                # Vision backbone output
                feats, _ = model.vision_backbone(images, image_masks, return_tokens_before_MLP=True)
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    feats = feats[:, :, :use_n_token_only, :]
                embedding = feats[0, 0, patch_idx, :].float().cpu().numpy()
            else:
                # LLM hidden states
                output = model(
                    input_ids=input_ids,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    output_hidden_states=True,
                    last_logits_only=False,
                )
                hidden_states = output.hidden_states

                # Get the position of this patch in the sequence
                flat_pos = image_input_idx.view(1, -1)[0]
                token_pos = flat_pos[patch_idx].item()

                layer_idx = min(visual_layer, len(hidden_states) - 1)
                embedding = hidden_states[layer_idx][0, token_pos, :].float().cpu().numpy()

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    # Also get the original image for visualization
    image_source = example_data["image"]
    if isinstance(image_source, str):
        original_image = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, np.ndarray):
        original_image = Image.fromarray(image_source)
    else:
        original_image = image_source  # Already PIL Image

    return embedding, original_image, grid_size


def compute_contextual_embedding(llm_model, tokenizer, phrase, layer_idx, device, target_token=None):
    """
    Compute contextual embedding for a phrase at a specific layer.

    If target_token is provided, finds that token in the phrase and returns
    its embedding. Otherwise returns the last token's embedding.
    """
    # Tokenize
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm_model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    layer = min(layer_idx, len(hidden_states) - 1)

    # Find the position of the target token
    if target_token is not None:
        # Decode each token to find where target appears
        input_ids = inputs["input_ids"][0]
        target_idx = None
        for idx in range(len(input_ids)):
            decoded = tokenizer.decode([input_ids[idx].item()])
            if decoded.strip() == target_token.strip():
                target_idx = idx
                break
            # Also check without stripping (for tokens with leading space)
            if decoded == target_token:
                target_idx = idx
                break

        if target_idx is not None:
            embedding = hidden_states[layer][0, target_idx, :].float().cpu().numpy()
        else:
            # Fallback: use last token if target not found
            seq_len = inputs["attention_mask"].sum().item()
            embedding = hidden_states[layer][0, seq_len - 1, :].float().cpu().numpy()
    else:
        # No target specified, use last token
        seq_len = inputs["attention_mask"].sum().item()
        embedding = hidden_states[layer][0, seq_len - 1, :].float().cpu().numpy()

    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def find_worst_context(visual_embedding, token, embeddings_cache, top_n=1):
    """
    Find the worst (lowest similarity) context for a given token from the embeddings cache.

    Returns: list of (phrase, similarity) tuples, sorted by similarity (ascending)
    """
    token_key = token if token.startswith(' ') else f' {token}'
    token_key_stripped = token.strip()

    # Try both versions of the token
    indices = None
    for key in [token_key, token_key_stripped, token.strip()]:
        if key in embeddings_cache['token_to_indices']:
            indices = embeddings_cache['token_to_indices'][key]
            break

    if indices is None:
        return []

    visual_emb = torch.tensor(visual_embedding, dtype=torch.float32)
    visual_emb = visual_emb / visual_emb.norm()

    results = []
    for idx in indices:
        emb = embeddings_cache['embeddings'][idx].float()
        emb = emb / emb.norm()
        sim = torch.dot(visual_emb, emb).item()
        caption = embeddings_cache['metadata'][idx]['caption']
        results.append((caption, sim))

    # Sort by similarity (ascending = worst first)
    results.sort(key=lambda x: x[1])
    return results[:top_n]


def generate_variations(client, seed_phrases, target_tokens, image_description, num_variations=10):
    """
    Use GPT-4o to generate phrase variations.

    IMPORTANT: We only vary the PRECEDING CONTEXT, keeping the target token at the end.
    This is because LLMs are autoregressive - only tokens before position i affect
    the contextual embedding at position i.
    """
    # Build seed examples showing the pattern: "context... TARGET_TOKEN"
    seed_examples = []
    for phrase, token in zip(seed_phrases[:5], target_tokens[:5]):
        seed_examples.append(f'- "{phrase}" (target token: "{token.strip()}")')
    seed_text = "\n".join(seed_examples)

    prompt = f"""You are helping with a vision-language interpretability study.

We have visual tokens from an image region. Each phrase below ends with a TARGET TOKEN - this is the word whose contextual embedding we're measuring. The words BEFORE it provide context.

Current best-matching phrases:
{seed_text}

The full image shows: {image_description[:300]}...

Generate {num_variations} NEW phrase variations. CRITICAL RULES:
1. Each phrase MUST end with one of the target tokens shown above (pick any of the 5)
2. ONLY vary the words BEFORE the target token (the preceding context)
3. Include variations that are:
   - SHORTER (remove words, e.g., "building" alone, or "tall building")
   - LONGER (add descriptive words before, e.g., "old stone castle building")
   - DIFFERENT CONTEXT (synonyms, related concepts, e.g., "apartment building" vs "office building")
4. Keep phrases 1-8 words total

Output ONLY the phrases, one per line, no numbering or bullets."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=500
    )

    variations = response.choices[0].message.content.strip().split("\n")
    variations = [v.strip() for v in variations if v.strip()]

    return variations


def highlight_token_in_phrase(phrase: str, token: str) -> str:
    """
    Return phrase with the token marked using **bold** markers.
    Expands subwords to full words when possible.
    """
    if not phrase or not token:
        return phrase or ""

    token = token.strip()
    if not token:
        return phrase

    low_phrase = phrase.lower()
    low_tok = token.lower()
    idx = low_phrase.find(low_tok)

    if idx == -1:
        return f"{phrase} [**{token}**]"

    # Expand to word boundaries
    def is_word_char(ch):
        return ch.isalnum() or ch == '_' or ch == '-'

    start = idx
    end = idx + len(low_tok)

    # Expand left
    while start > 0 and is_word_char(phrase[start - 1]):
        start -= 1
    # Expand right
    while end < len(phrase) and is_word_char(phrase[end]):
        end += 1

    word = phrase[start:end]
    return phrase[:start] + f"**{word}**" + phrase[end:]


def draw_text_with_highlight(draw, text, x, y, font, font_bold, fill="black"):
    """Draw text with **bold** markers rendered in bold."""
    import re
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    current_x = x
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            # Bold part - draw with bold font and color
            bold_text = part[2:-2]
            draw.text((current_x, y), bold_text, fill="darkred", font=font_bold)
            bbox = draw.textbbox((current_x, y), bold_text, font=font_bold)
            current_x = bbox[2]
        else:
            draw.text((current_x, y), part, fill=fill, font=font)
            if part:
                bbox = draw.textbbox((current_x, y), part, font=font)
                current_x = bbox[2]


def create_visualization(image, patch_row, patch_col, grid_size,
                         original_phrases, original_tokens, evolved_phrases,
                         original_sims, evolved_sims,
                         output_path, worst_contexts=None):
    """Create a visualization PNG for manual evaluation with highlighted tokens.

    worst_contexts: list of (phrase, similarity, token) tuples for worst contexts
    """
    # Resize image to 512x512 for consistency
    img_size = 512
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Calculate patch bounds
    patch_size = img_size / grid_size
    x1 = int(patch_col * patch_size)
    y1 = int(patch_row * patch_size)
    x2 = int((patch_col + 1) * patch_size)
    y2 = int((patch_row + 1) * patch_size)

    # Create canvas - taller if we have worst contexts
    canvas_width = img_size + 700  # Image + text area
    canvas_height = img_size + 150 if worst_contexts else img_size + 100
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Paste image
    canvas.paste(image, (0, 50))

    # Draw bounding box
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x1, y1 + 50, x2, y2 + 50], outline="red", width=3)

    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except:
        font = ImageFont.load_default()
        font_bold = font
        font_title = font

    # Title
    draw.text((10, 10), f"Patch ({patch_row}, {patch_col}) - Grid {grid_size}x{grid_size}",
              fill="black", font=font_title)

    # Original phrases with highlighted tokens
    text_x = img_size + 20
    draw.text((text_x, 60), "CORPUS BEST:", fill="blue", font=font_title)
    y = 85
    for i, (phrase, token, sim) in enumerate(zip(original_phrases[:3], original_tokens[:3], original_sims[:3])):
        highlighted = highlight_token_in_phrase(phrase, token)
        if len(highlighted) > 55:
            highlighted = highlighted[:52] + "..."
        text = f"{i+1}. {highlighted} ({sim:.3f})"
        draw_text_with_highlight(draw, text, text_x, y, font, font_bold)
        y += 20

    # Evolved phrases with highlighted tokens
    draw.text((text_x, y + 10), "EVOLVED BEST:", fill="green", font=font_title)
    y += 32
    for i, (phrase, sim) in enumerate(zip(evolved_phrases[:3], evolved_sims[:3])):
        matched_token = None
        for token in original_tokens[:5]:
            if token.strip().lower() in phrase.lower():
                matched_token = token
                break
        if matched_token:
            highlighted = highlight_token_in_phrase(phrase, matched_token)
        else:
            highlighted = phrase
        if len(highlighted) > 55:
            highlighted = highlighted[:52] + "..."
        text = f"{i+1}. {highlighted} ({sim:.3f})"
        draw_text_with_highlight(draw, text, text_x, y, font, font_bold)
        y += 20

    # Worst contexts (if provided)
    if worst_contexts:
        draw.text((text_x, y + 10), "CORPUS WORST:", fill="red", font=font_title)
        y += 32
        for i, (phrase, sim, token) in enumerate(worst_contexts[:3]):
            highlighted = highlight_token_in_phrase(phrase, token)
            if len(highlighted) > 55:
                highlighted = highlighted[:52] + "..."
            text = f"{i+1}. {highlighted} ({sim:.3f})"
            draw_text_with_highlight(draw, text, text_x, y, font, font_bold, fill="darkred")
            y += 20

    # Summary
    best_orig = max(original_sims[:5]) if original_sims else 0
    best_evol = max(evolved_sims[:5]) if evolved_sims else 0
    worst_sim = worst_contexts[0][1] if worst_contexts else 0
    improvement = best_evol - best_orig
    gap = best_evol - worst_sim

    draw.text((text_x, y + 15), f"Improvement: {improvement:+.4f}  |  Best-Worst Gap: {gap:.3f}",
              fill="green" if improvement > 0 else "red", font=font_title)

    canvas.save(output_path)
    print(f"  Saved: {output_path}")


def run_evolutionary_search(visual_embedding, seed_phrases, seed_tokens, image_description,
                            llm_model, tokenizer, client, layer_idx, device,
                            num_rounds=3, variations_per_round=10, keep_top_k=3):
    """Run evolutionary search to find better matching phrases.

    Returns: (final_phrases, final_sims, evolution_history)
    """

    # Track evolution history for analysis
    evolution_history = {
        "seed_phrases": [],
        "rounds": []
    }

    # Compute similarities for seed phrases
    current_phrases = seed_phrases[:5]
    current_tokens = seed_tokens[:5]
    current_sims = []

    print(f"  Computing seed phrase similarities...")
    for phrase, token in zip(current_phrases, current_tokens):
        emb = compute_contextual_embedding(llm_model, tokenizer, phrase, layer_idx, device, target_token=token)
        sim = float(np.dot(visual_embedding, emb))
        current_sims.append(sim)
        evolution_history["seed_phrases"].append({
            "phrase": phrase,
            "token": token,
            "similarity": sim
        })

    # Track all phrases and sims for history
    all_phrases = list(current_phrases)
    all_sims = list(current_sims)

    print(f"  Seed best sim: {max(current_sims):.4f}")

    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}...")
        round_data = {
            "round": round_idx + 1,
            "new_phrases": [],
            "best_so_far": None
        }

        # Generate variations - use ALL 5 seed tokens for diversity
        variations = generate_variations(client, seed_phrases, seed_tokens, image_description, variations_per_round)
        print(f"    Generated {len(variations)} variations")

        # Compute similarities for new variations
        for phrase in variations:
            if phrase in all_phrases:
                continue  # Skip duplicates

            # Find which target token is in this phrase
            matched_token = None
            for token in seed_tokens:
                if token.strip().lower() in phrase.lower():
                    matched_token = token
                    break

            emb = compute_contextual_embedding(llm_model, tokenizer, phrase, layer_idx, device, target_token=matched_token)
            sim = float(np.dot(visual_embedding, emb))

            all_phrases.append(phrase)
            all_sims.append(sim)
            current_phrases.append(phrase)
            current_sims.append(sim)

            round_data["new_phrases"].append({
                "phrase": phrase,
                "token": matched_token,
                "similarity": sim
            })

        # Keep only top phrases for next round's seed
        combined = list(zip(current_phrases, current_sims))
        combined.sort(key=lambda x: x[1], reverse=True)
        current_phrases = [p for p, s in combined[:keep_top_k * 2]]
        current_sims = [s for p, s in combined[:keep_top_k * 2]]

        best_idx = np.argmax(all_sims)
        round_data["best_so_far"] = {
            "phrase": all_phrases[best_idx],
            "similarity": all_sims[best_idx]
        }
        evolution_history["rounds"].append(round_data)

        print(f"    Best sim so far: {max(all_sims):.4f}")

    # Return sorted results
    final = list(zip(all_phrases, all_sims))
    final.sort(key=lambda x: x[1], reverse=True)

    return [p for p, s in final], [s for p, s in final], evolution_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--visual-layer", type=int, default=16)
    parser.add_argument("--num-rounds", type=int, default=6,
                        help="Number of evolutionary rounds (default: 6)")
    parser.add_argument("--variations-per-round", type=int, default=20,
                        help="Number of variations to generate per round (default: 20)")
    parser.add_argument("--keep-top-k", type=int, default=5,
                        help="Number of top phrases to keep between rounds (default: 5)")
    parser.add_argument("--output-dir", type=str,
                        default="analysis_results/dynamic_latentlens_study")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths - OLMo + CLIP only
    ckpt_path = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded"
    llm_judge_path = f"analysis_results/llm_judge_contextual_nn/llm_judge_olmo-7b_vit-l-14-336_contextual{args.visual_layer}_gpt5_cropped/results_validation.json"
    nn_path = f"analysis_results/contextual_nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/contextual_neighbors_visual{args.visual_layer}_allLayers.json"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DYNAMIC LATENTLENS STUDY (OLMo + CLIP)")
    print("=" * 70)
    print(f"Visual layer: {args.visual_layer}")
    print(f"Num examples: {args.num_examples}")
    print(f"Evolution: {args.num_rounds} rounds × {args.variations_per_round} variations, keep top-{args.keep_top_k}")
    print(f"Output: {output_dir}")
    print()

    # Load LLM judge results
    print("Loading LLM judge results...")
    interpretable, judge_data = load_llm_judge_results(llm_judge_path)
    print(f"  Found {len(interpretable)} interpretable patches")

    # Select diverse examples (different images)
    selected = []
    seen_images = set()
    for result in interpretable:
        if result["image_idx"] not in seen_images:
            selected.append(result)
            seen_images.add(result["image_idx"])
        if len(selected) >= args.num_examples:
            break

    print(f"  Selected {len(selected)} examples from different images")
    print()

    # ===== PHASE 1: Extract all visual embeddings first (VLM) =====
    print("=" * 70)
    print("PHASE 1: Extracting visual embeddings")
    print("=" * 70)

    print("Loading VLM (OLMo + CLIP)...")
    cfg = TrainConfig.load(f"{ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    vlm_model = Molmo(cfg.model)

    ckpt_file = f"{ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024**3)
    if ckpt_size_gb < 1.0:
        vlm_model.reset_with_pretrained_weights()

    weights = torch.load(ckpt_file, map_location="cpu")
    vlm_model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    vlm_model = vlm_model.half().cuda().eval()

    model_config = ModelConfig.load(resource_path(ckpt_path, "config.yaml"), key="model", validate_paths=False)
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(model_config, for_inference=True, shuffle_messages=False,
                                          is_training=False, require_image_features=True)
    use_n_token_only = model_config.vision_backbone.use_n_token_only
    dataset = PixMoCap(split="validation", mode="captions")
    print("  VLM loaded")

    # Extract all visual embeddings and images
    extracted_data = []
    for i, example in enumerate(selected):
        image_idx = example["image_idx"]
        patch_row = example["patch_row"]
        patch_col = example["patch_col"]

        print(f"  Extracting {i+1}/{len(selected)}: image {image_idx}, patch ({patch_row}, {patch_col})")

        nn_results, gt_caption = load_contextual_nn_results(nn_path, image_idx, patch_row, patch_col)
        if nn_results is None:
            print(f"    WARNING: Could not find NN results, skipping")
            continue

        visual_embedding, orig_image, grid_size = extract_visual_token_embedding(
            vlm_model, preprocessor, dataset, image_idx, patch_row, patch_col,
            args.visual_layer, device, use_n_token_only
        )

        # Save visual embedding
        emb_path = output_dir / f"example{i}_visual_embedding.npy"
        np.save(emb_path, visual_embedding)

        extracted_data.append({
            "example": example,
            "nn_results": nn_results,
            "gt_caption": gt_caption,
            "visual_embedding": visual_embedding,
            "orig_image": orig_image,
            "grid_size": grid_size,
            "emb_path": emb_path
        })

    # Unload VLM to free GPU memory
    print("  Unloading VLM...")
    del vlm_model, preprocessor
    gc.collect()
    torch.cuda.empty_cache()

    # ===== PHASE 2: Evolutionary search (LLM) =====
    print()
    print("=" * 70)
    print("PHASE 2: Evolutionary search")
    print("=" * 70)

    print("Loading LLM (OLMo) for contextual embeddings...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-7B-1024-preview",
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-1024-preview", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  LLM loaded")

    # Initialize OpenAI client
    client = OpenAI()
    print("  OpenAI client ready")

    # Load embeddings cache for finding worst contexts
    cache_path = f"molmo_data/contextual_llm_embeddings_vg/allenai_OLMo-7B-1024-preview/layer_{args.visual_layer}/embeddings_cache.pt"
    print(f"Loading embeddings cache from {cache_path}...")
    embeddings_cache = torch.load(cache_path, weights_only=False)
    print(f"  Cache loaded: {len(embeddings_cache['token_to_indices'])} tokens")
    print()

    # Process each example
    results = []

    for i, data in enumerate(extracted_data):
        example = data["example"]
        image_idx = example["image_idx"]
        patch_row = example["patch_row"]
        patch_col = example["patch_col"]

        print("=" * 70)
        print(f"EXAMPLE {i + 1}/{len(extracted_data)}")
        print("=" * 70)
        print(f"Image {image_idx}, patch ({patch_row}, {patch_col})")
        print(f"LLM Judge words: {example['words']}")

        original_phrases = [r["caption"] for r in data["nn_results"][:5]]
        original_tokens = [r["token_str"] for r in data["nn_results"][:5]]
        original_sims = [r["similarity"] for r in data["nn_results"][:5]]
        print(f"Original top phrase: {original_phrases[0]} (token: '{original_tokens[0]}', sim={original_sims[0]:.4f})")

        # Run evolutionary search
        print(f"Running evolutionary search ({args.num_rounds} rounds, {args.variations_per_round} variations, keep top-{args.keep_top_k})...")
        evolved_phrases, evolved_sims, evolution_history = run_evolutionary_search(
            data["visual_embedding"], original_phrases, original_tokens, data["gt_caption"],
            llm_model, tokenizer, client, args.visual_layer, device,
            num_rounds=args.num_rounds,
            variations_per_round=args.variations_per_round,
            keep_top_k=args.keep_top_k
        )

        print(f"Evolved top phrase: {evolved_phrases[0]} (sim={evolved_sims[0]:.4f})")
        improvement = evolved_sims[0] - original_sims[0]
        print(f"Improvement: {improvement:+.4f}")

        # Find worst contexts for the evolved best token
        # Find which token is in the evolved best phrase
        evolved_token = None
        for token in original_tokens[:5]:
            if token.strip().lower() in evolved_phrases[0].lower():
                evolved_token = token
                break

        worst_contexts = []
        if evolved_token:
            worst_results = find_worst_context(data["visual_embedding"], evolved_token, embeddings_cache, top_n=3)
            for phrase, sim in worst_results:
                worst_contexts.append((phrase, sim, evolved_token))
            if worst_results:
                print(f"Worst context for '{evolved_token.strip()}': {worst_results[0][0]} (sim={worst_results[0][1]:.4f})")
                print(f"Best-Worst gap: {evolved_sims[0] - worst_results[0][1]:.4f}")

        # Create visualization with highlighted tokens
        viz_path = output_dir / f"example{i}_img{image_idx}_r{patch_row}_c{patch_col}.png"
        create_visualization(
            data["orig_image"], patch_row, patch_col, data["grid_size"],
            original_phrases, original_tokens, evolved_phrases,
            original_sims, evolved_sims,
            viz_path, worst_contexts=worst_contexts
        )

        # Save evolution history to text file
        history_path = output_dir / f"example{i}_evolution_history.txt"
        with open(history_path, "w") as f:
            f.write(f"DYNAMIC LATENTLENS EVOLUTION HISTORY\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Image: {image_idx}, Patch: ({patch_row}, {patch_col})\n")
            f.write(f"LLM Judge words: {example['words']}\n\n")

            f.write(f"SEED PHRASES (from VG corpus):\n")
            f.write(f"-" * 40 + "\n")
            for j, seed in enumerate(evolution_history["seed_phrases"]):
                highlighted = highlight_token_in_phrase(seed["phrase"], seed["token"])
                f.write(f"  {j+1}. {highlighted}\n")
                f.write(f"     Token: '{seed['token']}' | Similarity: {seed['similarity']:.4f}\n")
            f.write(f"\n")

            for round_data in evolution_history["rounds"]:
                f.write(f"ROUND {round_data['round']}:\n")
                f.write(f"-" * 40 + "\n")
                # Sort by similarity for display
                sorted_new = sorted(round_data["new_phrases"], key=lambda x: x["similarity"], reverse=True)
                for j, phrase_data in enumerate(sorted_new[:10]):  # Show top 10
                    if phrase_data["token"]:
                        highlighted = highlight_token_in_phrase(phrase_data["phrase"], phrase_data["token"])
                    else:
                        highlighted = phrase_data["phrase"]
                    marker = " *" if phrase_data["similarity"] == round_data["best_so_far"]["similarity"] else ""
                    f.write(f"  {j+1}. {highlighted} ({phrase_data['similarity']:.4f}){marker}\n")
                f.write(f"\n  Best so far: {round_data['best_so_far']['phrase']} ({round_data['best_so_far']['similarity']:.4f})\n\n")

            f.write(f"FINAL RESULT:\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Original best: {original_phrases[0]} ({original_sims[0]:.4f})\n")
            f.write(f"Evolved best:  {evolved_phrases[0]} ({evolved_sims[0]:.4f})\n")
            f.write(f"Improvement:   {improvement:+.4f}\n")

        print(f"  Saved history: {history_path}")

        # Store results
        result_entry = {
            "example_idx": i,
            "image_idx": image_idx,
            "patch_row": patch_row,
            "patch_col": patch_col,
            "visual_layer": args.visual_layer,
            "grid_size": data["grid_size"],
            "llm_judge_words": example["words"],
            "ground_truth_caption": data["gt_caption"][:500] if data["gt_caption"] else "",
            "original_phrases": original_phrases,
            "original_tokens": original_tokens,
            "original_similarities": original_sims,
            "evolved_phrases": evolved_phrases[:10],
            "evolved_similarities": evolved_sims[:10],
            "improvement": improvement,
            "evolution_history": evolution_history,
            "embedding_path": str(data["emb_path"]),
            "visualization_path": str(viz_path),
            "history_path": str(history_path)
        }
        # Add worst context info
        if worst_contexts:
            result_entry["worst_context_phrase"] = worst_contexts[0][0]
            result_entry["worst_context_sim"] = worst_contexts[0][1]
            result_entry["worst_context_token"] = worst_contexts[0][2]
            result_entry["best_worst_gap"] = evolved_sims[0] - worst_contexts[0][1]
        results.append(result_entry)

        print()

    # Save all results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "visual_layer": args.visual_layer,
                "num_examples": args.num_examples,
                "num_rounds": args.num_rounds,
                "model": "olmo-7b_vit-l-14-336"
            },
            "results": results
        }, f, indent=2)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    improvements = [r["improvement"] for r in results]
    print(f"Examples processed: {len(results)}")
    print(f"Average improvement: {np.mean(improvements):+.4f}")
    print(f"Max improvement: {max(improvements):+.4f}")
    print(f"Positive improvements: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
    print()
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")

    # Save summary text file for paper
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("DYNAMIC LATENTLENS STUDY - SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: OLMo-7B + CLIP ViT-L/14-336\n")
        f.write(f"Visual Layer: {args.visual_layer}\n")
        f.write(f"Evolution: {args.num_rounds} rounds × {args.variations_per_round} variations, keep top-{args.keep_top_k}\n")
        f.write(f"Examples: {len(results)}\n\n")

        f.write("AGGREGATE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average improvement: {np.mean(improvements):+.4f}\n")
        f.write(f"Max improvement:     {max(improvements):+.4f}\n")
        f.write(f"Min improvement:     {min(improvements):+.4f}\n")
        f.write(f"Positive:            {sum(1 for x in improvements if x > 0)}/{len(improvements)} ({100*sum(1 for x in improvements if x > 0)/len(improvements):.0f}%)\n")

        # Best-worst gap stats
        gaps = [r.get("best_worst_gap", 0) for r in results if r.get("best_worst_gap")]
        if gaps:
            f.write(f"\nBest-Worst Gap (evolved best vs corpus worst):\n")
            f.write(f"  Average gap:       {np.mean(gaps):.4f}\n")
            f.write(f"  Max gap:           {max(gaps):.4f}\n")
            f.write(f"  Min gap:           {min(gaps):.4f}\n")
        f.write("\n")

        f.write("ALL EXAMPLES:\n")
        f.write("=" * 70 + "\n\n")

        # Sort by improvement
        sorted_results = sorted(results, key=lambda x: x["improvement"], reverse=True)

        for r in sorted_results:
            f.write(f"Example {r['example_idx']}: Image {r['image_idx']}, Patch ({r['patch_row']}, {r['patch_col']})\n")
            f.write(f"-" * 50 + "\n")

            # Original
            orig_phrase = r["original_phrases"][0]
            orig_token = r["original_tokens"][0]
            orig_sim = r["original_similarities"][0]
            orig_highlighted = highlight_token_in_phrase(orig_phrase, orig_token)
            f.write(f"Original: {orig_highlighted}\n")
            f.write(f"          Similarity: {orig_sim:.4f}\n\n")

            # Evolved
            evol_phrase = r["evolved_phrases"][0]
            evol_sim = r["evolved_similarities"][0]
            # Find matching token
            matched_token = None
            for token in r["original_tokens"]:
                if token.strip().lower() in evol_phrase.lower():
                    matched_token = token
                    break
            if matched_token:
                evol_highlighted = highlight_token_in_phrase(evol_phrase, matched_token)
            else:
                evol_highlighted = evol_phrase
            f.write(f"Evolved:  {evol_highlighted}\n")
            f.write(f"          Similarity: {evol_sim:.4f}\n\n")

            # Worst context
            if r.get("worst_context_phrase"):
                worst_highlighted = highlight_token_in_phrase(r["worst_context_phrase"], r["worst_context_token"])
                f.write(f"Worst:    {worst_highlighted}\n")
                f.write(f"          Similarity: {r['worst_context_sim']:.4f}\n\n")
                f.write(f"Improvement: {r['improvement']:+.4f}  |  Best-Worst Gap: {r['best_worst_gap']:.4f}\n")
            else:
                f.write(f"Improvement: {r['improvement']:+.4f}\n")
            f.write(f"\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
