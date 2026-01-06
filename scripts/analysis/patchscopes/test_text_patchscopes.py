#!/usr/bin/env python3
"""
Reproduce Patchscopes paper results on TEXT tokens.

This test verifies our implementation by checking if we can recover
text token identities using the identity prompt + patching mechanism.

The paper shows high accuracy (>90%) for recovering token identity from
hidden states using this method. If our implementation works on text,
then the poor results on vision tokens are due to the nature of visual
representations, not bugs in our code.
"""

import argparse
import gc
import os
import torch
from pathlib import Path

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path


def get_transformer_blocks(model):
    if hasattr(model, 'module'):
        return model.module.transformer.blocks
    else:
        return model.transformer.blocks


def test_text_token_recovery(model, tokenizer, device, test_words, layers_to_test):
    """
    Test if we can recover text token identity via Patchscopes.

    For each word:
    1. Run "The word is {word}" through the model, capture hidden state at {word}
    2. Patch that hidden state into identity prompt "cat->cat; dog->dog; hello->hello; ?"
    3. Check if model predicts {word}
    """
    print("=" * 70)
    print("TEXT TOKEN RECOVERY TEST (Patchscopes Paper Reproduction)")
    print("=" * 70)

    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    # Identity prompt (same as paper)
    identity_prompt = "cat->cat; dog->dog; hello->hello; ?->"
    identity_tokens = tokenizer.encode(identity_prompt)
    identity_input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)

    # Find position of "?" in identity prompt (the position we'll patch)
    # The "?" should be near the end, before "->"
    print(f"\nIdentity prompt: \"{identity_prompt}\"")
    print(f"Identity tokens: {identity_tokens}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in identity_tokens]}")

    # Find ? position
    q_mark_token = tokenizer.encode("?")[-1]  # Get the "?" token
    patch_position = None
    for i, t in enumerate(identity_tokens):
        if t == q_mark_token:
            patch_position = i
            break

    if patch_position is None:
        # Fallback: use second-to-last position
        patch_position = len(identity_tokens) - 2

    print(f"Patch position: {patch_position} (token: '{tokenizer.decode([identity_tokens[patch_position]])}')")

    results_by_layer = {l: {"correct": 0, "total": 0, "details": []} for l in layers_to_test}

    for word in test_words:
        # Create source sentence
        source_sentence = f"The word is {word}"
        source_tokens = tokenizer.encode(source_sentence)
        source_input_ids = torch.tensor(source_tokens, device=device).unsqueeze(0)

        # Find position of the target word in source
        word_tokens = tokenizer.encode(word)
        # The word should be at the end
        word_position = len(source_tokens) - len(word_tokens)

        print(f"\n--- Testing word: '{word}' ---")
        print(f"Source: \"{source_sentence}\"")
        print(f"Source tokens: {[tokenizer.decode([t]) for t in source_tokens]}")
        print(f"Word position: {word_position}")

        # Get hidden states from source sentence
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                source_output = model(input_ids=source_input_ids, output_hidden_states=True)

        for layer_idx in layers_to_test:
            if layer_idx >= num_layers:
                continue

            # Get hidden state of the word at this layer
            word_hidden_state = source_output.hidden_states[layer_idx][0, word_position, :].clone()

            # Create patching hook
            def make_patch_hook(patch_hs, patch_pos):
                def hook_fn(module, args):
                    if isinstance(args, tuple) and len(args) > 0:
                        hs = args[0].clone()
                        hs[:, patch_pos, :] = patch_hs.unsqueeze(0)
                        return (hs,) + args[1:]
                    return None
                return hook_fn

            hook = make_patch_hook(word_hidden_state, patch_position)
            handle = blocks[layer_idx].register_forward_pre_hook(hook)

            # Run identity prompt with patched hidden state
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    patched_output = model(input_ids=identity_input_ids)

            handle.remove()

            # Get prediction at LAST position (predicts what comes after "->")
            logits = patched_output.logits[0, -1, :]
            top5_vals, top5_idxs = torch.topk(logits, k=5)
            top5_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top5_idxs]
            top1_token = top5_tokens[0]

            # Check if word is in top-5
            word_stripped = word.strip()
            in_top5 = any(t.lower() == word_stripped.lower() or
                        word_stripped.lower() in t.lower() or
                        t.lower() in word_stripped.lower()
                        for t in top5_tokens)
            in_top1 = (top1_token.lower() == word_stripped.lower() or
                      word_stripped.lower() in top1_token.lower() or
                      top1_token.lower() in word_stripped.lower())

            results_by_layer[layer_idx]["total"] += 1
            if in_top1:
                results_by_layer[layer_idx]["correct"] += 1

            status = "✓" if in_top1 else ("~" if in_top5 else "✗")
            results_by_layer[layer_idx]["details"].append({
                "word": word,
                "top5": top5_tokens,
                "correct": in_top1,
                "in_top5": in_top5
            })

            print(f"  Layer {layer_idx:2d}: top-5 = {top5_tokens[:3]}... {status}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Text Token Recovery Accuracy")
    print("=" * 70)

    for layer_idx in layers_to_test:
        if layer_idx >= num_layers:
            continue
        res = results_by_layer[layer_idx]
        acc = res["correct"] / res["total"] * 100 if res["total"] > 0 else 0
        top5_correct = sum(1 for d in res["details"] if d["in_top5"])
        top5_acc = top5_correct / res["total"] * 100 if res["total"] > 0 else 0
        print(f"  Layer {layer_idx:2d}: Top-1 = {acc:5.1f}% ({res['correct']}/{res['total']}), "
              f"Top-5 = {top5_acc:5.1f}% ({top5_correct}/{res['total']})")

    return results_by_layer


def test_same_model_patching(model, tokenizer, device):
    """
    Simpler test: patch token X's hidden state into prompt expecting X.

    Use: "cat->cat; dog->dog; X->" and patch X's embedding at "X" position.
    Should predict "X" with high confidence.
    """
    print("\n" + "=" * 70)
    print("SAME-POSITION PATCHING TEST")
    print("=" * 70)
    print("Testing if patching a token's own hidden state recovers that token.")

    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    test_words = ["apple", "house", "water", "green", "happy"]
    layers_to_test = [0, 8, 16, 24, num_layers - 1]
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    for word in test_words:
        print(f"\n--- Testing: '{word}' ---")

        # Create prompt with the word
        prompt = f"cat->cat; dog->dog; {word}->"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

        # Find position of the word (before "->")
        word_tokens = tokenizer.encode(word)
        # Word should be right before the final "->"
        arrow_tokens = tokenizer.encode("->")
        word_end_pos = len(tokens) - len(arrow_tokens)
        word_start_pos = word_end_pos - len(word_tokens)

        print(f"Prompt: \"{prompt}\"")
        print(f"Word '{word}' at positions {word_start_pos}-{word_end_pos-1}")

        # Get hidden states
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                output = model(input_ids=input_ids, output_hidden_states=True)

        # Baseline: what does model predict without patching?
        baseline_logits = output.logits[0, -1, :]
        baseline_top5 = [tokenizer.decode([idx.item()]).strip()
                        for idx in torch.topk(baseline_logits, k=5)[1]]
        print(f"Baseline prediction (no patch): {baseline_top5}")

        # Now test patching at different layers
        for layer_idx in layers_to_test:
            # Get the word's hidden state at this layer
            word_hs = output.hidden_states[layer_idx][0, word_start_pos, :].clone()

            # Patch it back to the same position
            def make_hook(hs, pos):
                def hook_fn(module, args):
                    if isinstance(args, tuple) and len(args) > 0:
                        h = args[0].clone()
                        h[:, pos, :] = hs.unsqueeze(0)
                        return (h,) + args[1:]
                    return None
                return hook_fn

            handle = blocks[layer_idx].register_forward_pre_hook(
                make_hook(word_hs, word_start_pos)
            )

            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    patched_output = model(input_ids=input_ids)

            handle.remove()

            patched_logits = patched_output.logits[0, -1, :]
            patched_top5 = [tokenizer.decode([idx.item()]).strip()
                          for idx in torch.topk(patched_logits, k=5)[1]]

            # Check if word is predicted
            word_in_top5 = any(word.lower() in t.lower() or t.lower() in word.lower()
                             for t in patched_top5)
            status = "✓" if word_in_top5 else "✗"

            print(f"  Layer {layer_idx:2d} patched: {patched_top5[:3]}... {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,28,31",
                       help="Comma-separated layers to test")
    args = parser.parse_args()

    device = torch.device("cuda")

    print("Loading model...")
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)

    ckpt_file = f"{args.ckpt_path}/model.pt"
    if os.path.getsize(ckpt_file) / (1024**3) < 1.0:
        model.reset_with_pretrained_weights()

    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    model = model.half().cuda().eval()
    torch.cuda.empty_cache()

    model_config = ModelConfig.load(
        resource_path(args.ckpt_path, "config.yaml"),
        key="model", validate_paths=False
    )
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=True, shuffle_messages=False,
        is_training=False, require_image_features=True
    )
    tokenizer = preprocessor.tokenizer

    num_layers = len(get_transformer_blocks(model))
    print(f"Model has {num_layers} layers")

    # Parse layers
    layers_to_test = [int(l.strip()) for l in args.layers.split(",")]
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    # Test words - variety of common words
    test_words = [
        "cat", "dog", "house", "water", "green",
        "happy", "running", "beautiful", "computer", "mountain",
        "coffee", "music", "phone", "book", "smile"
    ]

    # Run main test
    results = test_text_token_recovery(model, tokenizer, device, test_words, layers_to_test)

    # Run simpler sanity check
    test_same_model_patching(model, tokenizer, device)

    # Final verdict
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if any layer achieves >50% accuracy
    best_layer = max(layers_to_test, key=lambda l: results[l]["correct"] / max(results[l]["total"], 1))
    best_acc = results[best_layer]["correct"] / results[best_layer]["total"] * 100

    if best_acc > 50:
        print(f"✓ Patchscopes works on TEXT! Best: Layer {best_layer} with {best_acc:.1f}% accuracy")
        print("  This confirms our implementation is correct.")
        print("  Poor results on vision tokens are due to the nature of visual representations,")
        print("  not bugs in our patching mechanism.")
    elif best_acc > 20:
        print(f"~ Partial success. Best: Layer {best_layer} with {best_acc:.1f}% accuracy")
        print("  The mechanism works but may not be optimal for this model.")
    else:
        print(f"✗ Low accuracy even on text. Best: Layer {best_layer} with {best_acc:.1f}%")
        print("  This might indicate an issue, or the model/prompt format isn't suitable.")


if __name__ == "__main__":
    main()
