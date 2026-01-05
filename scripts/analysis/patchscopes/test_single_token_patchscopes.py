#!/usr/bin/env python3
"""
Test Patchscopes with single-token hidden states.

The original paper likely uses minimal context for the source token.
This test patches hidden states from running just the token itself.
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


def test_single_token_patching(model, tokenizer, device):
    """
    Test patching with single-token hidden states.

    1. Run just the token "X" through the model
    2. Patch its hidden state into "cat->cat; dog->dog; ?->"
    3. Check if model predicts "X"
    """
    print("=" * 70)
    print("SINGLE-TOKEN PATCHSCOPES TEST")
    print("=" * 70)
    print("Testing if single-token hidden states can be recovered via identity prompt.\n")

    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    # Identity prompt
    identity_prompt = "cat->cat; dog->dog; hello->hello; ?->"
    identity_tokens = tokenizer.encode(identity_prompt)
    identity_input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)

    # Find ? position
    decoded = [tokenizer.decode([t]) for t in identity_tokens]
    print(f"Identity prompt tokens: {decoded}")
    patch_position = len(identity_tokens) - 2  # Position of "?" before "->"
    print(f"Patch position: {patch_position} ('{decoded[patch_position]}')\n")

    # Test words
    test_words = ["cat", "dog", "apple", "house", "water", "tree", "blue", "red", "car", "sun"]
    layers_to_test = [0, 4, 8, 12, 16, 20, 24, 28, num_layers-1]
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    results = {l: {"correct": 0, "top5": 0, "total": 0} for l in layers_to_test}

    for word in test_words:
        # Tokenize just the word
        word_tokens = tokenizer.encode(word)
        if len(word_tokens) > 1:
            print(f"Skipping '{word}' - multi-token ({len(word_tokens)} tokens)")
            continue

        word_input_ids = torch.tensor(word_tokens, device=device).unsqueeze(0)

        # Get hidden states from single token
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                word_output = model(input_ids=word_input_ids, output_hidden_states=True)

        print(f"Testing: '{word}'")

        for layer_idx in layers_to_test:
            # Get hidden state at this layer (position 0 since single token)
            word_hs = word_output.hidden_states[layer_idx][0, 0, :].clone()

            # Patch into identity prompt
            def make_hook(hs, pos):
                def hook_fn(module, args):
                    if isinstance(args, tuple) and len(args) > 0:
                        h = args[0].clone()
                        h[:, pos, :] = hs.unsqueeze(0)
                        return (h,) + args[1:]
                    return None
                return hook_fn

            handle = blocks[layer_idx].register_forward_pre_hook(
                make_hook(word_hs, patch_position)
            )

            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    patched_output = model(input_ids=identity_input_ids)

            handle.remove()

            # Get prediction at patch position
            logits = patched_output.logits[0, patch_position, :]
            top10_idxs = torch.topk(logits, k=10)[1]
            top10_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top10_idxs]

            # Check accuracy
            word_clean = word.strip().lower()
            in_top1 = top10_tokens[0].lower() == word_clean
            in_top5 = any(t.lower() == word_clean for t in top10_tokens[:5])

            results[layer_idx]["total"] += 1
            if in_top1:
                results[layer_idx]["correct"] += 1
            if in_top5:
                results[layer_idx]["top5"] += 1

            status = "✓" if in_top1 else ("~" if in_top5 else "✗")
            print(f"  Layer {layer_idx:2d}: {top10_tokens[:5]} {status}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Single-Token Recovery")
    print("=" * 70)
    for layer_idx in layers_to_test:
        r = results[layer_idx]
        if r["total"] > 0:
            acc1 = r["correct"] / r["total"] * 100
            acc5 = r["top5"] / r["total"] * 100
            print(f"Layer {layer_idx:2d}: Top-1 = {acc1:5.1f}%, Top-5 = {acc5:5.1f}%")

    return results


def test_embedding_only(model, tokenizer, device):
    """
    Test what happens if we just use the token embedding (layer 0 input).
    This is the simplest possible case.
    """
    print("\n" + "=" * 70)
    print("EMBEDDING-ONLY TEST (Layer 0 Hidden State)")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    # Get the embedding layer
    if hasattr(model, 'module'):
        embed = model.module.transformer.wte
    else:
        embed = model.transformer.wte

    identity_prompt = "cat->cat; dog->dog; hello->hello; ?->"
    identity_tokens = tokenizer.encode(identity_prompt)
    identity_input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)
    patch_position = len(identity_tokens) - 2

    test_words = ["cat", "dog", "apple", "house", "water"]

    for word in test_words:
        word_tokens = tokenizer.encode(word)
        if len(word_tokens) > 1:
            continue

        # Get embedding directly
        word_id = torch.tensor(word_tokens, device=device)
        word_embed = embed(word_id)[0]  # [hidden_dim]

        # Patch at layer 0
        def make_hook(emb, pos):
            def hook_fn(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    h = args[0].clone()
                    h[:, pos, :] = emb.unsqueeze(0)
                    return (h,) + args[1:]
                return None
            return hook_fn

        handle = blocks[0].register_forward_pre_hook(
            make_hook(word_embed, patch_position)
        )

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                output = model(input_ids=identity_input_ids)

        handle.remove()

        logits = output.logits[0, patch_position, :]
        top5 = [tokenizer.decode([idx.item()]).strip()
               for idx in torch.topk(logits, k=5)[1]]

        in_top1 = top5[0].lower() == word.lower()
        status = "✓" if in_top1 else "✗"
        print(f"'{word}' -> {top5} {status}")


def test_cross_context_comparison(model, tokenizer, device):
    """
    Compare: same token from different contexts.

    Run "cat" in:
    1. Just "cat"
    2. "The cat is cute"
    3. "I have a cat"

    Then patch each into identity prompt and see which works best.
    """
    print("\n" + "=" * 70)
    print("CROSS-CONTEXT COMPARISON")
    print("=" * 70)
    print("Testing if context affects recoverability.\n")

    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    identity_prompt = "cat->cat; dog->dog; hello->hello; ?->"
    identity_tokens = tokenizer.encode(identity_prompt)
    identity_input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)
    patch_position = len(identity_tokens) - 2

    target_word = "apple"
    contexts = [
        (f"{target_word}", -1, "Single token"),  # Just the word
        (f"The {target_word} is red", 1, "Subject position"),  # After "The"
        (f"I ate an {target_word}", -1, "Object position"),  # Last word
    ]

    layers_to_test = [0, 8, 16, 24, num_layers - 1]
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    for context, word_pos, context_name in contexts:
        tokens = tokenizer.encode(context)
        input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

        # Find actual position of target word
        if word_pos == -1:
            word_pos = len(tokens) - 1  # Last token

        print(f"\nContext: \"{context}\" ({context_name})")
        print(f"Target position: {word_pos}, token: '{tokenizer.decode([tokens[word_pos]])}'")

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                output = model(input_ids=input_ids, output_hidden_states=True)

        for layer_idx in layers_to_test:
            word_hs = output.hidden_states[layer_idx][0, word_pos, :].clone()

            def make_hook(hs, pos):
                def hook_fn(module, args):
                    if isinstance(args, tuple) and len(args) > 0:
                        h = args[0].clone()
                        h[:, pos, :] = hs.unsqueeze(0)
                        return (h,) + args[1:]
                    return None
                return hook_fn

            handle = blocks[layer_idx].register_forward_pre_hook(
                make_hook(word_hs, patch_position)
            )

            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    patched = model(input_ids=identity_input_ids)

            handle.remove()

            logits = patched.logits[0, patch_position, :]
            top5 = [tokenizer.decode([idx.item()]).strip()
                   for idx in torch.topk(logits, k=5)[1]]

            in_top1 = top5[0].lower() == target_word.lower()
            in_top5 = any(t.lower() == target_word.lower() for t in top5[:5])
            status = "✓" if in_top1 else ("~" if in_top5 else "✗")
            print(f"  Layer {layer_idx:2d}: {top5[:3]} {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
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

    print(f"Model has {len(get_transformer_blocks(model))} layers\n")

    # Run tests
    test_single_token_patching(model, tokenizer, device)
    test_embedding_only(model, tokenizer, device)
    test_cross_context_comparison(model, tokenizer, device)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If single-token patching works well (>50% top-1):
  → Our implementation is correct
  → Vision tokens fail because they encode different information than text

If single-token patching fails:
  → The identity prompt format may not work well for this model
  → This would explain both text and vision token failures
""")


if __name__ == "__main__":
    main()
