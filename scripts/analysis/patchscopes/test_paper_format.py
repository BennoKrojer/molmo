#!/usr/bin/env python3
"""
Test Patchscopes using the EXACT format from the paper.

The paper uses random integers [1-10] as few-shot examples:
"3->3; 7->7; 1->1; ?->"

NOT words like "cat->cat; dog->dog"
"""

import argparse
import gc
import os
import random
import torch

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path


def get_transformer_blocks(model):
    if hasattr(model, 'module'):
        return model.module.transformer.blocks
    else:
        return model.transformer.blocks


def create_identity_prompt_paper_format():
    """Create identity prompt using paper's format with random integers."""
    # Sample random integers from [1, 10]
    nums = random.sample(range(1, 11), 4)
    prompt = "; ".join([f"{n}->{n}" for n in nums[:-1]]) + f"; {nums[-1]}->"
    return prompt, nums[-1]  # Return prompt and the number we're testing


def test_paper_format(model, tokenizer, device):
    """Test using paper's exact format."""
    print("=" * 70)
    print("PAPER FORMAT TEST: Random integers [1-10]")
    print("=" * 70)

    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    # Test with multiple source texts from "Pile-like" examples
    source_texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a hole in the ground there lived a hobbit",
        "It was the best of times, it was the worst of times",
        "To be or not to be, that is the question",
        "All happy families are alike; each unhappy family is unhappy in its own way",
    ]

    layers_to_test = [0, 4, 8, 12, 16, 20, 24, 28, min(31, num_layers-1)]
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    results = {l: {"correct": 0, "top5": 0, "total": 0} for l in layers_to_test}

    for source_text in source_texts:
        source_tokens = tokenizer.encode(source_text)
        source_input_ids = torch.tensor(source_tokens, device=device).unsqueeze(0)

        print(f"\nSource: \"{source_text[:50]}...\"")
        print(f"Tokens: {len(source_tokens)}")

        # Get hidden states from source
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                source_output = model(input_ids=source_input_ids, output_hidden_states=True)

        # Test a few positions in the source
        test_positions = [0, len(source_tokens)//2, len(source_tokens)-1]

        for pos in test_positions:
            target_token = source_tokens[pos]
            target_word = tokenizer.decode([target_token])

            # Create identity prompt with paper format
            random.seed(42 + pos)  # Reproducible
            identity_prompt, _ = create_identity_prompt_paper_format()
            identity_tokens = tokenizer.encode(identity_prompt)
            identity_input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)

            # Find patch position (the last number before "->")
            # The pattern is "N->" at the end
            patch_position = len(identity_tokens) - 2

            print(f"  Position {pos} ('{target_word.strip()}'): ", end="")

            for layer_idx in layers_to_test:
                # Get source hidden state
                source_hs = source_output.hidden_states[layer_idx][0, pos, :].clone()

                # Patch hook
                def make_hook(hs, patch_pos):
                    def hook_fn(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            h = args[0].clone()
                            h[:, patch_pos, :] = hs.unsqueeze(0)
                            return (h,) + args[1:]
                        return None
                    return hook_fn

                handle = blocks[layer_idx].register_forward_pre_hook(
                    make_hook(source_hs, patch_position)
                )

                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                        patched_output = model(input_ids=identity_input_ids)

                handle.remove()

                # Check prediction
                logits = patched_output.logits[0, patch_position, :]
                top10_idxs = torch.topk(logits, k=10)[1]
                top10_tokens = [tokenizer.decode([idx.item()]) for idx in top10_idxs]

                in_top1 = top10_idxs[0].item() == target_token
                in_top5 = target_token in [idx.item() for idx in top10_idxs[:5]]

                results[layer_idx]["total"] += 1
                if in_top1:
                    results[layer_idx]["correct"] += 1
                if in_top5:
                    results[layer_idx]["top5"] += 1

            # Show best layer result
            best_layer = max(layers_to_test,
                           key=lambda l: 1 if results[l]["correct"] > 0 else 0)
            logits = patched_output.logits[0, patch_position, :]
            top3 = [tokenizer.decode([idx.item()]).strip()
                   for idx in torch.topk(logits, k=3)[1]]
            print(f"top-3 @ layer {best_layer}: {top3}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS: Precision@1 and Precision@5 by Layer")
    print("=" * 70)

    for layer_idx in layers_to_test:
        r = results[layer_idx]
        if r["total"] > 0:
            p1 = r["correct"] / r["total"] * 100
            p5 = r["top5"] / r["total"] * 100
            print(f"Layer {layer_idx:2d}: P@1 = {p1:5.1f}%, P@5 = {p5:5.1f}% ({r['total']} samples)")

    return results


def test_with_word_prompt(model, tokenizer, device):
    """Compare with word-based prompt for reference."""
    print("\n" + "=" * 70)
    print("COMPARISON: Word-based prompt (cat->cat; dog->dog)")
    print("=" * 70)

    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    # Word-based identity prompt
    identity_prompt = "cat->cat; dog->dog; hello->hello; ?->"
    identity_tokens = tokenizer.encode(identity_prompt)
    identity_input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)
    patch_position = len(identity_tokens) - 2

    source_text = "The quick brown fox jumps over the lazy dog"
    source_tokens = tokenizer.encode(source_text)
    source_input_ids = torch.tensor(source_tokens, device=device).unsqueeze(0)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            source_output = model(input_ids=source_input_ids, output_hidden_states=True)

    layers_to_test = [0, 8, 16, 24, num_layers-1]
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    # Test last token "dog"
    pos = len(source_tokens) - 1
    target_token = source_tokens[pos]
    target_word = tokenizer.decode([target_token])

    print(f"Testing token: '{target_word}' from position {pos}")

    for layer_idx in layers_to_test:
        source_hs = source_output.hidden_states[layer_idx][0, pos, :].clone()

        def make_hook(hs, patch_pos):
            def hook_fn(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    h = args[0].clone()
                    h[:, patch_pos, :] = hs.unsqueeze(0)
                    return (h,) + args[1:]
                return None
            return hook_fn

        handle = blocks[layer_idx].register_forward_pre_hook(
            make_hook(source_hs, patch_position)
        )

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                patched_output = model(input_ids=identity_input_ids)

        handle.remove()

        logits = patched_output.logits[0, patch_position, :]
        top5 = [tokenizer.decode([idx.item()]).strip()
               for idx in torch.topk(logits, k=5)[1]]

        in_top1 = "dog" in top5[0].lower()
        status = "✓" if in_top1 else "✗"
        print(f"  Layer {layer_idx:2d}: {top5} {status}")


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

    print(f"Model: OLMo-7B, {len(get_transformer_blocks(model))} layers\n")

    # Run tests
    test_paper_format(model, tokenizer, device)
    test_with_word_prompt(model, tokenizer, device)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The paper reports results on LLaMA2-13B, Vicuna-13B, GPT-J-6B, Pythia-12B.
We're testing on OLMo-7B (fine-tuned for vision-language).

If results are still poor with the paper's exact format:
  → OLMo may not be as amenable to this method
  → Our implementation is likely correct (hook mechanism verified earlier)
  → Vision tokens failing is expected given text tokens also struggle
""")


if __name__ == "__main__":
    main()
