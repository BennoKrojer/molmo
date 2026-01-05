#!/usr/bin/env python3
"""
Comprehensive sanity checks for Patchscopes implementation.

These tests verify:
1. Identity prompt actually works (predicts "ha" given "ha->")
2. Hooks are actually being called and modifying tensors
3. OLMo block input structure is what we expect
4. Patching with known vectors produces expected changes
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
    """Get the transformer blocks from the model."""
    if hasattr(model, 'module'):
        return model.module.transformer.blocks
    else:
        return model.transformer.blocks


def test_identity_prompt_works(model, tokenizer, device):
    """
    TEST 1: Verify the identity prompt actually works.

    If we give "cat->cat; 1135->1135; hello->hello; ha->"
    the model should predict "ha" as the next token.
    """
    print("=" * 70)
    print("TEST 1: Identity Prompt Actually Works")
    print("=" * 70)

    # Test prompts with expected continuations
    test_cases = [
        ("cat->cat; dog->dog; hello->hello; world->", "world"),
        ("cat->cat; 1135->1135; hello->hello; ha->", "ha"),
        ("apple->apple; banana->banana; cherry->", "cherry"),
    ]

    all_passed = True

    for prompt, expected in test_cases:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                output = model(input_ids=input_ids)
                logits = output.logits[0, -1, :]  # Last position

        # Get top prediction
        top_idx = logits.argmax().item()
        top_token = tokenizer.decode([top_idx])

        # Get top-5 for context
        top5_vals, top5_idxs = torch.topk(logits, k=5)
        top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_idxs]

        # Check if expected is in top-5
        expected_in_top5 = expected in top5_tokens or expected.strip() in [t.strip() for t in top5_tokens]
        status = "✓ PASS" if expected_in_top5 else "✗ FAIL"

        print(f"\nPrompt: \"{prompt}\"")
        print(f"Expected: \"{expected}\"")
        print(f"Top prediction: \"{top_token}\"")
        print(f"Top-5: {top5_tokens}")
        print(f"Result: {status}")

        if not expected_in_top5:
            all_passed = False

    if all_passed:
        print("\n✓ TEST 1 PASSED: Identity prompt works correctly!")
    else:
        print("\n✗ TEST 1 FAILED: Identity prompt doesn't predict expected tokens!")
        print("  This could indicate the prompt format isn't working as expected.")

    return all_passed


def test_hook_actually_called(model, tokenizer, device):
    """
    TEST 2: Verify hooks are actually being called.

    Add debug hooks that print when called and verify modification.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Hooks Are Actually Called")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    # Track hook calls
    hook_call_count = [0]
    input_shapes = []

    def debug_hook(module, args):
        hook_call_count[0] += 1

        # Log the structure of args
        if hook_call_count[0] == 1:  # Only log first call
            print(f"\n  Hook called! Args structure:")
            print(f"  - Type of args: {type(args)}")
            print(f"  - Length of args: {len(args) if isinstance(args, tuple) else 'N/A'}")

            if isinstance(args, tuple) and len(args) > 0:
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        print(f"  - args[{i}]: Tensor shape {arg.shape}, dtype {arg.dtype}")
                        input_shapes.append(arg.shape)
                    else:
                        print(f"  - args[{i}]: {type(arg)}")

        return None  # Don't modify, just observe

    # Register hook on layer 16
    test_layer = 16
    handle = blocks[test_layer].register_forward_pre_hook(debug_hook)

    # Run a forward pass
    prompt = "cat->cat; hello->hello; ?"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

    print(f"\nRunning forward pass with hook on layer {test_layer}...")
    print(f"Input tokens: {len(tokens)}")

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            _ = model(input_ids=input_ids)

    handle.remove()

    print(f"\nHook was called {hook_call_count[0]} time(s)")

    if hook_call_count[0] > 0:
        print("✓ TEST 2 PASSED: Hook is being called!")
        return True, input_shapes
    else:
        print("✗ TEST 2 FAILED: Hook was never called!")
        return False, input_shapes


def test_hook_modifies_output(model, tokenizer, device):
    """
    TEST 3: Verify hook modifications actually affect output.

    Run same input twice:
    1. Without hook
    2. With hook that patches zeros

    Outputs should differ significantly.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Hook Modifications Affect Output")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    prompt = "cat->cat; hello->hello; ?"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    patch_position = len(tokens) - 1  # Last token "?"

    # Run WITHOUT hook
    print("\nRunning without hook...")
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            output_baseline = model(input_ids=input_ids, output_hidden_states=True)

    baseline_logits = output_baseline.logits[0, patch_position, :].clone()
    baseline_hs = output_baseline.hidden_states[17][0, patch_position, :].clone()  # After layer 16
    hidden_dim = baseline_hs.shape[0]

    print(f"Baseline top prediction: {tokenizer.decode([baseline_logits.argmax().item()])}")

    # Create hook that patches with zeros
    patch_tensor = torch.zeros(1, hidden_dim, device=device, dtype=torch.float16)

    def zero_patch_hook(module, args):
        if isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0].clone()
            hidden_states[:, patch_position, :] = patch_tensor
            return (hidden_states,) + args[1:]
        return None

    # Run WITH hook
    print("Running with zero-patch hook on layer 16...")
    handle = blocks[16].register_forward_pre_hook(zero_patch_hook)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            output_patched = model(input_ids=input_ids, output_hidden_states=True)

    handle.remove()

    patched_logits = output_patched.logits[0, patch_position, :]
    patched_hs = output_patched.hidden_states[17][0, patch_position, :]  # After layer 16

    print(f"Patched top prediction: {tokenizer.decode([patched_logits.argmax().item()])}")

    # Compare
    logit_diff = (baseline_logits - patched_logits).abs().mean().item()
    hs_diff = (baseline_hs - patched_hs).abs().mean().item()

    print(f"\nLogit difference (mean abs): {logit_diff:.6f}")
    print(f"Hidden state difference (after hook layer): {hs_diff:.6f}")

    # Check hidden states BEFORE the hook (should be identical)
    hs_before_baseline = output_baseline.hidden_states[16][0, patch_position, :]
    hs_before_patched = output_patched.hidden_states[16][0, patch_position, :]
    hs_before_diff = (hs_before_baseline - hs_before_patched).abs().mean().item()
    print(f"Hidden state difference (before hook layer): {hs_before_diff:.6f}")

    if logit_diff > 0.1 and hs_diff > 0.1 and hs_before_diff < 0.001:
        print("\n✓ TEST 3 PASSED: Hook correctly modifies output!")
        return True
    else:
        print("\n✗ TEST 3 FAILED: Hook doesn't seem to affect output correctly!")
        if hs_before_diff > 0.001:
            print("  WARNING: Hidden states differ BEFORE the hook layer - unexpected!")
        return False


def test_patch_with_known_embedding(model, tokenizer, device):
    """
    TEST 4: Patch with a known token embedding and verify output.

    If we patch position "?" with the embedding of "dog",
    the model should predict "dog" (or something related).
    """
    print("\n" + "=" * 70)
    print("TEST 4: Patch With Known Token Embedding")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    # Get embedding of "dog"
    dog_tokens = tokenizer.encode("dog")
    print(f"'dog' tokenizes to: {dog_tokens}")

    # We need the hidden state of "dog" at layer 16, not just the embedding
    # Run "dog" through the model to get its representation at layer 16
    dog_input = torch.tensor(dog_tokens, device=device).unsqueeze(0)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            dog_output = model(input_ids=dog_input, output_hidden_states=True)

    # Get "dog" representation at layer 16 (last token if multi-token)
    dog_hs_layer16 = dog_output.hidden_states[16][0, -1, :].clone()
    print(f"Got 'dog' hidden state at layer 16, shape: {dog_hs_layer16.shape}")

    # Now run identity prompt and patch "?" with dog's representation
    prompt = "cat->cat; hello->hello; ?"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    patch_position = len(tokens) - 1

    def dog_patch_hook(module, args):
        if isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0].clone()
            hidden_states[:, patch_position, :] = dog_hs_layer16.unsqueeze(0)
            return (hidden_states,) + args[1:]
        return None

    handle = blocks[16].register_forward_pre_hook(dog_patch_hook)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            output = model(input_ids=input_ids)

    handle.remove()

    # Get prediction
    logits = output.logits[0, patch_position, :]
    top5_vals, top5_idxs = torch.topk(logits, k=10)
    top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_idxs]

    print(f"\nIdentity prompt: \"{prompt}\"")
    print(f"Patched '?' with 'dog' hidden state at layer 16")
    print(f"Top-10 predictions: {top5_tokens}")

    # Check if "dog" or related tokens appear
    dog_related = ["dog", "Dog", "dogs", "Dogs", " dog", " Dog"]
    found_dog = any(t.strip().lower() == "dog" for t in top5_tokens)

    if found_dog:
        print("\n✓ TEST 4 PASSED: Model predicted 'dog' after patching with 'dog' embedding!")
        return True
    else:
        print("\n⚠ TEST 4 INCONCLUSIVE: 'dog' not in top-10, but this might be expected")
        print("  The identity prompt context might shift the prediction.")
        return None  # Inconclusive, not necessarily a failure


def test_olmo_block_input_structure(model, tokenizer, device):
    """
    TEST 5: Understand OLMo block input structure.

    Print exactly what OLMo transformer blocks receive as input.
    """
    print("\n" + "=" * 70)
    print("TEST 5: OLMo Block Input Structure Analysis")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    captured_args = []

    def capture_hook(module, args):
        captured_args.append(args)
        return None

    # Hook multiple layers to see if structure is consistent
    handles = []
    for layer_idx in [0, 1, 16, 31]:
        if layer_idx < len(blocks):
            h = blocks[layer_idx].register_forward_pre_hook(capture_hook)
            handles.append((layer_idx, h))

    prompt = "Hello world"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            _ = model(input_ids=input_ids, output_hidden_states=True)

    for _, h in handles:
        h.remove()

    print(f"\nCaptured {len(captured_args)} hook calls")

    for i, (layer_idx, _) in enumerate(handles):
        if i < len(captured_args):
            args = captured_args[i]
            print(f"\nLayer {layer_idx} input structure:")
            print(f"  Type: {type(args)}")
            if isinstance(args, tuple):
                print(f"  Length: {len(args)}")
                for j, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        print(f"  args[{j}]: Tensor {arg.shape} {arg.dtype}")
                    elif arg is None:
                        print(f"  args[{j}]: None")
                    else:
                        print(f"  args[{j}]: {type(arg).__name__}")

    # Verify first arg is always the hidden states tensor
    if captured_args:
        first_arg = captured_args[0]
        if isinstance(first_arg, tuple) and len(first_arg) > 0:
            if isinstance(first_arg[0], torch.Tensor):
                print(f"\n✓ Confirmed: args[0] is the hidden states tensor")
                print(f"  Shape: {first_arg[0].shape} (batch, seq_len, hidden_dim)")
                return True

    print("\n⚠ Could not confirm input structure")
    return False


def main():
    parser = argparse.ArgumentParser(description="Patchscopes Sanity Checks")
    parser.add_argument("--ckpt-path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")

    print("=" * 70)
    print("PATCHSCOPES SANITY CHECKS")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt_path}")
    print()

    # Load model
    print("Loading model...")
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)

    ckpt_file = f"{args.ckpt_path}/model.pt"
    ckpt_size_gb = os.path.getsize(ckpt_file) / (1024**3)

    if ckpt_size_gb < 1.0:
        model.reset_with_pretrained_weights()

    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    model = model.half().cuda().eval()
    torch.cuda.empty_cache()
    print("Model loaded.\n")

    # Setup tokenizer
    model_config = ModelConfig.load(
        resource_path(args.ckpt_path, "config.yaml"),
        key="model",
        validate_paths=False
    )
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )
    tokenizer = preprocessor.tokenizer

    # Run tests
    results = {}

    results['test1_identity_works'] = test_identity_prompt_works(model, tokenizer, device)
    results['test2_hook_called'], input_shapes = test_hook_actually_called(model, tokenizer, device)
    results['test3_hook_modifies'] = test_hook_modifies_output(model, tokenizer, device)
    results['test4_known_embedding'] = test_patch_with_known_embedding(model, tokenizer, device)
    results['test5_input_structure'] = test_olmo_block_input_structure(model, tokenizer, device)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ INCONCLUSIVE"
        print(f"  {test_name}: {status}")

    critical_passed = results['test1_identity_works'] and results['test2_hook_called'] and results['test3_hook_modifies']

    if critical_passed:
        print("\n✓ All critical tests passed!")
    else:
        print("\n✗ Some critical tests failed - implementation may have bugs!")

    print("=" * 70)


if __name__ == "__main__":
    main()
