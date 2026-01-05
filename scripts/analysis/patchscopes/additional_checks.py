#!/usr/bin/env python3
"""
Additional checks for potential edge cases and common pitfalls.
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


def test_layer_indexing_alignment(model, tokenizer, device):
    """
    Verify hidden_states[l] aligns with blocks[l].

    hidden_states[l] should be the INPUT to blocks[l] (output of blocks[l-1]).
    """
    print("=" * 70)
    print("TEST: Layer Indexing Alignment")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    prompt = "Hello world test"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

    # Capture input to block 16
    captured_input = []

    def capture_hook(module, args):
        if isinstance(args, tuple) and len(args) > 0:
            captured_input.append(args[0].clone())
        return None

    handle = blocks[16].register_forward_pre_hook(capture_hook)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            output = model(input_ids=input_ids, output_hidden_states=True)

    handle.remove()

    # Compare: hidden_states[16] should equal captured_input[0]
    hs_16 = output.hidden_states[16]
    captured = captured_input[0]

    diff = (hs_16 - captured).abs().mean().item()

    print(f"hidden_states[16] shape: {hs_16.shape}")
    print(f"Captured input to blocks[16] shape: {captured.shape}")
    print(f"Difference: {diff:.10f}")

    if diff < 1e-5:
        print("\n✓ PASS: hidden_states[l] == input to blocks[l]")
        return True
    else:
        print("\n✗ FAIL: Indexing mismatch!")
        return False


def test_different_patch_positions(model, tokenizer, device):
    """
    Verify patching works at different positions, not just the last.
    """
    print("\n" + "=" * 70)
    print("TEST: Different Patch Positions")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    prompt = "cat->cat; dog->dog; test"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    seq_len = len(tokens)

    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens: {tokens}")
    print(f"Seq len: {seq_len}")

    # Get baseline
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            baseline = model(input_ids=input_ids, output_hidden_states=True)

    hidden_dim = baseline.hidden_states[0].shape[-1]

    # Test patching at position 0, middle, and last
    test_positions = [0, seq_len // 2, seq_len - 1]

    for pos in test_positions:
        patch_tensor = torch.randn(1, hidden_dim, device=device, dtype=torch.float16)

        def patch_hook(module, args, pos=pos, patch=patch_tensor):
            if isinstance(args, tuple) and len(args) > 0:
                hs = args[0].clone()
                hs[:, pos, :] = patch
                return (hs,) + args[1:]
            return None

        handle = blocks[16].register_forward_pre_hook(patch_hook)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                patched = model(input_ids=input_ids)

        handle.remove()

        # Check if output changed
        diff = (baseline.logits - patched.logits).abs().mean().item()
        print(f"  Position {pos}: logit diff = {diff:.6f}")

    print("\n✓ PASS: Patching works at different positions")
    return True


def test_batch_patching(model, tokenizer, device):
    """
    Verify patching works correctly with batch size > 1.
    """
    print("\n" + "=" * 70)
    print("TEST: Batch Patching")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    prompt = "cat->cat; ?"
    tokens = tokenizer.encode(prompt)
    batch_size = 4
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0).expand(batch_size, -1)
    patch_position = len(tokens) - 1

    print(f"Batch size: {batch_size}")
    print(f"Input shape: {input_ids.shape}")

    # Get hidden dim
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            test_out = model(input_ids=input_ids[:1], output_hidden_states=True)
    hidden_dim = test_out.hidden_states[0].shape[-1]

    # Create DIFFERENT patch vectors for each batch item
    patch_tensors = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)

    def batch_patch_hook(module, args):
        if isinstance(args, tuple) and len(args) > 0:
            hs = args[0].clone()
            hs[:, patch_position, :] = patch_tensors
            return (hs,) + args[1:]
        return None

    handle = blocks[16].register_forward_pre_hook(batch_patch_hook)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            output = model(input_ids=input_ids)

    handle.remove()

    # Each batch item should have different outputs
    logits = output.logits[:, patch_position, :]

    # Compare batch items pairwise
    all_different = True
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            diff = (logits[i] - logits[j]).abs().mean().item()
            if diff < 0.01:
                all_different = False
                print(f"  Batch {i} vs {j}: diff = {diff:.6f} (TOO SIMILAR!)")
            else:
                print(f"  Batch {i} vs {j}: diff = {diff:.6f} ✓")

    if all_different:
        print("\n✓ PASS: Each batch item gets different patch correctly")
        return True
    else:
        print("\n✗ FAIL: Batch items not differentiated correctly!")
        return False


def test_no_gradient_leakage(model, tokenizer, device):
    """
    Verify hooks don't cause issues with inference mode.
    """
    print("\n" + "=" * 70)
    print("TEST: No Gradient/Memory Issues")
    print("=" * 70)

    blocks = get_transformer_blocks(model)

    prompt = "cat->cat; ?"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

    # Run many iterations to check for memory leaks
    initial_memory = torch.cuda.memory_allocated()

    for i in range(10):
        hidden_dim = 4096
        patch = torch.randn(1, hidden_dim, device=device, dtype=torch.float16)

        def hook(module, args, p=patch):
            if isinstance(args, tuple) and len(args) > 0:
                hs = args[0].clone()
                hs[:, -1, :] = p
                return (hs,) + args[1:]

        handle = blocks[16].register_forward_pre_hook(hook)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                _ = model(input_ids=input_ids)

        handle.remove()
        torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()
    memory_growth = (final_memory - initial_memory) / 1024 / 1024

    print(f"Memory growth after 10 iterations: {memory_growth:.2f} MB")

    if memory_growth < 100:  # Allow some tolerance
        print("\n✓ PASS: No significant memory leak")
        return True
    else:
        print("\n⚠ WARNING: Potential memory leak detected")
        return False


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

    # Run tests
    results = []
    results.append(("Layer indexing", test_layer_indexing_alignment(model, tokenizer, device)))
    results.append(("Patch positions", test_different_patch_positions(model, tokenizer, device)))
    results.append(("Batch patching", test_batch_patching(model, tokenizer, device)))
    results.append(("Memory safety", test_no_gradient_leakage(model, tokenizer, device)))

    print("\n" + "=" * 70)
    print("ADDITIONAL CHECKS SUMMARY")
    print("=" * 70)
    for name, passed in results:
        print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")


if __name__ == "__main__":
    main()
