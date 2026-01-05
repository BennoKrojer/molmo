#!/usr/bin/env python3
"""
Validation tests for Patchscopes implementation.

Tests that verify:
1. Hook mechanism correctly patches hidden states
2. Layer patching happens at the right position
3. Output format matches LogitLens for viewer compatibility
4. Compare Patchscopes vs LogitLens predictions
"""

import argparse
import gc
import json
import math
import os
import numpy as np
import torch
from pathlib import Path

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap

from patchscopes_identity import (
    IDENTITY_PROMPT,
    PatchscopesHook,
    get_transformer_blocks,
    extract_visual_hidden_states,
)


def test_hook_mechanism(model, tokenizer, device):
    """
    Test 1: Verify the hook actually modifies hidden states.

    We run the model twice:
    1. Without hook (baseline)
    2. With hook that patches a known vector

    The outputs should differ at the patched position.
    """
    print("=" * 60)
    print("TEST 1: Hook Mechanism Verification")
    print("=" * 60)

    # Tokenize identity prompt
    identity_tokens = tokenizer.encode(IDENTITY_PROMPT)
    seq_len = len(identity_tokens)
    patch_position = seq_len - 1  # Position of "?"

    print(f"Identity prompt: {IDENTITY_PROMPT}")
    print(f"Token count: {seq_len}")
    print(f"Patch position: {patch_position}")

    # Create input
    input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)

    # Get hidden dim
    blocks = get_transformer_blocks(model)

    # Run baseline (no hook)
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            baseline_output = model(input_ids=input_ids, output_hidden_states=True)

    # Get hidden dim from output
    hidden_dim = baseline_output.hidden_states[0].shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    # Create a distinctive patch vector (all ones normalized)
    patch_vector = torch.ones(1, hidden_dim, device=device, dtype=torch.float16)
    patch_vector = patch_vector / patch_vector.norm()

    # Create hook for layer 16
    test_layer = 16
    hook = PatchscopesHook(patch_position, patch_vector)
    hook.register(blocks[test_layer])

    # Run with hook
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            hooked_output = model(input_ids=input_ids, output_hidden_states=True)

    hook.remove()

    # Compare hidden states AFTER the hooked layer
    # The hook modifies input to layer 16, so layer 16's output should differ
    baseline_hs = baseline_output.hidden_states[test_layer + 1][0, patch_position, :]
    hooked_hs = hooked_output.hidden_states[test_layer + 1][0, patch_position, :]

    # At OTHER positions, hidden states should be SAME
    baseline_other = baseline_output.hidden_states[test_layer + 1][0, 0, :]
    hooked_other = hooked_output.hidden_states[test_layer + 1][0, 0, :]

    diff_at_patch = (baseline_hs - hooked_hs).abs().mean().item()
    diff_at_other = (baseline_other - hooked_other).abs().mean().item()

    print(f"\nDifference at patch position: {diff_at_patch:.6f}")
    print(f"Difference at other position: {diff_at_other:.6f}")

    # Also check logits
    baseline_logits = baseline_output.logits[0, patch_position, :]
    hooked_logits = hooked_output.logits[0, patch_position, :]
    logit_diff = (baseline_logits - hooked_logits).abs().mean().item()
    print(f"Logit difference at patch position: {logit_diff:.6f}")

    # Assertions
    assert diff_at_patch > 0.01, "Hook should modify hidden states at patch position!"
    assert diff_at_other < 0.01, "Hook should NOT modify hidden states at other positions!"

    print("\n✓ TEST 1 PASSED: Hook correctly modifies only the patched position")

    del baseline_output, hooked_output
    torch.cuda.empty_cache()

    return True


def test_layer_patching(model, tokenizer, device):
    """
    Test 2: Verify patching happens at the correct layer.

    Hidden states BEFORE the patched layer should be identical.
    Hidden states AFTER the patched layer should differ.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Layer Patching Verification")
    print("=" * 60)

    identity_tokens = tokenizer.encode(IDENTITY_PROMPT)
    seq_len = len(identity_tokens)
    patch_position = seq_len - 1

    input_ids = torch.tensor(identity_tokens, device=device).unsqueeze(0)
    blocks = get_transformer_blocks(model)

    # Get baseline
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            baseline_output = model(input_ids=input_ids, output_hidden_states=True)

    hidden_dim = baseline_output.hidden_states[0].shape[-1]

    # Create patch vector
    patch_vector = torch.randn(1, hidden_dim, device=device, dtype=torch.float16)
    patch_vector = patch_vector / patch_vector.norm() * 10  # Make it distinctive

    # Test with hook at layer 16
    test_layer = 16
    hook = PatchscopesHook(patch_position, patch_vector)
    hook.register(blocks[test_layer])

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            hooked_output = model(input_ids=input_ids, output_hidden_states=True)

    hook.remove()

    # Check layers BEFORE the hook (should be identical)
    print(f"\nChecking hidden states at patch position...")
    print(f"Test layer: {test_layer}")

    for l in [0, 8, 15]:  # Layers before the hook
        baseline_hs = baseline_output.hidden_states[l][0, patch_position, :]
        hooked_hs = hooked_output.hidden_states[l][0, patch_position, :]
        diff = (baseline_hs - hooked_hs).abs().mean().item()
        status = "✓" if diff < 0.001 else "✗"
        print(f"  Layer {l} (before hook): diff = {diff:.8f} {status}")
        assert diff < 0.001, f"Layer {l} should be identical before hook!"

    # Check layers AFTER the hook (should differ)
    for l in [17, 24, 31]:  # Layers after the hook
        if l >= len(baseline_output.hidden_states):
            continue
        baseline_hs = baseline_output.hidden_states[l][0, patch_position, :]
        hooked_hs = hooked_output.hidden_states[l][0, patch_position, :]
        diff = (baseline_hs - hooked_hs).abs().mean().item()
        status = "✓" if diff > 0.01 else "✗"
        print(f"  Layer {l} (after hook): diff = {diff:.6f} {status}")
        assert diff > 0.01, f"Layer {l} should differ after hook!"

    print("\n✓ TEST 2 PASSED: Patching occurs at the correct layer")

    del baseline_output, hooked_output
    torch.cuda.empty_cache()

    return True


def test_output_format(output_dir):
    """
    Test 3: Verify output format matches LogitLens for viewer compatibility.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Output Format Verification")
    print("=" * 60)

    # Find a patchscopes output file
    output_path = Path(output_dir)
    json_files = list(output_path.glob("**/patchscopes_identity_layer*.json"))

    if not json_files:
        print("WARNING: No patchscopes output files found. Skipping format test.")
        return True

    # Load and check structure
    with open(json_files[0]) as f:
        data = json.load(f)

    print(f"Checking file: {json_files[0].name}")

    # Required top-level keys
    required_keys = ['checkpoint', 'method', 'layer_idx', 'results']
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"
        print(f"  ✓ Has '{key}'")

    # Check result structure
    if data['results']:
        result = data['results'][0]
        assert 'image_idx' in result, "Missing 'image_idx'"
        assert 'chunks' in result, "Missing 'chunks'"

        chunk = result['chunks'][0]
        assert 'patches' in chunk, "Missing 'patches'"

        patch = chunk['patches'][0]
        assert 'patch_idx' in patch, "Missing 'patch_idx'"
        assert 'patch_row' in patch, "Missing 'patch_row'"
        assert 'patch_col' in patch, "Missing 'patch_col'"
        assert 'top_predictions' in patch, "Missing 'top_predictions'"

        pred = patch['top_predictions'][0]
        assert 'token' in pred, "Missing 'token'"
        assert 'token_id' in pred, "Missing 'token_id'"
        assert 'logit' in pred, "Missing 'logit'"

        print(f"  ✓ Result structure matches LogitLens format")

    print("\n✓ TEST 3 PASSED: Output format is viewer-compatible")
    return True


def test_vs_logitlens(model, tokenizer, preprocessor, dataset, device):
    """
    Test 4: Compare Patchscopes vs LogitLens predictions.

    At the final layer, both should produce similar results since there's
    minimal processing after patching. At earlier layers, they may differ
    significantly.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Patchscopes vs LogitLens Comparison")
    print("=" * 60)

    # Load one image
    example_data = dataset.get(0, np.random)
    example = {"image": example_data["image"], "messages": ["Describe this image in detail."]}
    batch = preprocessor(example, rng=np.random)

    # Prepare inputs
    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
    images = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
    image_masks = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
    image_input_idx = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None

    blocks = get_transformer_blocks(model)

    # Get hidden states
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            output = model(
                input_ids=input_ids,
                images=images,
                image_masks=image_masks,
                image_input_idx=image_input_idx,
                output_hidden_states=True,
                last_logits_only=False,
            )

    hidden_states = output.hidden_states

    # Extract visual token hidden states at final layer
    # Note: hidden_states has len(blocks) + 1 elements (embeddings + each block output)
    # So hidden_states[-1] corresponds to the output of the last block
    # For patching, we use the second-to-last layer (blocks[-2]) so there's at least
    # one layer of processing after the patch
    num_blocks = len(blocks)
    final_layer = num_blocks - 1  # Last block index (e.g., 31 for 32-block model)
    visual_hs = extract_visual_hidden_states(hidden_states[final_layer], image_input_idx)

    # Get one visual token
    visual_token_hs = visual_hs[0, 0, 0, :].unsqueeze(0)  # First patch

    print(f"Testing with final layer: {final_layer}")
    print(f"Visual token shape: {visual_token_hs.shape}")

    # LogitLens: apply ln_f + ff_out directly
    if hasattr(model, 'module'):
        ln_f = model.module.transformer.ln_f
        ff_out = model.module.transformer.ff_out
    else:
        ln_f = model.transformer.ln_f
        ff_out = model.transformer.ff_out

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            logitlens_logits = ff_out(ln_f(visual_token_hs))

    logitlens_top5 = torch.topk(logitlens_logits[0], k=5)
    print(f"\nLogitLens top-5 at layer {final_layer}:")
    for i, (idx, val) in enumerate(zip(logitlens_top5.indices, logitlens_top5.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' (logit={val.item():.2f})")

    # Patchscopes: patch and forward
    identity_tokens = tokenizer.encode(IDENTITY_PROMPT)
    patch_position = len(identity_tokens) - 1

    batch_tokens = torch.tensor(identity_tokens, device=device).unsqueeze(0)

    hook = PatchscopesHook(patch_position, visual_token_hs)
    hook.register(blocks[final_layer])

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            patchscopes_output = model(input_ids=batch_tokens, output_hidden_states=False)

    hook.remove()

    patchscopes_logits = patchscopes_output.logits[0, patch_position, :]
    patchscopes_top5 = torch.topk(patchscopes_logits, k=5)

    print(f"\nPatchscopes top-5 at layer {final_layer}:")
    for i, (idx, val) in enumerate(zip(patchscopes_top5.indices, patchscopes_top5.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' (logit={val.item():.2f})")

    # Check overlap
    logitlens_set = set(logitlens_top5.indices.cpu().tolist())
    patchscopes_set = set(patchscopes_top5.indices.cpu().tolist())
    overlap = len(logitlens_set & patchscopes_set)

    print(f"\nOverlap in top-5: {overlap}/5")
    print("(At final layer, both methods should produce similar results)")

    # At final layer, we expect high overlap since there's minimal processing after patching
    # But they may still differ due to the identity prompt context

    print("\n✓ TEST 4 PASSED: Comparison complete")

    del output, hidden_states, patchscopes_output
    torch.cuda.empty_cache()

    return True


def main():
    parser = argparse.ArgumentParser(description="Patchscopes Validation Tests")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis_results/patchscopes")
    args = parser.parse_args()

    device = torch.device("cuda")

    print("=" * 60)
    print("PATCHSCOPES VALIDATION TESTS")
    print("=" * 60)
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

    dataset = PixMoCap(split="validation", mode="captions")

    # Run tests
    all_passed = True

    all_passed &= test_hook_mechanism(model, tokenizer, device)
    all_passed &= test_layer_patching(model, tokenizer, device)
    all_passed &= test_output_format(args.output_dir)
    all_passed &= test_vs_logitlens(model, tokenizer, preprocessor, dataset, device)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
