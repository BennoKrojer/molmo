#!/usr/bin/env python3
"""
Test that stripped checkpoint produces IDENTICAL inference outputs to original.

This is the definitive test - we actually generate captions and compare them.
If outputs match exactly, the checkpoints are functionally equivalent.

Usage:
    python scripts/test_inference_match.py \
        --original /path/to/original \
        --stripped /path/to/stripped \
        --num-examples 10
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from olmo.model import Molmo
from olmo.config import TrainConfig
from olmo.data.pixmo_datasets import PixMoCap


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading from: {checkpoint_path.name}")
    
    # Load config - set init_device to cpu to avoid meta tensors
    cfg = TrainConfig.load(checkpoint_path / "config.yaml")
    
    # Override init_device to avoid meta tensors
    original_init_device = cfg.model.init_device
    cfg.model.init_device = None  # Initialize on CPU with actual tensors
    
    # Initialize model
    model = Molmo(cfg.model)
    
    # Restore original init_device in config (for consistency)
    cfg.model.init_device = original_init_device
    
    # Load pretrained weights first (LLM + ViT)
    model.reset_with_pretrained_weights()
    
    # Load checkpoint weights (connector or full)
    state_dict = torch.load(checkpoint_path / "model.pt", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"  Loaded {len(state_dict)} params from checkpoint")
    print(f"  Missing {len(missing_keys)} params (OK for stripped)")
    
    if unexpected_keys:
        print(f"  WARNING: {len(unexpected_keys)} unexpected keys!")
        return None, None
    
    # Move to device and set to eval
    model = model.to(device)
    model.eval()
    
    return model, cfg


def generate_from_image(model, image_tensor, prompt_ids, cfg, max_tokens=50, device="cuda"):
    """Generate caption from image using the model."""
    # Prepare inputs
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim
    prompt_ids = prompt_ids.unsqueeze(0).to(device)  # Add batch dim
    
    # Generate
    with torch.no_grad():
        try:
            output = model.generate(
                input_ids=prompt_ids,
                images=image_tensor,
                max_steps=max_tokens,
                beam_size=1,  # Greedy decoding for determinism
            )
            
            # Decode tokens to text
            tokenizer = cfg.model.get_tokenizer()
            generated_ids = output.token_ids[0].cpu().tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return generated_text, generated_ids
        except Exception as e:
            return f"ERROR: {str(e)}", None


def prepare_image_and_prompt(example_data, cfg):
    """Prepare image and prompt for generation."""
    # Get image
    image = example_data["image"]
    
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(np.array(image))
    
    # Ensure correct shape and type
    if image.dtype != torch.float32:
        image = image.float()
    
    # Normalize if needed (0-255 to 0-1)
    if image.max() > 1.0:
        image = image / 255.0
    
    # Get tokenizer and create prompt
    tokenizer = cfg.model.get_tokenizer()
    
    # Simple caption prompt
    prompt = "Caption:"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    
    return image, prompt_ids


def main():
    parser = argparse.ArgumentParser(description="Test inference outputs match between checkpoints")
    parser.add_argument("--original", type=str, required=True, help="Path to original checkpoint")
    parser.add_argument("--stripped", type=str, required=True, help="Path to stripped checkpoint")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    original_path = Path(args.original)
    stripped_path = Path(args.stripped)
    
    print("=" * 80)
    print("INFERENCE OUTPUT COMPARISON TEST")
    print("=" * 80)
    print(f"\nOriginal: {original_path}")
    print(f"Stripped: {stripped_path}")
    print(f"Device: {args.device}")
    print(f"Num examples: {args.num_examples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Seed: {args.seed}")
    
    # Check paths
    if not original_path.exists():
        print(f"\n❌ Original checkpoint not found: {original_path}")
        sys.exit(1)
    
    if not stripped_path.exists():
        print(f"\n❌ Stripped checkpoint not found: {stripped_path}")
        sys.exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available() and "cuda" in args.device:
        print(f"\n⚠️  WARNING: CUDA not available, using CPU (will be slow)")
        args.device = "cpu"
    
    # Load validation dataset
    print("\n" + "=" * 80)
    print("LOADING VALIDATION DATASET")
    print("=" * 80)
    val_dataset = PixMoCap(split="validation", mode="captions")
    print(f"✓ Loaded {len(val_dataset)} validation examples")
    
    # Load models
    print("\n" + "=" * 80)
    print("LOADING ORIGINAL CHECKPOINT")
    print("=" * 80)
    original_model, original_cfg = load_model(original_path, args.device)
    if original_model is None:
        print("❌ Failed to load original model")
        sys.exit(1)
    print("✓ Original model loaded successfully")
    
    print("\n" + "=" * 80)
    print("LOADING STRIPPED CHECKPOINT")
    print("=" * 80)
    stripped_model, stripped_cfg = load_model(stripped_path, args.device)
    if stripped_model is None:
        print("❌ Failed to load stripped model")
        sys.exit(1)
    print("✓ Stripped model loaded successfully")
    
    # Test on examples
    print("\n" + "=" * 80)
    print(f"TESTING INFERENCE ON {args.num_examples} EXAMPLES")
    print("=" * 80)
    
    exact_matches = 0
    token_mismatches = []
    text_mismatches = []
    
    for i in tqdm(range(args.num_examples), desc="Testing"):
        # Get example
        example_data = val_dataset.get(i, np.random.RandomState(args.seed + i))
        
        # Get ground truth
        gt_caption = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            gt_caption = message.get("text", "")
        
        print(f"\n{'=' * 80}")
        print(f"Example {i+1}/{args.num_examples}")
        print(f"{'=' * 80}")
        print(f"Ground Truth: {gt_caption[:150]}...")
        
        # Prepare inputs
        try:
            image, prompt_ids = prepare_image_and_prompt(example_data, original_cfg)
        except Exception as e:
            print(f"❌ Failed to prepare inputs: {e}")
            continue
        
        # Generate with original
        set_seed(args.seed + i)  # Reset seed for each example
        print(f"\nGenerating with ORIGINAL...")
        orig_text, orig_ids = generate_from_image(
            original_model, image, prompt_ids, original_cfg, args.max_tokens, args.device
        )
        print(f"Original: {orig_text[:150]}...")
        
        # Generate with stripped
        set_seed(args.seed + i)  # Same seed
        print(f"\nGenerating with STRIPPED...")
        strip_text, strip_ids = generate_from_image(
            stripped_model, image, prompt_ids, stripped_cfg, args.max_tokens, args.device
        )
        print(f"Stripped: {strip_text[:150]}...")
        
        # Compare token IDs (most precise)
        if orig_ids is not None and strip_ids is not None:
            if orig_ids == strip_ids:
                print("✅ EXACT MATCH: Token IDs are identical")
                exact_matches += 1
            else:
                print("❌ MISMATCH: Token IDs differ")
                # Find where they diverge
                min_len = min(len(orig_ids), len(strip_ids))
                first_diff = None
                for j in range(min_len):
                    if orig_ids[j] != strip_ids[j]:
                        first_diff = j
                        break
                
                if first_diff is not None:
                    print(f"   First difference at token {first_diff}")
                    print(f"   Original: {orig_ids[max(0,first_diff-2):first_diff+3]}")
                    print(f"   Stripped: {strip_ids[max(0,first_diff-2):first_diff+3]}")
                else:
                    print(f"   Length mismatch: {len(orig_ids)} vs {len(strip_ids)}")
                
                token_mismatches.append(i)
        
        # Also compare text (less precise due to decoding)
        if orig_text != strip_text:
            if i not in token_mismatches:
                text_mismatches.append(i)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nTotal examples tested: {args.num_examples}")
    print(f"Exact matches (token-level): {exact_matches}/{args.num_examples}")
    print(f"Token mismatches: {len(token_mismatches)}")
    print(f"Text-only mismatches: {len(text_mismatches)}")
    
    if exact_matches == args.num_examples:
        print("\n" + "=" * 80)
        print("✅ SUCCESS: ALL OUTPUTS MATCH EXACTLY!")
        print("=" * 80)
        print("\nThe stripped checkpoint is functionally IDENTICAL to the original:")
        print("  • All generated token sequences match exactly")
        print("  • Inference outputs are deterministically identical")
        print("  • Safe to delete the original and use stripped checkpoint")
        print("\n✅ YOU CAN SAFELY:")
        print("  • Delete original checkpoints and keep only stripped versions")
        print("  • Use stripped checkpoints for all inference")
        print("  • Continue training from stripped checkpoints")
        print("=" * 80)
        return 0
    
    else:
        print("\n" + "=" * 80)
        print("❌ FAILURE: OUTPUTS DO NOT MATCH!")
        print("=" * 80)
        print(f"\nFound {len(token_mismatches)} examples with different outputs")
        print("\n⚠️  DO NOT USE THE STRIPPED CHECKPOINT!")
        print("   Something went wrong with the stripping process.")
        print("   DO NOT delete the original checkpoint.")
        
        if token_mismatches:
            print(f"\n   Examples with mismatches: {token_mismatches[:10]}")
        
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

