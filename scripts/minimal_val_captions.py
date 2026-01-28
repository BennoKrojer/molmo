"""Minimal script to iterate over validation examples and generate captions or other task outputs.
Automatically detects the task from checkpoint config and uses appropriate prompts.
Saves outputs to organized JSON files under analysis_results/captions/.

Usage: torchrun --nproc_per_node=2 scripts/minimal_val_captions.py --ckpt-path <checkpoint_path>

Or with specific GPUs:
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 scripts/minimal_val_captions.py --ckpt-path <checkpoint_path>
"""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from olmo.model import Molmo
from olmo.config import TrainConfig, ModelConfig
from olmo.torch_util import get_local_rank, get_world_size
from olmo.data.pixmo_datasets import PixMoCap, PixMoPoints, PixMoPointsLeftRight, PixMoPointsSpatial, PixMoPointsTopBottom
from olmo.data import build_mm_preprocessor
from olmo.util import resource_path


def main():
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # Set CUDA device
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    
    # Parse arguments (use parse_known_args to ignore distributed launcher arguments like --local-rank)
    parser = argparse.ArgumentParser(description="Generate captions for validation images using the model")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--num-images", type=int, default=300, help="Number of validation images to process")
    parser.add_argument("--output-base-dir", type=str, default="analysis_results/captions",
                       help="Base directory for saving caption results")
    parser.add_argument("--use-fp16", action="store_true",
                       help="Use FP16 instead of BF16 (for testing precision issues)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug output (logits, probs, etc.)")
    args, unknown = parser.parse_known_args()
    
    checkpoint_path = args.ckpt_path
    num_val_images = args.num_images
    
    if local_rank == 0:
        print(f"Running on {world_size} processes")
        print(f"Processing {num_val_images} validation images")
        print(f"Loading model from {checkpoint_path}")
    
    # Load model on CPU first
    # This works with both full checkpoints and stripped (MLP-only) checkpoints
    cfg = TrainConfig.load(f"{checkpoint_path}/config.yaml")
    
    # Detect dataset type from config
    dataset_name = cfg.data.dataset
    if local_rank == 0:
        print(f"Detected dataset: {dataset_name}")
    # Override init_device to avoid meta tensors
    cfg.model.init_device = None
    
    # Initialize model
    model = Molmo(cfg.model)
    
    # Check checkpoint size to determine if it's a full or stripped checkpoint
    import os
    checkpoint_file = f"{checkpoint_path}/model.pt"
    checkpoint_size_gb = os.path.getsize(checkpoint_file) / (1024**3)
    
    is_full_checkpoint = checkpoint_size_gb > 1.0
    
    if not is_full_checkpoint:
        if local_rank == 0:
            print(f"Detected stripped checkpoint ({checkpoint_size_gb:.2f} GB)")
            print("Loading pretrained weights (LLM + ViT)...")
        model.reset_with_pretrained_weights()
    else:
        if local_rank == 0:
            print(f"Detected full checkpoint ({checkpoint_size_gb:.2f} GB)")
            print("Skipping pretrained weights loading (checkpoint contains all weights)")
    
    # Load checkpoint weights
    if local_rank == 0:
        print("Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    
    if local_rank == 0:
        print(f"Loaded {len(checkpoint_weights)} parameters from checkpoint")
        
        # DEBUG: Check for NaN in MLP connector weights
        print("\n[DEBUG] Checking MLP connector for NaN values...")
        mlp_has_nan = False
        mlp_stats = {}
        for name, param in checkpoint_weights.items():
            if 'image_projector' in name or 'vision_backbone' in name:
                param_np = param.float().numpy()
                has_nan = np.isnan(param_np).any()
                has_inf = np.isinf(param_np).any()
                mlp_stats[name] = {
                    'shape': param.shape,
                    'min': param_np.min(),
                    'max': param_np.max(),
                    'mean': param_np.mean(),
                    'std': param_np.std(),
                    'has_nan': has_nan,
                    'has_inf': has_inf
                }
                if has_nan:
                    print(f"  ⚠️  Found NaN in {name}")
                    mlp_has_nan = True
                if has_inf:
                    print(f"  ⚠️  Found Inf in {name}")
                    mlp_has_nan = True
        
        if not mlp_has_nan:
            print("  ✓ No NaN/Inf found in MLP connector weights")
            print("\n  MLP Weight Statistics:")
            for name, stats in mlp_stats.items():
                print(f"    {name}:")
                print(f"      Shape: {stats['shape']}, Range: [{stats['min']:.4f}, {stats['max']:.4f}], Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    model.eval()
    
    # Create preprocessor
    model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # DON'T override system_prompt_kind - keep it as it was during training!
    # model_config.system_prompt_kind = "none"  # This was causing the mismatch!
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )
    
    # Wrap model with FSDP for sharding
    if local_rank == 0:
        print("Wrapping model with FSDP for sharding...")
    
    # Get FSDP wrap policy from the model
    wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")
    
    # Wrap model in FSDP
    # IMPORTANT: Use BF16 to match training precision (amp_bf16 in config)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    
    if local_rank == 0:
        print(f"Model wrapped with FSDP on device: {device}")
    
    # Wait for all processes to finish loading model
    dist.barrier()
    
    # Distribute images across processes
    images_per_process = num_val_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    
    # Handle remainder for last process
    if local_rank == world_size - 1:
        end_idx = num_val_images
    
    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1}")
    
    # Load validation dataset based on detected dataset type
    # Match the dataset loading logic from olmo/data/__init__.py
    if "pixmo_cap" in dataset_name or "cap" in dataset_name:
        val_dataset = PixMoCap(split="validation", mode="captions")
        task_type = "caption"
        if local_rank == 0:
            print("Task type: Caption generation")
    elif "spatial" in dataset_name:
        val_dataset = PixMoPointsSpatial(split="validation", kind="basic")
        task_type = "spatial"
        if local_rank == 0:
            print("Task type: Spatial classification (Left/Right/Top/Bottom)")
    elif "left_right" in dataset_name or "leftright" in dataset_name:
        val_dataset = PixMoPointsLeftRight(split="validation", kind="basic")
        task_type = "left_right"
        if local_rank == 0:
            print("Task type: Left/Right classification")
    elif "top_bottom" in dataset_name or "topbottom" in dataset_name:
        val_dataset = PixMoPointsTopBottom(split="validation", kind="basic")
        task_type = "top_bottom"
        if local_rank == 0:
            print("Task type: Top/Bottom classification")
    elif "pixmo_points" in dataset_name or "points" in dataset_name:
        # Use same parameters as get_dataset_by_name in olmo/data/__init__.py
        if dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
            val_dataset = PixMoPoints(split="validation", kind="high_frequency", counting=False)
        elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
            val_dataset = PixMoPoints(split="validation", kind="high_frequency", counting=True)
        elif dataset_name in ["point_count", "pixmo_points_counting"]:
            val_dataset = PixMoPoints(split="validation", kind="basic", counting=True)
        else:  # Default for "pixmo_points" and "pointing"
            val_dataset = PixMoPoints(split="validation", kind="basic", counting=False)
        task_type = "pointing"
        if local_rank == 0:
            print("Task type: Pointing/Counting")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    # Store captions from this process
    local_captions = []
    
    # Process assigned images for this process
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}", disable=(local_rank != 0)):
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(42 + i)
        example_data = val_dataset.get(i, rng)
        
        # Create example with appropriate prompt based on task type
        if task_type == "caption":
            example = {
                "image": example_data["image"],
                "messages": ["Caption:"]
            }
            ground_truth = example_data.get("caption", "")
        elif task_type == "left_right" or task_type == "spatial" or task_type == "top_bottom":
            # For spatial tasks, use the actual task messages from the dataset
            message_list = example_data.get("message_list", [])
            if message_list:
                # Use the first message
                first_msg = message_list[0]
                label = first_msg.get("label", "object")
                position = first_msg.get("position", "unknown")
                
                # Create prompt - the preprocessor will add the system prompt automatically
                # based on the style in the message
                prompt = f"Where is {label}?"
                
                # Pass the message with style so preprocessor can add system prompt
                example = {
                    "image": example_data["image"],
                    "message_list": [first_msg]  # Pass the full message with style
                }
                
                # Generate ground truth
                from olmo.data.data_formatter import DataFormatter
                formatter = DataFormatter()
                ground_truth = formatter.format_spatial(first_msg)
            else:
                # Fallback
                example = {
                    "image": example_data["image"],
                    "message_list": [{
                        "label": "object",
                        "position": "left",
                        "style": "spatial" if task_type != "top_bottom" else "top_bottom"
                    }]
                }
                ground_truth = ""
        else:  # pointing task
            # For pointing tasks, use the actual task messages from the dataset
            # PixMoPoints returns message_list with label and points
            message_list = example_data.get("message_list", [])
            if message_list:
                # Use the first message (could be pointing or counting)
                first_msg = message_list[0]
                label = first_msg.get("label", "object")
                style = first_msg.get("style", "point_count")
                
                # Create appropriate prompt based on style
                if style == "point_count":
                    prompt = f"How many {label} are there?"
                elif style == "pointing":
                    prompt = f"Point to {label}"
                else:
                    prompt = f"How many {label} are there?"
                
                example = {
                    "image": example_data["image"],
                    "messages": [prompt]
                }
                
                # Generate ACTUAL ground truth as model was trained to produce
                # Load image for format_points to work
                from olmo.data.model_preprocessor import load_image
                from olmo.data.data_formatter import DataFormatter
                
                image_array = load_image(example_data["image"])
                msg_with_image = first_msg.copy()
                msg_with_image["image"] = image_array
                
                formatter = DataFormatter()
                ground_truth = formatter.format_points(msg_with_image)
            else:
                # Fallback
                example = {
                    "image": example_data["image"],
                    "messages": ["How many objects are there?"]
                }
                ground_truth = ""
        
        # Preprocess example
        batch = preprocessor(example, rng=rng)
        
        # DEBUG: Print what's being fed to the model (same format as training logs)
        if args.debug and local_rank == 0 and i < start_idx + 3:  # Only print first 3 examples from rank 0
            print(f"\n{'='*80}")
            print(f"[DEBUG] Image {i} - Inference Preprocessing")
            print(f"{'='*80}")
            
            # Show the full input sequence
            input_tokens = batch["input_tokens"]
            print(f"\n[FULL INPUT] Length: {len(input_tokens)} tokens")
            full_input_text = preprocessor.tokenizer.decode([t for t in input_tokens if t >= 0])
            print(f"Full input text:\n{repr(full_input_text)}")
            
            # Show last ~50 tokens before padding (the prompt area)
            # Find where actual content ends (before -1 padding)
            actual_tokens = [t for t in input_tokens if t >= 0]
            last_50_tokens = actual_tokens[-50:] if len(actual_tokens) > 50 else actual_tokens
            last_50_text = preprocessor.tokenizer.decode(last_50_tokens)
            
            print(f"\n[PROMPT] (last ~50 tokens before target):")
            print(f"  Text: {repr(last_50_text)}")
            print(f"  Token IDs: {last_50_tokens[-20:]}")  # Show last 20 token IDs
            
            print(f"\n{'='*80}\n")
        
        # Generate output - NO TRY-EXCEPT per .cursorrules (let errors fail loudly)
        with torch.inference_mode():
            # CRITICAL: Use BF16 to match training (config says amp_bf16)
            # Can override with --use-fp16 for testing
            precision_dtype = torch.float16 if args.use_fp16 else torch.bfloat16
            if local_rank == 0 and i == start_idx:
                print(f"[INFO] Using precision: {'FP16' if args.use_fp16 else 'BF16'}")
            with torch.autocast("cuda", enabled=True, dtype=precision_dtype):
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                images = torch.tensor(batch["images"]).unsqueeze(0).to(device)
                image_masks = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                image_input_idx = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                
                # DEBUG: Comprehensive diagnostics (first 3 images only)
                if args.debug and local_rank == 0 and i < start_idx + 3:
                    print(f"\n{'='*80}")
                    print(f"[DEEP DEBUG] Image {i} - Model Behavior Analysis")
                    print(f"{'='*80}")
                    
                    # Do a single forward pass to see what the model predicts
                    forward_output = model(
                        input_ids=input_ids,
                        images=images,
                        image_masks=image_masks,
                        image_input_idx=image_input_idx,
                    )
                    logits = forward_output.logits[0, -1, :]  # Last position, all vocab
                    probs = torch.softmax(logits, dim=-1)
                    
                    # CHECK 1: NaN/Inf in logits or probs
                    has_nan = torch.isnan(probs).any().item()
                    has_inf = torch.isinf(probs).any().item()
                    has_nan_logits = torch.isnan(logits).any().item()
                    has_inf_logits = torch.isinf(logits).any().item()
                    
                    print(f"\n[CHECK 1] NaN/Inf Detection:")
                    print(f"  Logits: min={logits.min().item():.2f}, max={logits.max().item():.2f}")
                    print(f"  Logits: has_nan={has_nan_logits}, has_inf={has_inf_logits}")
                    print(f"  Probs: min={probs.min().item():.6e}, max={probs.max().item():.6e}")
                    print(f"  Probs: has_nan={has_nan}, has_inf={has_inf}")
                    
                    if has_nan_logits or has_inf_logits:
                        print("  ⚠️  WARNING: NaN or Inf detected in logits! Model weights may be corrupted.")
                    
                    # CHECK 2: Top predictions
                    top_probs, top_indices = torch.topk(probs, 10)
                    print(f"\n[CHECK 2] Top 10 Predicted Tokens:")
                    for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        token_text = preprocessor.tokenizer.decode([idx.item()])
                        print(f"  {j+1}. Token {idx.item():5d} ({repr(token_text):20s}): prob={prob.item():.6f}, logit={logits[idx].item():.2f}")
                    
                    # CHECK 3: Token 0 specifically
                    token0_prob = probs[0].item()
                    token0_logit = logits[0].item()
                    print(f"\n[CHECK 3] Token 0 ('!!!') Analysis:")
                    print(f"  Probability: {token0_prob:.6e}")
                    print(f"  Logit: {token0_logit:.2f}")
                    print(f"  Rank: {(probs > token0_prob).sum().item() + 1}")
                    
                    if token0_prob > 0.1:
                        print(f"  ⚠️  WARNING: Token 0 has high probability ({token0_prob:.2%})!")
                    
                    # CHECK 4: Probability distribution sanity
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    max_entropy = np.log(len(probs))
                    normalized_entropy = entropy / max_entropy
                    
                    print(f"\n[CHECK 4] Distribution Analysis:")
                    print(f"  Entropy: {entropy:.2f} (max possible: {max_entropy:.2f})")
                    print(f"  Normalized entropy: {normalized_entropy:.4f}")
                    print(f"  Top-1 confidence: {top_probs[0].item():.6f}")
                    
                    if normalized_entropy > 0.95:
                        print(f"  ⚠️  WARNING: Nearly uniform distribution - model is very confused!")
                    elif normalized_entropy < 0.01:
                        print(f"  ⚠️  WARNING: Very peaked distribution - model is overconfident!")
                    
                    # CHECK 5: Check expected tokens
                    print(f"\n[CHECK 5] Expected Token Probabilities:")
                    expected_tokens = [" is", "is", " on", " Stairs", "Stairs", " left", " right", "Left", "Right", "The", " The"]
                    for tok_text in expected_tokens:
                        try:
                            tok_ids = preprocessor.tokenizer.encode(tok_text, add_special_tokens=False)
                            if len(tok_ids) == 1:
                                tok_id = tok_ids[0]
                                tok_prob = probs[tok_id].item()
                                tok_logit = logits[tok_id].item()
                                rank = (probs > tok_prob).sum().item() + 1
                                print(f"  '{tok_text:10s}' (id={tok_id:5d}): prob={tok_prob:.6e}, logit={tok_logit:6.2f}, rank={rank:4d}")
                        except:
                            pass
                    
                    print(f"\n{'='*80}\n")
                
                # DEBUG: For first 2 images, check what token generation picks
                if args.debug and local_rank == 0 and i < start_idx + 2:
                    print(f"[DEBUG] Generation strategy check for image {i}...")
                    
                    # What should be generated (from forward pass)
                    print("\n  Expected next token (from forward pass logits):")
                    next_token = top_indices[0].item()
                    print(f"    Token: {next_token} ({repr(preprocessor.tokenizer.decode([next_token]))})")
                    print(f"    Probability: {top_probs[0].item():.6f}")
                    print()
                
                
                output = model.generate(
                    input_ids=input_ids,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    max_steps=args.max_tokens,
                    min_steps=4,
                    is_distributed=True  # Model is wrapped in FSDP, needs distributed=True
                )
                token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                generated_text = preprocessor.tokenizer.decode(token_ids[0])
                
                # Print generated output for debugging
                if args.debug and local_rank == 0:
                    print(f"\n[Image {i}]")
                    print(f"  Generated: {generated_text}")
                    print(f"  GT: {ground_truth}")
        # NO TRY-EXCEPT: Let errors fail loudly per .cursorrules
        # Store output with metadata
        result_entry = {
            "image_idx": i,
            "generated_output": generated_text,
            "ground_truth": ground_truth
        }
        
        # Add prompt for context
        if task_type == "left_right" or task_type == "spatial" or task_type == "top_bottom":
            result_entry["prompt"] = prompt
            if message_list:
                result_entry["label"] = first_msg["label"]  # No .get() - fail loudly if missing
                result_entry["position"] = first_msg["position"]  # No .get() - fail loudly if missing
                # For lenient evaluation: save horizontal and vertical positions
                # Note: top_bottom task only has vertical_position, not horizontal
                if task_type == "top_bottom":
                    result_entry["vertical_position"] = first_msg["vertical_position"]
                else:
                    result_entry["horizontal_position"] = first_msg["horizontal_position"]
                    result_entry["vertical_position"] = first_msg["vertical_position"]
                result_entry["image_path"] = example_data["image"]
        elif task_type == "pointing":
            result_entry["prompt"] = prompt
            if message_list:
                result_entry["label"] = first_msg["label"]  # No .get() - fail loudly if missing
                result_entry["style"] = first_msg["style"]  # No .get() - fail loudly if missing
        
        local_captions.append(result_entry)
    
    # Wait for all processes to finish
    dist.barrier()
    
    # Gather all captions on rank 0
    if local_rank == 0:
        print("Gathering captions from all processes...")
        gathered_captions = [None] * world_size
    else:
        gathered_captions = None
    
    # Gather objects from all ranks to rank 0
    dist.gather_object(local_captions, gathered_captions if local_rank == 0 else None, dst=0)
    
    # Only rank 0 processes the gathered data
    if local_rank == 0:
        all_captions = []
        for rank_captions in gathered_captions:
            all_captions.extend(rank_captions)
        
        # Sort by image index
        all_captions.sort(key=lambda x: x["image_idx"])
        
        # Save to JSON
        ckpt_name = checkpoint_path.rstrip("/").split("/")[-2] + "_" + checkpoint_path.rstrip("/").split("/")[-1]
        output_dir = Path(args.output_base_dir) / ckpt_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "generated_captions.json"
        
        result = {
            "checkpoint": checkpoint_path,
            "num_images": num_val_images,
            "max_tokens": args.max_tokens,
            "split": "validation",
            "task_type": task_type,
            "dataset": dataset_name,
            "outputs": all_captions
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Saved {len(all_captions)} captions to {output_file}")
        
        # Calculate accuracy for spatial tasks (left_right, spatial, or top_bottom)
        if task_type == "left_right" or task_type == "spatial" or task_type == "top_bottom":
            correct = 0
            total = 0
            # For spatial task, also track breakdown by direction
            if task_type == "spatial":
                direction_stats = {"left": {"correct": 0, "total": 0}, 
                                 "right": {"correct": 0, "total": 0},
                                 "top": {"correct": 0, "total": 0}, 
                                 "bottom": {"correct": 0, "total": 0}}
            
            for entry in all_captions:
                # Fail loudly if position is missing (per .cursorrules)
                if "position" not in entry:
                    raise KeyError(f"Missing 'position' field in entry {entry['image_idx']}. This should not happen!")
                if "generated_output" not in entry:
                    raise KeyError(f"Missing 'generated_output' field in entry {entry['image_idx']}. This should not happen!")
                
                total += 1
                gt_position = entry["position"].lower()
                generated = entry["generated_output"].lower()
                
                # Lenient evaluation: object can be in corner (e.g., bottom-left)
                # Get both horizontal and vertical classifications
                h_pos = entry["horizontal_position"] if "horizontal_position" in entry else None
                v_pos = entry["vertical_position"] if "vertical_position" in entry else None
                
                # Accept if generated matches EITHER horizontal OR vertical position
                # Example: object in bottom-left corner (h_pos='left', v_pos='bottom')
                #   - If model says "left" → correct
                #   - If model says "bottom" → correct
                #   - If model says "right" or "top" → wrong
                is_correct = False
                valid_answers = []
                if h_pos in ['left', 'right']:
                    valid_answers.append(h_pos)
                if v_pos in ['top', 'bottom']:
                    valid_answers.append(v_pos)
                
                # Check if any valid answer appears in generated text
                for valid_ans in valid_answers:
                    if valid_ans in generated:
                        is_correct = True
                        break
                
                if is_correct:
                    correct += 1
                
                # Track per-direction stats for spatial task
                if task_type == "spatial" and gt_position in direction_stats:
                    direction_stats[gt_position]["total"] += 1
                    if is_correct:
                        direction_stats[gt_position]["correct"] += 1
            
            if total > 0:
                accuracy = correct / total * 100
                print(f"\n{'='*60}")
                if task_type == "spatial":
                    print(f"SPATIAL ACCURACY: {correct}/{total} = {accuracy:.2f}%")
                    # Calculate random chance from actual data distribution
                    # For each example, count how many valid answers it has
                    total_random_prob = 0
                    for entry in all_captions:
                        if "position" in entry:
                            h_pos = entry.get("horizontal_position")
                            v_pos = entry.get("vertical_position")
                            num_valid = 0
                            if h_pos in ['left', 'right']:
                                num_valid += 1
                            if v_pos in ['top', 'bottom']:
                                num_valid += 1
                            # Random chance for this example: num_valid / 4 (uniform random over 4 directions)
                            total_random_prob += (num_valid / 4.0)
                    
                    random_chance = (total_random_prob / total * 100) if total > 0 else 0
                    improvement = accuracy - random_chance
                    print(f"Random chance (lenient eval): {random_chance:.2f}%")
                    print(f"Improvement over random: {improvement:+.2f}%")
                    print(f"{'='*60}")
                    print(f"Per-direction breakdown:")
                    for direction in ["left", "right", "top", "bottom"]:
                        stats = direction_stats[direction]
                        if stats["total"] > 0:
                            dir_acc = stats["correct"] / stats["total"] * 100
                            print(f"  {direction.upper():6s}: {stats['correct']:3d}/{stats['total']:3d} = {dir_acc:5.1f}%")
                elif task_type == "top_bottom":
                    print(f"TOP/BOTTOM ACCURACY: {correct}/{total} = {accuracy:.2f}%")
                    # Count actual distribution
                    top_count = sum(1 for e in all_captions if e.get("vertical_position") == "top")
                    bottom_count = sum(1 for e in all_captions if e.get("vertical_position") == "bottom")
                    print(f"Data distribution: {top_count} top, {bottom_count} bottom")
                    # Random baseline: if balanced 50/50, random chance = 50%
                    # If imbalanced, random chance depends on distribution
                    random_chance = ((top_count/total)**2 + (bottom_count/total)**2) * 100 if total > 0 else 0
                    improvement = accuracy - random_chance
                    print(f"Random chance (actual distribution): {random_chance:.2f}%")
                    print(f"Improvement over random: {improvement:+.2f}%")
                else:
                    print(f"LEFT/RIGHT ACCURACY: {correct}/{total} = {accuracy:.2f}%")
                    # Calculate random chance from actual data for left/right task too
                    total_random_prob = 0
                    for entry in all_captions:
                        if "position" in entry:
                            h_pos = entry.get("horizontal_position")
                            v_pos = entry.get("vertical_position")
                            num_valid = 0
                            if h_pos in ['left', 'right']:
                                num_valid += 1
                            if v_pos in ['top', 'bottom']:
                                num_valid += 1
                            total_random_prob += (num_valid / 4.0)
                    
                    random_chance = (total_random_prob / total * 100) if total > 0 else 0
                    improvement = accuracy - random_chance
                    print(f"Random chance (lenient eval): {random_chance:.2f}%")
                    print(f"Improvement over random: {improvement:+.2f}%")
                print(f"{'='*60}\n")
        
        print("Done!")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

