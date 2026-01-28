"""Visualize spatial task results with images (left/right/top/bottom).

Usage:
    python scripts/visualize_left_right_results.py --results-json analysis_results/captions/.../generated_captions.json
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from olmo.data.pixmo_datasets import PixMoPointsSpatial, PixMoPointsLeftRight


def main():
    parser = argparse.ArgumentParser(description="Visualize left/right predictions")
    parser.add_argument("--results-json", type=str, required=True, help="Path to generated_captions.json")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to show")
    parser.add_argument("--show-correct-only", action="store_true", help="Only show correct predictions")
    parser.add_argument("--show-incorrect-only", action="store_true", help="Only show incorrect predictions")
    args = parser.parse_args()
    
    # Load results
    with open(args.results_json, 'r') as f:
        data = json.load(f)
    
    outputs = data["outputs"]
    task_type = data.get("task_type", "unknown")
    
    # Load dataset to get images
    print(f"Loading dataset (task: {task_type})...")
    if task_type == "spatial":
        dataset = PixMoPointsSpatial(split="validation", kind="basic")
    elif task_type == "left_right":
        dataset = PixMoPointsLeftRight(split="validation", kind="basic")
    else:
        print(f"Warning: Unknown task type '{task_type}', assuming spatial")
        dataset = PixMoPointsSpatial(split="validation", kind="basic")
    
    # Filter if requested
    if args.show_correct_only or args.show_incorrect_only:
        filtered_outputs = []
        for entry in outputs:
            # Extract GT position from ground_truth text if position field not available
            gt_position = entry.get("position", "").lower()
            if not gt_position and "ground_truth" in entry:
                gt_text = entry["ground_truth"].lower()
                for pos in ["left", "right", "top", "bottom"]:
                    if pos in gt_text:
                        gt_position = pos
                        break
            
            generated = entry.get("generated_output", "").lower()
            is_correct = gt_position in generated
            
            if (args.show_correct_only and is_correct) or (args.show_incorrect_only and not is_correct):
                filtered_outputs.append(entry)
        outputs = filtered_outputs
        print(f"Filtered to {len(outputs)} examples")
    
    # Create output directory next to JSON file
    results_path = Path(args.results_json)
    output_dir = results_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    end_idx = min(args.start_idx + args.num_examples, len(outputs))
    
    print(f"Generating {end_idx - args.start_idx} visualizations...")
    
    for i in range(args.start_idx, end_idx):
        entry = outputs[i]
        img_idx = entry.get("image_idx", i)
        
        # Load image from dataset
        rng = np.random.RandomState(42 + img_idx)
        example_data = dataset.get(img_idx, rng)
        image_path = example_data["image"]
        
        if not Path(image_path).exists():
            print(f"Skipping example {img_idx}: image not found at {image_path}")
            continue
        
        img = Image.open(image_path)
        
        # Get label from dataset (not from JSON which doesn't have it)
        message_list = example_data.get("message_list", [])
        if message_list:
            label = message_list[0].get("label", "unknown")
        else:
            label = "unknown"
        gt_position = entry.get("position", "")
        
        # Extract GT position from ground_truth text if position field not available
        if not gt_position and "ground_truth" in entry:
            gt_text = entry["ground_truth"].lower()
            for pos in ["left", "right", "top", "bottom"]:
                if pos in gt_text:
                    gt_position = pos
                    break
        
        if not gt_position:
            gt_position = "unknown"
        
        generated = entry.get("generated_output", "")
        prompt = entry.get("prompt", f"Where is {label}?")
        img_idx = entry.get("image_idx", i)
        
        # Check if correct
        is_correct = gt_position.lower() in generated.lower()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(img)
        ax.axis('off')
        
        # Title with result
        color = 'green' if is_correct else 'red'
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        title = f"Example {img_idx}: {status}\n"
        title += f"Prompt: {prompt}\n"
        title += f"Ground Truth: {gt_position}\n"
        title += f"Generated: {generated}"
        
        ax.set_title(title, fontsize=14, color=color, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save with descriptive filename
        status_str = "correct" if is_correct else "incorrect"
        output_file = output_dir / f"example_{img_idx:04d}_{status_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (i - args.start_idx + 1) % 10 == 0:
            print(f"  Generated {i - args.start_idx + 1}/{end_idx - args.start_idx} images...")
    
    print(f"\n✓ Saved {end_idx - args.start_idx} visualizations to {output_dir}")
    print(f"  Correct examples: example_XXXX_correct.png")
    print(f"  Incorrect examples: example_XXXX_incorrect.png")


if __name__ == "__main__":
    main()

