"""Visualize left/right training data examples.

Usage:
    python scripts/visualize_left_right_training_data.py --output-dir analysis_results/training_data_viz --num-examples 50
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from olmo.data.pixmo_datasets import PixMoPointsLeftRight
from olmo.data.data_formatter import DataFormatter
from olmo.data.model_preprocessor import load_image


def main():
    parser = argparse.ArgumentParser(description="Visualize left/right training data")
    parser.add_argument("--output-dir", type=str, default="analysis_results/training_data_viz", 
                       help="Directory to save visualizations")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to visualize")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], 
                       help="Which split to visualize")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = PixMoPointsLeftRight(split=args.split, kind="basic")
    formatter = DataFormatter()
    
    # Generate visualizations
    print(f"Generating {args.num_examples} visualizations...")
    
    for i in range(args.start_idx, args.start_idx + args.num_examples):
        if i >= len(dataset):
            print(f"Reached end of dataset at index {i}")
            break
        
        # Get example
        rng = np.random.RandomState(42 + i)
        example_data = dataset.get(i, rng)
        
        # Get first message
        message_list = example_data.get("message_list", [])
        if not message_list:
            print(f"Skipping example {i}: no messages")
            continue
        
        first_msg = message_list[0]
        label = first_msg.get("label", "unknown")
        position = first_msg.get("position", "unknown")
        
        # Load image
        image_path = example_data["image"]
        if not Path(image_path).exists():
            print(f"Skipping example {i}: image not found")
            continue
        
        img = Image.open(image_path)
        
        # Generate ground truth text
        image_array = load_image(image_path)
        msg_with_image = first_msg.copy()
        msg_with_image["image"] = image_array
        ground_truth = formatter.format_left_right(msg_with_image)
        
        # Create prompt
        prompt = f"Where is {label}?"
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(img)
        ax.axis('off')
        
        # Title with training info
        title = f"Training Example {i} ({args.split} split)\n"
        title += f"Prompt: {prompt}\n"
        title += f"Label: {label}\n"
        title += f"Position: {position}\n"
        title += f"Ground Truth Output: {ground_truth}"
        
        ax.set_title(title, fontsize=14, color='blue', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f"{args.split}_example_{i:04d}_{position}_{label.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (i - args.start_idx + 1) % 10 == 0:
            print(f"  Generated {i - args.start_idx + 1}/{args.num_examples} images...")
    
    print(f"\n✓ Saved visualizations to {output_dir}")
    print(f"  Files named: {args.split}_example_XXXX_<position>_<label>.png")


if __name__ == "__main__":
    main()

