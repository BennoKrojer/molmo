"""
Explore a simplified version of PixMo Points for left/right classification.

This script:
1. Loads PixMo Points examples
2. Filters to only examples with objects present
3. Classifies objects as left (x < 40) or right (x > 60), skipping middle zone
4. Creates simple prompt/answer pairs: "Where is {object}?" -> "{Object} is on the left/right of the image."
5. Visualizes images with zone markings and object locations

Usage:
    python scripts/explore_simple_leftright.py --num-examples 20 --output-dir explore_leftright_output
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

from olmo.data.pixmo_datasets import PixMoPoints
from olmo.data.model_preprocessor import load_image


def classify_position(points):
    """
    Classify object position as 'left', 'right', or 'middle'.
    
    Args:
        points: Array of (x, y) coordinates in 0-100 scale
        
    Returns:
        'left' if all points have x < 33.33
        'right' if all points have x > 66.67
        'middle' otherwise (we skip these)
    """
    if len(points) == 0:
        return None
    
    # Get mean x coordinate across all points
    mean_x = np.mean(points[:, 0])
    
    if mean_x < 33.33:
        return 'left'
    elif mean_x > 66.67:
        return 'right'
    else:
        return 'middle'


def visualize_example(example_data, output_path, example_idx):
    """Visualize a single training example with zone markings."""
    
    # Load image
    image_array = load_image(example_data['image'])
    image = Image.fromarray(image_array)
    
    # Create a copy for drawing
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    h, w = image_array.shape[:2]
    
    # Draw zone boundaries (33.33% and 66.67% lines)
    left_boundary = int(w * 0.3333)
    right_boundary = int(w * 0.6667)
    
    # Draw vertical lines for zones
    draw.line([(left_boundary, 0), (left_boundary, h)], fill='yellow', width=3)
    draw.line([(right_boundary, 0), (right_boundary, h)], fill='yellow', width=3)
    
    # Add zone labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((left_boundary // 2 - 30, 20), "LEFT", fill='yellow', font=font, stroke_width=2, stroke_fill='black')
    draw.text((right_boundary + (w - right_boundary) // 2 - 40, 20), "RIGHT", fill='yellow', font=font, stroke_width=2, stroke_fill='black')
    draw.text(((left_boundary + right_boundary) // 2 - 40, 20), "MIDDLE", fill='red', font=font, stroke_width=2, stroke_fill='black')
    
    # Process messages
    messages = example_data['message_list']
    
    all_text_info = []
    valid_examples = []
    
    for msg_idx, msg in enumerate(messages):
        label = msg.get('label', 'unknown')
        points = msg.get('points', np.array([]))
        
        if len(points) == 0:
            continue  # Skip examples with no objects
        
        position = classify_position(points)
        
        if position == 'middle':
            continue  # Skip middle zone
        
        # Create prompt and answer
        prompt = f"Where is {label}?"
        answer = f"{label.capitalize()} is on the {position} of the image."
        
        all_text_info.append(f"[Example {len(valid_examples) + 1}]")
        all_text_info.append(f"Label: {label}")
        all_text_info.append(f"Num points: {len(points)}")
        all_text_info.append(f"Mean X coordinate: {np.mean(points[:, 0]):.1f}")
        all_text_info.append(f"Position: {position.upper()}")
        all_text_info.append(f"Prompt: {prompt}")
        all_text_info.append(f"Answer: {answer}")
        all_text_info.append("")
        
        valid_examples.append({
            'label': label,
            'position': position,
            'prompt': prompt,
            'answer': answer,
            'num_points': len(points)
        })
        
        # Draw points on image
        for point in points:
            x, y = point
            # Convert from normalized coordinates (0-100 scale) to pixel coordinates
            px = int(x * w / 100)
            py = int(y * h / 100)
            
            # Draw a colored dot (green for left, blue for right)
            color = 'green' if position == 'left' else 'blue'
            radius = 10
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], 
                       fill=color, outline='yellow', width=3)
            
            # Draw a small cross in the center for precision
            cross_size = 4
            draw.line([px - cross_size, py, px + cross_size, py], fill='yellow', width=2)
            draw.line([px, py - cross_size, px, py + cross_size], fill='yellow', width=2)
    
    # If no valid examples, return None
    if len(valid_examples) == 0:
        return None, None
    
    # Add text summary at the bottom
    text_summary = "\n".join(all_text_info)
    
    # Create a figure with text and image
    text_height = max(400, len(all_text_info) * 20)
    canvas = Image.new('RGB', (max(image.width, 1200), image.height + text_height), 'white')
    
    # Paste the annotated image
    canvas.paste(annotated_image, (0, 0))
    
    # Draw text at the bottom
    draw_canvas = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    y_offset = image.height + 10
    for line in all_text_info:
        draw_canvas.text((10, y_offset), line, fill='black', font=font)
        y_offset += 20
    
    # Save
    canvas.save(output_path)
    
    return all_text_info, valid_examples


def main():
    parser = argparse.ArgumentParser(description="Explore simplified left/right pointing data")
    parser.add_argument("--num-examples", type=int, default=20, 
                       help="Number of examples to visualize")
    parser.add_argument("--output-dir", type=str, default="explore_leftright_output",
                       help="Directory to save visualizations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                       help="Which split to explore")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PixMo Points dataset...")
    print(f"  Split: {args.split}")
    
    # Load dataset - use basic pointing data (not counting)
    dataset = PixMoPoints(split=args.split, kind="basic", counting=False)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample random examples
    rng = np.random.RandomState(args.seed)
    
    # Collect statistics
    stats = {
        "total_dataset_examples": 0,
        "total_messages_seen": 0,
        "messages_with_objects": 0,
        "messages_without_objects": 0,
        "left_examples": 0,
        "right_examples": 0,
        "middle_examples_skipped": 0,
        "valid_examples_generated": 0,
    }
    
    all_summaries = []
    
    print(f"\nProcessing examples...")
    print(f"Looking for {args.num_examples} images with valid left/right examples...")
    
    valid_count = 0
    attempts = 0
    max_attempts = args.num_examples * 20  # Try up to 20x the requested examples
    
    while valid_count < args.num_examples and attempts < max_attempts:
        # Get a random example
        idx = rng.randint(0, len(dataset))
        example_data = dataset.get(idx, rng)
        attempts += 1
        stats["total_dataset_examples"] += 1
        
        # Check if this example has any valid left/right instances
        has_valid = False
        for msg in example_data['message_list']:
            stats["total_messages_seen"] += 1
            points = msg.get('points', np.array([]))
            if len(points) == 0:
                stats["messages_without_objects"] += 1
                continue
            stats["messages_with_objects"] += 1
            position = classify_position(points)
            if position == 'left':
                stats["left_examples"] += 1
                has_valid = True
            elif position == 'right':
                stats["right_examples"] += 1
                has_valid = True
            elif position == 'middle':
                stats["middle_examples_skipped"] += 1
        
        if not has_valid:
            continue
        
        # Visualize and save
        output_path = output_dir / f"example_{valid_count:03d}_idx{idx}.jpg"
        text_info, valid_examples = visualize_example(example_data, output_path, valid_count)
        
        if valid_examples is None or len(valid_examples) == 0:
            continue
        
        stats["valid_examples_generated"] += len(valid_examples)
        
        all_summaries.append({
            "example_idx": valid_count,
            "dataset_idx": idx,
            "text_info": text_info,
            "valid_examples": valid_examples,
            "image_path": str(output_path)
        })
        
        valid_count += 1
        
        if valid_count % 5 == 0:
            print(f"  Found {valid_count}/{args.num_examples} valid images (tried {attempts} dataset examples)...")
    
    if valid_count < args.num_examples:
        print(f"\nWarning: Only found {valid_count} valid images after {attempts} attempts")
    
    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "statistics": stats,
            "examples": all_summaries
        }, f, indent=2)
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Dataset examples examined: {stats['total_dataset_examples']}")
    print(f"Total messages seen: {stats['total_messages_seen']}")
    print(f"Messages with objects: {stats['messages_with_objects']}")
    print(f"Messages without objects: {stats['messages_without_objects']}")
    print(f"\nPosition distribution:")
    print(f"  LEFT examples: {stats['left_examples']}")
    print(f"  RIGHT examples: {stats['right_examples']}")
    print(f"  MIDDLE examples (skipped): {stats['middle_examples_skipped']}")
    print(f"\nValid examples generated: {stats['valid_examples_generated']}")
    
    if stats['messages_with_objects'] > 0:
        left_pct = 100 * stats['left_examples'] / stats['messages_with_objects']
        right_pct = 100 * stats['right_examples'] / stats['messages_with_objects']
        middle_pct = 100 * stats['middle_examples_skipped'] / stats['messages_with_objects']
        print(f"\nPosition percentages (of messages with objects):")
        print(f"  LEFT: {left_pct:.1f}%")
        print(f"  RIGHT: {right_pct:.1f}%")
        print(f"  MIDDLE (skipped): {middle_pct:.1f}%")
    
    print(f"\n✓ Saved {valid_count} visualizations to {output_dir}/")
    print(f"✓ Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

